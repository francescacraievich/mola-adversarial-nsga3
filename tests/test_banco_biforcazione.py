"""Bifurcation bench: extensions of --g-banco.

Previous spec unchanged (same fields in banco_G.json), real init, salvata_banco and
restore, offsets in the robot frame, attitude, zones and meter in the spec, branch outcome
on real lines of a full run (with TF_OLD_DATA warnings that split lines), comparisons with
the acceptance definitions, offset plans, fan and configurations. Fake objects, no ROS
node, Isaac or MOLA; parts that use run data are skipped if the run is missing."""

import csv
import importlib.util
import json
import math
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.optimization import attack_orchestrator as ao  # noqa: E402
from src.optimization import banco_g  # noqa: E402
from src.optimization import window_record as wrec  # noqa: E402
from src.analysis.baseline_distance import load_baseline  # noqa: E402
from src.analysis.path_deviation import true_path  # noqa: E402
from src.perturbations import zone_attaccante as za  # noqa: E402

tg = pytest.importorskip("test_banco_G_d")

DATI = ROOT / "tests" / "dati" / "biforcazione_31ott"
REP = (ROOT / "data/attack/serieB/nsga3_senzabudget/s1001_accettazione_30ott/campagna/straight/"
       "nsga3_g3_p32x16_lag0.5_rkeep_sel2_warm_stopramp_k1.0_reliable_serieB_s1001_pt1_opinsieme_sgross_"
       "coda_N5_dmin140/rep_01")
BASELINE = ROOT / "data/attack/baseline/rettilineo_direction_N5_media3semi_dmin140_sosta1800.csv"
serve_run = pytest.mark.skipif(not (REP / "history.json").is_file() or not BASELINE.is_file(),
                               reason="run della prova di accettazione assente")

# fields of the banco_G.json record with the previous spec
CAMPI_VECCHI = ["r", "caso", "braccio", "tag", "init", "target_scene", "mola_env", "genome", "drift_init",
                "saved_bench", "saved_scene_bench", "saved_err_cm", "saved_scene_err_cm", "status", "travelled",
                "frames", "end", "mola_log", "esito", "fp"]


def _carica_assetto_isaac():
    """isaac/assetto.py as the server loads it (exec from the file, no .pyc)."""
    p = ROOT / "isaac" / "assetto.py"
    m = types.ModuleType("assetto_isaac")
    m.__file__ = str(p)
    exec(compile(p.read_text(), str(p), "exec"), m.__dict__)
    return m


def _spec_nuova(tmp_path, **braccio):
    s = tg._spec(tmp_path)
    s["casi"][0]["bracci"] = [{"nome": "X", "init": "vera", **braccio}]
    return s


def _genoma():
    return [0.1] * 17


# --- previous spec unchanged -----------------------------------------------------------

def test_specifica_vecchia_invariata(tmp_path):
    s = banco_g.check_spec(tg._spec(tmp_path))
    assert not banco_g.usa_estensioni(s)
    assert banco_g.schedule(s) == [(r, c, b) for r in range(s["ripetizioni"]) for c in s["casi"] for b in c["bracci"]]
    a = banco_g.init_target(s["casi"][0], s["casi"][0]["bracci"][0])
    assert list(a) == ["target_odom", "target_scene", "v_set_scene", "w_set", "zero_joints", "rule", "stimatore"]
    assert a["target_scene"] == s["casi"][0]["attacker_target_scene"] and a["stimatore"] == "attaccante"
    c = banco_g.init_target(s["casi"][0], s["casi"][0]["bracci"][1])
    assert c["stimatore"] == "ombra" and "scarti" not in c and "assetto" not in c
    assert banco_g.check_args(tg._args(g_banco=None)) is None


def test_ciclo_vecchio_stessi_campi(tmp_path, monkeypatch):
    """Previous spec: same Isaac client at rollout, same fields in the record, no new key
    in banco_G.json, ZONE unchanged."""
    s = tg._spec(tmp_path)
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(s))
    out = tmp_path / "rep_01"
    out.mkdir()
    isaac = tg._IsaacBanco()
    visti = []

    def _rollout(genome, isa, mola, node, goal, horizon, tag, **kw):
        visti.append(isa)
        log = tmp_path / f"mola_{tag}.log"
        log.write_text(tg.RIGHE)
        mola.log_path = str(log)
        return None, (1.0, 0.0, 0.0), 0.0, "ok", 1.0, 10

    monkeypatch.setattr(ao, "rollout", _rollout)
    monkeypatch.setattr(ao, "TWIN_STATE", {})
    zone_prima = dict(ao.ZONE)
    args = types.SimpleNamespace(g_banco=str(p), g_overlay="/ovl", g_resave="ogni", horizon=1.0, settle_sec=2.0,
                                 catchup_sec=0.8, attack_from="spawn", stop_on="estimate",
                                 drift_reset_on="genome", trace=True)
    assert ao._g_banco_run(args, isaac, types.SimpleNamespace(), tg._NodeBanco(), out, [(0.0, 0.0)]) == 0
    assert visti and all(v is isaac for v in visti)
    d = json.loads((out / "banco_G.json").read_text())
    assert list(d) == ["spec", "esiti"]
    assert all(list(e) == CAMPI_VECCHI for e in d["esiti"])
    assert not (out / "zone").exists() and dict(ao.ZONE) == zone_prima


# --- new inits and offsets in the robot frame ------------------------------------------

def test_init_nuovi(tmp_path):
    s = tg._spec(tmp_path)
    c = banco_g.check_spec(s)["casi"][0]
    v = banco_g.init_target(c, {"nome": "V0", "init": "vera"})
    assert v["target_scene"] == c["saved_scene"] and v["zero_joints"] and v["stimatore"] is None
    sb = banco_g.init_target(c, {"nome": "Rs", "init": "salvata_banco"}, [1.0, 2.0, 0.3])
    assert sb["target_scene"] == [1.0, 2.0, 0.3]
    with pytest.raises(ValueError):
        banco_g.init_target(c, {"nome": "Rs", "init": "salvata_banco"})
    r = banco_g.init_target(c, {"nome": "R", "init": "ripristino"})
    assert r["target_scene"] is None and "set_state" in r["rule"]


def test_scarti_riferimento_robot():
    # heading 90 deg: forward = world +y, side (left) = -x
    p = banco_g.compone_offset([1.0, 2.0, math.pi / 2], avanti_mm=2.0)
    assert p[0] == pytest.approx(1.0, abs=1e-12) and p[1] == pytest.approx(2.002, abs=1e-12)
    p = banco_g.compone_offset([1.0, 2.0, math.pi / 2], lato_mm=2.0, yaw_deg=0.03)
    assert p[0] == pytest.approx(0.998, abs=1e-12) and p[1] == pytest.approx(2.0, abs=1e-12)
    assert p[2] == pytest.approx(math.pi / 2 + math.radians(0.03), abs=1e-15)
    rng = np.random.default_rng(7)
    for _ in range(50):
        base = [*rng.uniform(-5, 5, 2), rng.uniform(-3, 3)]
        off = [*rng.uniform(-10, 10, 2), rng.uniform(-0.2, 0.2)]
        assert banco_g.offset_relativo(base, banco_g.compone_offset(base, *off)) == pytest.approx(off, abs=1e-9)
    # attacker estimate vs the true pose saved at w4 of a recorded run
    # (history[4].shadow_state.lab_true_saved and twin_init_target.target_odom)
    vera = [4.131251428124246, 0.6101813958607295, 0.16127870603641245]
    att = [4.132814574400111, 0.6084957736307892, 0.16066467778426982]
    off = banco_g.offset_relativo(vera, att)
    assert off[0] == pytest.approx(1.272, abs=5e-4) and off[1] == pytest.approx(-1.915, abs=5e-4)
    assert off[2] == pytest.approx(-0.0352, abs=5e-5)
    assert banco_g.compone_offset(vera, *off)[:2] == pytest.approx(att[:2], abs=1e-9)


def test_scarti_e_assetto_nel_bersaglio(tmp_path):
    s = _spec_nuova(tmp_path, lato_mm=-2.0, assetto={"modo": "conservato"})
    c = banco_g.check_spec(s)["casi"][0]
    t = banco_g.init_target(c, c["bracci"][0])
    assert t["target_scene_senza_scarti"] == c["saved_scene"] and t["scarti"] == {"lato_mm": -2.0}
    assert t["target_scene"] == pytest.approx(banco_g.compone_offset(c["saved_scene"], lato_mm=-2.0))
    assert t["assetto"] == {"modo": "conservato"} and banco_g.usa_estensioni(s)
    s = tg._spec(tmp_path)
    s["casi"][0]["bracci"][0]["avanti_mm"] = 2.0
    c = banco_g.check_spec(s)["casi"][0]
    t = banco_g.init_target(c, c["bracci"][0])
    assert t["target_scene"] == pytest.approx(banco_g.compone_offset(c["attacker_target_scene"], avanti_mm=2.0))


GUASTI = [
    (dict(init="ripristino", lato_mm=1.0), "scarti"),
    (dict(lato_mm=10.5), "oltre"),
    (dict(yaw_deg=0.3), "oltre"),
    (dict(init="ripristino", assetto={"modo": "conservato"}), "assetto solo con init"),
    (dict(assetto={"modo": "inclinato"}), "assetto ="),
    (dict(assetto={"modo": "esplicito", "roll_rad": 0.001, "pitch_rad": 0.002}), "senza fonte"),
    (dict(assetto={"modo": "esplicito", "roll_rad": 0.17, "pitch_rad": 0.0, "fonte": "x"}), "gradi al posto"),
    (dict(assetto={"modo": "esplicito_relativo", "roll_rad": 0.0, "pitch_rad": 0.0, "fonte": "x"}), "yaw_rad"),
    (dict(assetto={"modo": "conservato", "roll_rad": 0.0}), "non gestite"),
    (dict(ripetizioni=3), "ripetizioni"),
    (dict(init="riallineato"), "riallinea_da"),
    (dict(rampa={}), "rampa"),
    (dict(genome=[0.1] * 17, zone={"path": "x"}), "zone"),
]


@pytest.mark.parametrize("braccio,msg", GUASTI)
def test_spec_estensioni_rifiutate(tmp_path, braccio, msg):
    with pytest.raises(ValueError, match=msg):
        banco_g.check_spec(_spec_nuova(tmp_path, **braccio))


def test_spec_riallineato_e_confronti(tmp_path):
    s = tg._spec(tmp_path)
    g = _genoma()
    s["casi"][0]["bracci"] = [{"nome": "G", "init": "attacker_est", "genome": g},
                              {"nome": "V", "init": "ripristino", "genome": g},
                              {"nome": "Gr", "init": "riallineato", "riallinea_da": "V", "genome": g}]
    s["casi"][0]["confronti"] = [{"nome": "a", "vittima": "V", "gemello": "G"},
                                 {"nome": "b", "vittima": "V", "gemello": "Gr"}]
    banco_g.check_spec(json.loads(json.dumps(s)))
    # reserved: the spec is valid, the arguments reject it (realignment not in the bench yet)
    p = tmp_path / "s.json"
    p.write_text(json.dumps(s))
    assert "D-31c" in banco_g.problema_estensioni(banco_g.load_spec(p), tg._args())
    s2 = json.loads(json.dumps(s))
    s2["casi"][0]["bracci"][2]["genome"] = [0.2] * 17
    with pytest.raises(ValueError, match="diversi"):
        banco_g.check_spec(s2)
    s3 = json.loads(json.dumps(s))
    s3["casi"][0]["bracci"][0]["genome"] = [0.2] * 17
    with pytest.raises(ValueError, match="confronto a"):
        banco_g.check_spec(s3)
    s4 = json.loads(json.dumps(s))
    s4["casi"][0]["confronti"][1]["gemello"] = "Z"
    with pytest.raises(ValueError, match="assenti"):
        banco_g.check_spec(s4)


def test_ripetizioni_per_braccio(tmp_path):
    s = tg._spec(tmp_path)
    s["ripetizioni"] = 3
    s["casi"][0]["bracci"][0]["ripetizioni"] = 1
    s = banco_g.check_spec(s)
    o = [(r, c["nome"], b["nome"]) for r, c, b in banco_g.schedule(s)]
    assert ("guasta", "A") in [(x[1], x[2]) for x in o if x[0] == 0]
    assert all(not (x[1] == "guasta" and x[2] == "A") for x in o if x[0] > 0)
    assert len(o) == 3 * 4 - 2


def test_argomenti_estensioni(tmp_path):
    def ragione(spec, **kw):
        p = tmp_path / "s.json"
        p.write_text(json.dumps(spec))
        return banco_g.problema_estensioni(banco_g.load_spec(p), tg._args(**kw))
    s = _spec_nuova(tmp_path, rampa={"tau_s": 0.5})
    assert "rampa" in ragione(s)
    s = tg._spec(tmp_path)
    s["casi"][0]["bracci"][0]["genome"] = _genoma()
    assert "zone salvate" in ragione(s, zone_attaccante=True, metro_al_nodo=True)
    s["casi"][0]["drift_reset_on"] = "keep"
    assert "drift-reset-on keep" in ragione(s, drift_reset_on="genome")
    assert ragione(s, drift_reset_on="keep") is None
    assert "ogni caso" in ragione(s, drift_reset_on="keep", metro_al_nodo=True)
    s = _spec_nuova(tmp_path, assetto={"modo": "esplicito_relativo", "roll_rad": 0.0, "pitch_rad": 0.0,
                                       "yaw_rad": 0.1, "fonte": "log"})
    assert "--start-pose" in ragione(s)
    assert ragione(s, start_pose="-5.99,-1.0,0") is None
    s = tg._spec(tmp_path)
    s["casi"][0]["bracci"] = [{"nome": "G", "init": "attacker_est"}, {"nome": "V", "init": "ripristino"}]
    s["casi"][0]["confronti"] = [{"nome": "a", "vittima": "V", "gemello": "G"}]
    assert "baseline" in ragione(s)
    assert ragione(s, damage_ref="baseline", baseline_path=str(BASELINE)) is None
    assert banco_g.problema_capacita(banco_g.check_spec(_spec_nuova(tmp_path, assetto={"modo": "conservato"})),
                                     ["set_state"]) is not None
    assert banco_g.problema_capacita(banco_g.check_spec(_spec_nuova(tmp_path)), ["set_state"]) is None


# --- attitude ---------------------------------------------------------------------------

def test_assetto_parametri_set_state():
    assert banco_g.parametri_set_state(None) == {}
    assert banco_g.parametri_set_state({"modo": "azzerato"}) == {}
    assert banco_g.parametri_set_state({"modo": "conservato"}) == {"keep_tilt": True}
    assert banco_g.parametri_set_state({"modo": "esplicito", "roll_rad": 1e-3, "pitch_rad": -2e-3}) == \
        {"roll": 1e-3, "pitch": -2e-3}
    rel = {"modo": "esplicito_relativo", "roll_rad": 4.97e-4, "pitch_rad": -2.898e-3, "yaw_rad": 0.1607}
    with pytest.raises(ValueError):
        banco_g.parametri_set_state(rel)
    # spawn without tilt: relative roll and pitch remain, for any yaw
    p = banco_g.parametri_set_state(rel, (0.0, 0.0, 0.7))
    assert p["roll"] == pytest.approx(4.97e-4, abs=1e-12) and p["pitch"] == pytest.approx(-2.898e-3, abs=1e-12)


def test_assetto_composto_come_isaac():
    """ZYX matrix composition against the quaternions of isaac/assetto.py, and the server's
    set_state with roll/pitch returning the same angles."""
    ia = _carica_assetto_isaac()
    rng = np.random.default_rng(3)
    for _ in range(100):
        spawn = (*rng.uniform(-0.01, 0.01, 2), rng.uniform(-math.pi, math.pi))
        rel = (*rng.uniform(-0.01, 0.01, 2), rng.uniform(-0.5, 0.5))
        r, p, y = banco_g.assetto_composto(spawn, rel)
        q = ia.prodotto(ia.quat_da_euler(*spawn), ia.quat_da_euler(*rel))
        r2, p2, y2 = ia.euler_da_quat(ia.normalizza(q))
        assert (r, p) == pytest.approx((r2, p2), abs=1e-12)
        assert math.atan2(math.sin(y - y2), math.cos(y - y2)) == pytest.approx(0.0, abs=1e-12)
        qs = ia.quat_set_state({"roll": r, "pitch": p, "yaw": y}, None)
        assert ia.euler_da_quat(qs) == pytest.approx((r, p, y), abs=1e-12)
        assert banco_g.euler_zyx(banco_g.rot_zyx(*rel)) == pytest.approx(rel, abs=1e-12)


def test_assetto_attaccante_dal_log():
    """ICP line of the attacker's MOLA at the stamp of the w4 state (38.6)."""
    t = (DATI / "mola_attacker_38.5-38.6.log").read_text()
    r = banco_g.nuvola_alla_stamp(t, 38.6)
    assert r["pathStep"] == 267 and r["good"] is True and r["q"] == 0.7232
    assert (r["roll_deg"], r["pitch_deg"], r["yaw_deg"]) == (0.028472, -0.166056, 9.206027)
    assert (r["x"], r["y"]) == (4.116833, 0.60854) and r["nvoxels"] == 13241
    assert banco_g.nuvola_alla_stamp(t, 38.7) is None


# --- branch outcome on real run lines ---------------------------------------------------

def test_esito_ramo_gemello_con_avviso():
    t = (DATI / "mola_w4_sel0_38.7-39.6.log").read_text()
    assert t.count("[WARN]") == 3                # three lines split by a warning
    e = banco_g.esito_ramo(t)
    # offset from the run: sel0 -19.23 cm
    assert e["t0"] == 38.7 and e["stamp_y0"] == 38.7 and e["stamp_y1"] == 39.6
    assert e["dy_cm"] == pytest.approx(-19.229, abs=1e-3) and e["classe"] == "ramo_gemello"
    assert [x["nvoxels"] for x in e["prime"][:6]] == [13087, 13061, 13050, 13047, 13051, 13053]
    assert len(e["prime"]) == 10 and e["righe"] == 10
    r39 = next(x for x in e["prime"] if x["stamp"] == 39.1)
    # ICP line split by the warning and rejoined (an earlier parser lost it)
    assert r39["y"] == -0.531007 and r39["q"] == 0.2976 and len(r39["sigma"]) == 6
    assert r39["sigma"][1] == pytest.approx(4.550653e-03)
    # without removing the warnings the same cloud stays without pose
    grezza = banco_g.RX_AVVISO_TF
    try:
        banco_g.RX_AVVISO_TF = __import__("re").compile(r"(?!x)x")
        r39b = next(x for x in banco_g.nuvole_icp(t) if x["stamp"] == 39.1)
    finally:
        banco_g.RX_AVVISO_TF = grezza
    assert r39b["y"] is None


def test_esito_ramo_vittima_da_stamp():
    t = (DATI / "mola_victim_38.6_38.7-39.6.log").read_text()
    e = banco_g.esito_ramo(t, stamp0=38.7)
    assert e["dy_cm"] == pytest.approx(28.850, abs=1e-3) and e["classe"] == "ramo_vittima"
    assert [x["nvoxels"] for x in e["prime"][:6]] == [13157, 13146, 13138, 13138, 13118, 13097]
    # without stamp0 the first cloud of the continuous log is the one at the end of w3
    assert banco_g.esito_ramo(t)["t0"] == 38.6


def test_esito_ramo_posa_mancante():
    t = (DATI / "mola_w4_sel0_38.7-39.6.log").read_text()
    righe = t.split("\n")
    # cloud 39.6 not good: take the nearest published one within 0.2 s (39.5)
    t2 = "\n".join(x.replace("isGood=1", "isGood=0") if "timestamp=1970/01/01,00:00:39.600000" in x else x
                   for x in righe)
    e = banco_g.esito_ramo(t2)
    assert e["stamp_y1"] == 39.5 and e["dy_cm"] is not None
    # no good pose after 39.3: not classifiable
    t3 = "\n".join(x.replace("isGood=1", "isGood=0") if any(f"00:00:39.{d}00000" in x for d in "456789") else x
                   for x in righe)
    e = banco_g.esito_ramo(t3)
    assert e["dy_cm"] is None and e["classe"] == "non_classificabile"
    assert banco_g.esito_ramo("")["classe"] == "non_classificabile"
    assert [banco_g.classe_ramo(v) for v in (10.0, -10.0, 9.99, -9.99, None)] == \
        ["ramo_vittima", "ramo_gemello", "nessun_salto", "nessun_salto", "non_classificabile"]


# --- zones and comparisons --------------------------------------------------------------

def test_esito_zone():
    h = "a" * 64
    assert banco_g.esito_zone(40, 0, h, h, 40) == "ok"
    assert banco_g.esito_zone(39, 1, h, h, 40) == "errore"
    assert banco_g.esito_zone(40, 0, "b" * 64, h, 40) == "errore"
    assert banco_g.esito_zone(0, 0, None, h, 40) == "nessuna_zona_applicata"
    assert banco_g.esito_zone(0, 0, None, h, 0) == "nessuna_nuvola"


@serve_run
@pytest.mark.parametrize("metro,idx,attesi", [
    # tabella_per_metro.csv of the reference outcome (misure_criterio_A.py, uguali.py)
    (4, 480, (1.66, 7.405, 4.574)), (3, 465, (-0.05, 0.130, 0.046))])
def test_confronto_come_criterio_A(metro, idx, attesi):
    poly = load_baseline(BASELINE)
    p0, u = banco_g.asse_baseline(poly)
    tv = list(csv.DictReader(open(REP / "traces" / f"w{metro}_applied.csv")))
    tw = list(csv.DictReader(open(REP / "traces" / f"w{metro}_e{idx}.csv")))
    dv, dw = wrec.danno_metro(true_path(tv), poly), wrec.danno_metro(true_path(tw), poly)
    ramo = {"classe": "nessun_salto", "dy_cm": 0.0}
    cc = banco_g.confronto_coppia(banco_g.grandezze_traccia(tv, p0, u), banco_g.grandezze_traccia(tw, p0, u),
                                  ramo, ramo, dv, dw)
    assert (cc["R1_cm"], cc["laterale_cm"], cc["danno_cm"]) == pytest.approx(attesi, abs=1e-3)
    assert cc["segue"] is (metro == 3) and cc["stesso_ramo"]
    assert cc["oltre"] == ([] if metro == 3 else ["laterale_cm", "danno_cm"])
    cc = banco_g.confronto_coppia(banco_g.grandezze_traccia(tv, p0, u), banco_g.grandezze_traccia(tw, p0, u),
                                  {"classe": "ramo_vittima"}, {"classe": "ramo_gemello"}, dv, None)
    assert not cc["stesso_ramo"] and cc["non_misurati"] == ["danno_cm"] and not cc["segue"]


def test_confronto_tracce_vuote():
    p0, u = np.zeros(2), np.array([1.0, 0.0])
    g = banco_g.grandezze_traccia([], p0, u)
    cc = banco_g.confronto_coppia(g, g, None, None, None, None)
    assert cc["R1_cm"] is None and cc["laterale_cm"] is None and not cc["segue"]


# --- spec plans on run data -------------------------------------------------------------

def _piano(tmp_path, nome, *extra):
    tmp_path.mkdir(parents=True, exist_ok=True)
    out = tmp_path / f"spec_{nome}.json"
    r = subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / "banco_G_d_spec.py"), "--out", str(out),
                        "--piano", nome, "--rep", str(REP), "--metro", "4", *extra],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "NO:" not in r.stdout
    return banco_g.load_spec(out), r.stdout


@serve_run
def test_piano_k1(tmp_path):
    s, txt = _piano(tmp_path, "K1")
    assert "(1 casi, 69 rollout)" in txt and "avanti 1.272 mm, lato -1.915 mm" in txt
    c = s["casi"][0]
    assert c["metro"] == 4 and c["drift_reset_on"] == "keep" and s["ripetizioni"] == 8
    attesi = {n: (i, sc, a, rip) for n, i, sc, a, rip in [
        ("A", "attacker_est", {}, None, 6), ("V0", "vera", {}, None, 8), ("V0t", "vera", {}, "conservato", 6),
        ("At", "attacker_est", {}, "conservato", 6), ("R", "ripristino", {}, None, 5),
        ("Rs", "salvata_banco", {}, "conservato", 5), ("Lp2", "vera", {"avanti_mm": 2.0}, None, 5),
        ("Lm2", "vera", {"avanti_mm": -2.0}, None, 5), ("Tp2", "vera", {"lato_mm": 2.0}, None, 5),
        ("Tm2", "vera", {"lato_mm": -2.0}, None, 5), ("Yp003", "vera", {"yaw_deg": 0.03}, None, 5),
        ("Ym003", "vera", {"yaw_deg": -0.03}, None, 5), ("N", "vera", {}, None, 3)]}
    assert [b["nome"] for b in c["bracci"]] == list(attesi)
    for b in c["bracci"]:
        i, sc, a, rip = attesi[b["nome"]]
        assert b["init"] == i and b["ripetizioni"] == rip and (b.get("assetto") or {}).get("modo") == a
        assert {k: b[k] for k in banco_g.OFFSET_CHIAVI if k in b} == sc
    e480 = c["bracci"][0]
    assert za.genoma_sha(e480["genome"]) == "ebe99146f954b663" and e480["zone"]["hash"].startswith("a0c3330c")
    assert round(100 * math.hypot(*e480["drift_init"][:2]), 3) == 35.306
    n = c["bracci"][-1]
    assert n["genome"] == [-1.0] * 17 and n["drift_init"] == [0.0, 0.0, 0.0]
    assert Path(n["zone"]["path"]).is_file() and Path(n["zone"]["path"]).parent == tmp_path / "spec_K1_zone_calcolate"
    za.carica_zone(n["zone"]["path"], n["zone"]["sha256"], n["zone"]["hash"])


@serve_run
def test_piano_ventaglio(tmp_path):
    s, txt = _piano(tmp_path, "ventaglio")
    assert "(1 casi, 32 rollout)" in txt and s["ripetizioni"] == 1
    br = s["casi"][0]["bracci"]
    assert len(br) == 32 and {b["nome"].split("_")[0] for b in br} == {"e480", "e118", "e428", "e55"}
    assert sum(b["init"] == "vera" for b in br) == 4
    assert all(b["init"] == "attacker_est" for b in br if not b["nome"].endswith("_V"))
    yp = next(b for b in br if b["nome"] == "e118_Yp")
    assert yp["yaw_deg"] == 0.035 and yp["zone"]["genoma_sha"] == za.genoma_sha(yp["genome"])


@serve_run
def test_piano_configurazioni(tmp_path):
    s, txt = _piano(tmp_path, "configurazioni", "--configurazioni", "0,3", "--d31c", "no", "--ripetizioni", "2")
    assert "(2 casi, 8 rollout)" in txt
    c0, c3 = s["casi"]
    assert [(b["nome"], b["init"]) for b in c0["bracci"]] == [("G", "attacker_est"), ("V", "ripristino")]
    assert c0["confronti"] == [{"nome": "a", "vittima": "V", "gemello": "G"}] and "assetto" not in c0["bracci"][0]
    a = c3["bracci"][0]["assetto"]
    assert a["modo"] == "esplicito_relativo" and a["fonte"]["stamp"] == 38.6 and a["fonte"]["pathStep"] == 267
    assert a["roll_rad"] == pytest.approx(math.radians(0.028472)) and a["pitch_rad"] == pytest.approx(math.radians(-0.166056))
    assert a["yaw_rad"] == pytest.approx(math.radians(9.206027))
    s, _ = _piano(tmp_path / "x", "configurazioni", "--configurazioni", "1", "--assetto-attaccante", "esplicito")
    assert [b["init"] for b in s["casi"][0]["bracci"]] == ["attacker_est", "ripristino", "riallineato"]
    r = subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / "banco_G_d_spec.py"), "--out",
                        str(tmp_path / "r.json"), "--piano", "configurazioni", "--rep", str(REP), "--metro", "4",
                        "--configurazioni", "2"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "--rampa" in r.stdout + r.stderr


# --- bench loop with the extensions, fake objects ---------------------------------------

class _IsaacAssetto(tg._IsaacBanco):
    SPAWN = {"roll": 1.2e-3, "pitch": -0.8e-3, "yaw": 0.0}

    def __init__(self):
        super().__init__(caps=("set_state", "set_state_assetto", "assetto"))
        self.comandi = []

    def restore(self):
        self.chiamate.append("restore")
        return list(self.posa)

    def call(self, cmd, **kw):
        self.comandi.append((cmd, kw))
        if cmd == "assetto":
            return {"ok": True, "pos": [0.0, 0.0, 0.0], "quat_wxyz": [1.0, 0.0, 0.0, 0.0], **self.SPAWN,
                    "lin": [0.0] * 3, "ang": [0.0] * 3, "playing": False}
        if cmd == "set_state":
            return {"ok": True, "pose": [kw["x"], kw["y"], kw["yaw"]]}
        raise AssertionError(cmd)


class _NodeZone(tg._NodeBanco):
    def __init__(self):
        self.node_zone_on = True
        self.node_zone_ricevute = {}
        self.node_zone_applicate = {"gemello": {"n_ok": 0, "n_err": 0, "hash": None, "stato": None}}
        self.metri = []
        self.last_trace = []

    def set_twin_drift_init(self, d):
        return True

    def set_metro(self, flusso):
        self.metri.append((flusso, ao.TWIN_STATE.get("window")))
        return True

    def _zone_pub(self, topic):
        return topic, types.SimpleNamespace

    def _publish_confirmed(self, pub, msg, confirmed, what, **kw):
        d = json.loads(msg.data)
        z = za.carica_zone(d["path"], d["sha256"], d["hash"])
        self.node_zone_ricevute[d["flusso"]] = {"seq": d["seq"], "stato": "ok", "hash": z.hash}
        return confirmed()


@serve_run
def test_ciclo_banco_estensioni_finto(tmp_path, monkeypatch):
    s, _ = _piano(tmp_path, "configurazioni", "--configurazioni", "0,3", "--d31c", "no", "--ripetizioni", "2")
    p = tmp_path / "spec_configurazioni.json"
    out = tmp_path / "rep_01"
    out.mkdir()
    tracce = {n: list(csv.DictReader(open(REP / "traces" / f"w4_{f}.csv"))) for n, f in (("V", "applied"),
                                                                                     ("G", "e480"))}
    # victim: only the clouds after the restore (the bench log starts at load time)
    lv = (DATI / "mola_victim_38.6_38.7-39.6.log").read_text().split("\n")
    i0 = next(i for i, x in enumerate(lv) if "Processing observation" in x and "stamp=38.7000" in x)
    log = {"V": "\n".join(lv[i0:]), "G": (DATI / "mola_w4_sel0_38.7-39.6.log").read_text()}
    isaac, node = _IsaacAssetto(), _NodeZone()
    visti = []

    def _rollout(genome, isa, mola, nd, goal, horizon, tag, **kw):
        # as _rollout_once: restore, then set_state if there is a target
        isa.pause()
        isa.restore()
        ts = ao.TWIN_STATE
        if ts["init"] in ("attacker_est", "ombra_est"):
            x, y, yaw = ts["init_target"]["target_scene"]
            isa.set_state(x, y, yaw, 0.0, 0.0, 0.0, zero_joints=True)
        b = tag.split("_")[2]
        visti.append((tag, ts["init"], ao.ZONE["on"]))
        lp = tmp_path / f"{tag}.log"
        lp.write_text(log[b])
        mola.log_path = str(lp)
        nd.last_trace = tracce[b]
        a = nd.node_zone_applicate["gemello"]
        a.update(n_ok=a["n_ok"] + 40, hash=nd.node_zone_ricevute["gemello"]["hash"], stato="ok")
        return None, (1.0, 0.0, 0.0), 0.0, "ok", 1.0, 40

    monkeypatch.setattr(ao, "rollout", _rollout)
    monkeypatch.setattr(ao, "TWIN_STATE", {})
    monkeypatch.setitem(ao.ZONE, "on", True)
    monkeypatch.setitem(ao.METRO, "on", True)
    args = types.SimpleNamespace(g_banco=str(p), g_overlay="/ovl", g_resave="ogni", horizon=1.0, settle_sec=2.0,
                                 catchup_sec=0.8, attack_from="spawn", stop_on="estimate", drift_reset_on="keep",
                                 trace=False, start_pose="-5.99,-1.0,0", damage_ref="baseline",
                                 baseline_path=str(BASELINE))
    assert ao._g_banco_run(args, isaac, types.SimpleNamespace(), node, out, [(0.0, 0.0)]) == 0
    assert ao.ZONE["on"] is True and all(v[2] is False for v in visti)
    assert [v[1] for v in visti] == ["attacker_est", "true"] * 4
    assert node.metri == [("gemello", 4)] * 8
    assert (out / "zone" / "w4_fotografia.npz").is_file() and (out / "zone" / "w4_ebe99146f954b663.npz").is_file()
    d = json.loads((out / "banco_G.json").read_text())
    assert list(d) == ["spec", "esiti", "assetto_spawn", "confronti"]
    sp = d["assetto_spawn"]
    assert [x["dopo_play_s"] for x in sp["letture"]] == [1.0, 5.0] and sp["roll"] == 1.2e-3
    es = {e["tag"]: e for e in d["esiti"]}
    g0, v0, g3 = es["banco_c0_G_r0"], es["banco_c0_V_r0"], es["banco_c3_G_r0"]
    assert g0["assetto_set_state"] == {} and v0["target_scene"] is None
    assert [x["fase"] for x in g0["assetto_letture"]] == ["dopo_restore", "dopo_set_state", "fine_rollout"]
    assert [x["fase"] for x in v0["assetto_letture"]] == ["dopo_restore", "fine_rollout"]
    a3 = s["casi"][1]["bracci"][0]["assetto"]
    r, pt, _ = banco_g.assetto_composto((1.2e-3, -0.8e-3, 0.0), (a3["roll_rad"], a3["pitch_rad"], a3["yaw_rad"]))
    assert g3["assetto_set_state"] == {"roll": r, "pitch": pt}
    ss = [kw for c, kw in isaac.comandi if c == "set_state"]
    assert len(ss) == 4 and sum("roll" in kw for kw in ss) == 2 and all("keep_tilt" not in kw for kw in ss)
    assert g0["ramo"]["classe"] == "ramo_gemello" and v0["ramo"]["classe"] == "ramo_vittima"
    assert all(e["zone"]["esito"] == "ok" and e["zone"]["nuvole_ok"] == 40 for e in d["esiti"])
    assert g0["danno_metro"]["mean_cm"] == 73.033 and v0["danno_metro"]["mean_cm"] == 68.459
    cf = d["confronti"]
    assert [(x["caso"], x["r"], x["confronto"]) for x in cf] == [("c0", 0, "a"), ("c3", 0, "a"),
                                                                  ("c0", 1, "a"), ("c3", 1, "a")]
    assert all((x["R1_cm"], x["laterale_cm"], x["danno_cm"]) == (1.66, 7.405, 4.574) for x in cf)
    assert all(not x["stesso_ramo"] and not x["segue"] for x in cf)


def test_nuvole_icp_riga_spezzata_dal_segnale_di_chiusura():
    # replay c2_r08: the rclcpp SIGINT line lands in the middle of the ICP line
    testo = (
        "[05:38:30.2400|DEBUG|Generator (raw)] Processing observation x stamp=19.300000\n"
        "[05:38:30.2441|DEBUG|LidarOdometry] ICP (kind=0): goodness=71.60% iters=2 "
        "pose=[1.902573 0.089009 0.012766 -[INFO] [1793507910.613592513] [rclcpp]: "
        "signal_handler(SIGINT/SIGTERM)\n"
        "1.814693 -0.161097 -0.019614] termReason=IterTermReason::Stalled pose_cov diagonal "
        "sigmas:{4.612702e-04 3.471532e-04 6.354944e-04 [m] 1.824364e-03 9.644582e-03 6.004460e-03 [deg]}\n"
        "[05:38:30.2450|INFO|LidarOdometry] pathStep=190 timestamp=00:00:19.300 isGood=1\n"
    )
    v = banco_g.nuvole_icp(testo)
    assert len(v) == 1
    assert v[0]["q"] == pytest.approx(0.716)
    assert v[0]["x"] == pytest.approx(1.902573)
    assert v[0]["yaw_deg"] == pytest.approx(-1.814693)
    assert v[0]["roll_deg"] == pytest.approx(-0.019614)
    assert v[0]["sigma"][1] == pytest.approx(3.471532e-04)
