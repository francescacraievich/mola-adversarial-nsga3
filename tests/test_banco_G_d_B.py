"""Bench G, arm B: twin starting drift (node option --banco-twin-drift), per-arm genome and
drift in --g-banco, spec from pilot data. With the option off the node produces the same
clouds as before. Fake objects, no ROS node or MOLA."""

import importlib.util
import json
import math
import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

pn = pytest.importorskip("src.nodes.perturbation_node")
tm = pytest.importorskip("test_misure_fuori_percorso")
tg = pytest.importorskip("test_banco_G_d")
from src.optimization import attack_orchestrator as ao  # noqa: E402
from src.optimization import banco_g  # noqa: E402

# node before the option (reference copy); the comparison is skipped if missing
NODO_ORIGINALE = (Path(os.environ.get("SERIEB_LAVORO", str(Path.home() / "mola_lavoro"))) / "lavoro_G_banco_B/originale/src/nodes/perturbation_node.py")
PILOTA_G = (Path(__file__).resolve().parents[1] / "data/attack/verifica/2026-10-26_pilota_G")
RUN_S4001 = (PILOTA_G / "s4001/campagna/straight/nsga3_g3_p4x2_lag0.5_sel2_warm_stopramp_k1.0_reliable_"
             "serieB_s4001_pt1_oprandom_sgross_pilotaT5_dmin130/rep_01")
DERIVA_B = [0.0029, -0.39999, 0.0]


def _f32(v):
    return SimpleNamespace(data=list(v))


def _uscite(modulo, nuvole, prepara=None, cambio_a=4):
    """Output clouds (main topic) of the node built with the given module; prepara(n) sets
    options and state before the first cloud."""
    orig = tm.pn
    tm.pn = modulo
    try:
        n = tm._nodo(victim=False)
        if prepara is not None:
            prepara(n)
        n._set_genome(np.asarray(tm.GENOMA_MASSIMO, dtype=np.float64), source="test")
        _, out = tm._corri(n, nuvole, cambio_genoma_a=cambio_a)
    finally:
        tm.pn = orig
    return [o[0] for o in out], n


def _keep(n):
    n._drift_keep = True
    n.victim_generator.set_temporal_state(np.array([0.1, -0.2, 0.0]))


# --- node: option off ------------------------------------------------------------------

def test_nodo_argparse_spenta(monkeypatch):
    preso = {}

    class _Stop(Exception):
        pass

    orig = pn.argparse.ArgumentParser.parse_known_args

    def _parse(self, *a, **k):
        preso["args"] = orig(self, *a, **k)
        raise _Stop

    monkeypatch.setattr(pn.argparse.ArgumentParser, "parse_known_args", _parse)
    monkeypatch.setattr(sys, "argv", ["perturbation_node.py"])
    with pytest.raises(_Stop):
        pn.main()
    assert preso["args"][0].banco_twin_drift is False


@pytest.mark.skipif(not NODO_ORIGINALE.is_file(), reason="nodo di riferimento assente")
@pytest.mark.parametrize("prepara", [None, _keep], ids=["azzeramento", "keep"])
def test_nodo_spento_stesse_nuvole_di_prima(prepara):
    spec = importlib.util.spec_from_file_location("perturbation_node_originale", NODO_ORIGINALE)
    orig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orig)
    nuvole = tm._nuvole_sintetiche(10)
    a, _ = _uscite(orig, nuvole, prepara)
    b, _ = _uscite(pn, nuvole, prepara)
    assert a == b
    # option on with no value received: same as off
    c, _ = _uscite(pn, nuvole, lambda n: (prepara and prepara(n), setattr(n, "_banco_twin_drift_on", True)))
    assert c == b


def test_nodo_deriva_zero_come_spenta():
    """B0: starting drift (0, 0, 0) = the usual reset, same clouds."""
    nuvole = tm._nuvole_sintetiche(8)
    a, _ = _uscite(pn, nuvole)

    def b0(n):
        n._banco_twin_drift_on = True
        n._twin_drift_init_cb(_f32([0.0, 0.0, 0.0]))
    b, _ = _uscite(pn, nuvole, b0)
    assert a == b


def test_nodo_deriva_iniziale():
    nuvole = tm._nuvole_sintetiche(6)

    def b(n):
        n._banco_twin_drift_on = True
        n._twin_drift_init_cb(_f32(DERIVA_B))
    a, na = _uscite(pn, nuvole, cambio_a=None)
    c, nc = _uscite(pn, nuvole, b, cambio_a=None)
    assert a != c
    # starts from the given value (float32 of the message): same drift as a generator
    # started by hand from that value, one cloud at a time
    assert np.allclose(nc._twin_drift_init, DERIVA_B, atol=1e-6)
    for nodo, inizio in ((na, None), (nc, nc._twin_drift_init)):
        gen = pn.PerturbationGenerator(max_point_shift=0.05, noise_std=0.02, max_dropout_rate=0.15,
                                       max_drift_lag_s=0.5)
        if inizio is not None:
            gen.set_temporal_state(inizio)
        params = gen.encode_perturbation(np.asarray(tm.GENOMA_MASSIMO, dtype=np.float64))
        for _ in nuvole:
            gen._apply_temporal_drift(np.zeros((1, 4)), params)
        assert np.array_equal(nodo.generator.accumulated_drift, gen.accumulated_drift)
    assert not np.allclose(na.generator.accumulated_drift, nc.generator.accumulated_drift)
    # the next genome restarts from the same value
    nc._set_genome(np.asarray(tm.GENOMA_MASSIMO, dtype=np.float64), source="test")
    assert np.allclose(nc.generator.accumulated_drift, nc._twin_drift_init)
    # empty: reset again; wrong length: ignored
    nc._twin_drift_init_cb(_f32([1.0, 2.0]))
    assert nc._twin_drift_init is not None
    nc._twin_drift_init_cb(_f32([]))
    nc._set_genome(np.asarray(tm.GENOMA_MASSIMO, dtype=np.float64), source="test")
    assert nc._twin_drift_init is None and not np.any(nc.generator.accumulated_drift)


def _status(n):
    out = []
    n.status_pub = SimpleNamespace(publish=lambda m: out.append(m.data))
    n._meas_lock = None
    n.lat_times = n.lat_times or [0.001]      # filled by the publisher thread, absent here
    n._max_drift_lag_s = float("inf")
    n._replay_seq, n._replay_n, n._replay_total = 0, 0, 0
    n._replay_last_stamp = n._replay_first_stamp = None
    n._twin_offset = 0.0
    n._drift_keep = False
    pn.PerturbationNode._publish_status(n)
    return json.loads(out[-1])


def test_nodo_status_solo_con_opzione():
    _, n = _uscite(pn, tm._nuvole_sintetiche(2), cambio_a=None)
    assert "twin_drift_init" not in _status(n)
    n._banco_twin_drift_on = True
    assert _status(n)["twin_drift_init"] is None
    n._twin_drift_init_cb(_f32(DERIVA_B))
    assert _status(n)["twin_drift_init"] == pytest.approx(DERIVA_B, abs=1e-6)


# --- orchestrator ------------------------------------------------------------------------

def test_conferma_deriva():
    f = ao.twin_drift_confirmed
    assert f(False, None, None) is False                  # node without the option
    assert f(True, None, None) is True
    assert f(True, [0.0, 0.0, 0.0], None) is False
    assert f(True, [0.0029, -0.39999, 0.0], DERIVA_B) is True
    assert f(True, [0.0029, -0.3999, 0.0], DERIVA_B) is False
    assert f(True, None, DERIVA_B) is False


class _NodoConferma:
    """Reduced AttackNode: publish updates the status as the node does."""
    _publish_confirmed = ao.AttackNode._publish_confirmed
    set_twin_drift_init = ao.AttackNode.set_twin_drift_init
    _F32 = SimpleNamespace

    def __init__(self, opzione=True):
        self.st_twin_drift_on, self.st_twin_drift_init, self.inviati = False, None, []

        def publish(m):
            self.inviati.append(list(m.data))
            if opzione:
                self.st_twin_drift_on = True
                self.st_twin_drift_init = [round(float(np.float32(x)), 6) for x in m.data] or None
        self.node = SimpleNamespace(create_publisher=lambda *a: SimpleNamespace(publish=publish))

    def spin(self, t):
        pass


def test_set_twin_drift_init():
    n = _NodoConferma()
    assert n.set_twin_drift_init(DERIVA_B) is True and n.inviati[-1] == DERIVA_B
    assert n.set_twin_drift_init(None) is True and n.inviati[-1] == []
    assert _NodoConferma(opzione=False).set_twin_drift_init(None) is False


def _spec_b(tmp_path):
    s = tg._spec(tmp_path)
    s["casi"][0]["saved_scene"] = [3.947 - 5.99994, -0.278 - 1.00002, -0.092]
    s["casi"][1]["saved_scene"] = [2.95 - 5.99994, -0.1887 - 1.00002, -0.0667]
    g = [0.0039, -0.5305, 0.0] + [0.1] * 14
    s["casi"][0]["bracci"] = [{"nome": "A", "init": "attacker_est"},
                              {"nome": "B", "init": "attacker_est", "genome": g, "drift_init": DERIVA_B},
                              {"nome": "B0", "init": "attacker_est", "genome": g, "drift_init": [0.0] * 3}]
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(s))
    return p, g


class _NodeBancoB(tg._NodeBanco):
    def __init__(self, conferma=True):
        self.derive, self.conferma = [], conferma

    def set_twin_drift_init(self, d):
        self.derive.append(d)
        return self.conferma


def _args_banco(p):
    return types.SimpleNamespace(g_banco=str(p), g_overlay="/ovl", g_resave="ogni", horizon=1.0, settle_sec=2.0,
                                 catchup_sec=0.8, attack_from="spawn", stop_on="estimate",
                                 drift_reset_on="genome", trace=False)


def test_ciclo_banco_b(tmp_path, monkeypatch):
    p, g = _spec_b(tmp_path)
    out = tmp_path / "rep_01"
    out.mkdir()
    visti = []

    def _rollout(genome, isaac, mola, node, goal, horizon, tag, **kw):
        visti.append((tag, genome, node.derive[-1]))
        log = tmp_path / f"mola_{tag}.log"
        log.write_text(tg.RIGHE)
        mola.log_path = str(log)
        node.last_twin_fp = {"g_check": {}, "drift_twin_xy_cm": 40.0}
        return None, (1.0, 0.0, 0.0), 0.0, "ok", 1.0, 10

    monkeypatch.setattr(ao, "rollout", _rollout)
    monkeypatch.setattr(ao, "TWIN_STATE", {})
    node = _NodeBancoB()
    assert ao._g_banco_run(_args_banco(p), tg._IsaacBanco(), types.SimpleNamespace(), node, out,
                           [(0.0, 0.0), (5.0, 0.0)]) == 0
    per_tag = {t: (gg, d) for t, gg, d in visti}
    assert per_tag["banco_guasta_A_r0"] == (None, None)
    assert per_tag["banco_guasta_B_r0"] == (g, DERIVA_B)
    assert per_tag["banco_guasta_B0_r1"] == (g, [0.0, 0.0, 0.0])
    assert per_tag["banco_controllo_A_r1"] == (None, None)
    assert node.derive[0] is None and node.derive[-1] is None      # reset before and after
    es = json.loads((out / "banco_G.json").read_text())["esiti"]
    b = [e for e in es if e["braccio"] == "B"][0]
    assert b["genome"] == g and b["drift_init"] == DERIVA_B


def test_ciclo_banco_b_nodo_senza_opzione(tmp_path, monkeypatch):
    p, _ = _spec_b(tmp_path)
    monkeypatch.setattr(ao, "TWIN_STATE", {})
    monkeypatch.setattr(ao, "rollout", lambda *a, **k: pytest.fail("rollout senza conferma della deriva"))
    assert ao._g_banco_run(_args_banco(p), tg._IsaacBanco(), types.SimpleNamespace(),
                           _NodeBancoB(conferma=False), tmp_path, []) == 2


@pytest.mark.parametrize("guasto, msg", [
    (lambda b: b.update(genome=[0.0] * 16), "17 valori"),
    (lambda b: b.update(genome=[2.0] * 17), r"fuori da \[-1, 1\]"),
    (lambda b: b.pop("genome"), "drift_init senza genome"),
    (lambda b: b.update(drift_init=[4.0, 0.0, 0.0]), "oltre"),
    (lambda b: b.update(drift_init=[0.0, float("nan"), 0.0]), "non finiti"),
])
def test_spec_b_rifiutate(tmp_path, guasto, msg):
    p, _ = _spec_b(tmp_path)
    s = json.loads(p.read_text())
    guasto(s["casi"][0]["bracci"][1])
    with pytest.raises(ValueError, match=msg):
        banco_g.check_spec(s)


def test_vincoli_b(tmp_path):
    p, _ = _spec_b(tmp_path)
    ov = tg._overlay(tmp_path)
    a = lambda **kw: tg._args(g_banco=str(p), g_overlay=ov, **kw)  # noqa: E731
    assert banco_g.check_args(a()) is None
    assert banco_g.check_args(a(drift_reset_on="genome")) is None
    # keep allowed for arm E (keep runs)
    assert banco_g.check_args(a(drift_reset_on="keep")) is None
    for kw in (dict(attack_from="motion"), dict(drift_reset_on="drive")):
        assert "--attack-from spawn" in banco_g.check_args(a(**kw))
    # without genome arms the new constraints do not apply
    s = tg._spec(tmp_path)
    q = tmp_path / "spec_acd.json"
    q.write_text(json.dumps(s))
    assert banco_g.check_args(tg._args(g_banco=str(q), g_overlay=ov, drift_reset_on="drive")) is None
    assert banco_g.usa_deriva(banco_g.load_spec(p)) and not banco_g.usa_deriva(s)


def test_analisi_deriva_di_partenza():
    spec = importlib.util.spec_from_file_location("analisi_G_d", ROOT / "scripts" / "analisi_G_d_guasto.py")
    an = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(an)
    e = {"braccio": "B", "saved_err_cm": 0.1, "esito": {"prime_n": 10}, "drift_init": DERIVA_B,
         "fp": {"g_check": {"firme_ok": True, "loaded_is_shadow_file": True, "hash_ok": True, "clock_ok": True},
                "drift_twin_xy_cm": 40.0}}
    assert an.valido(e) == []
    e["fp"]["drift_twin_xy_cm"] = 0.0
    assert any("deriva di partenza" in m for m in an.valido(e))
    e.pop("drift_init")                       # A, C, D: no drift check
    assert an.valido(e) == []


# --- spec from real data (read only) -----------------------------------------------------

@pytest.mark.skipif(not (tg.RUN_G_S1001.is_dir() and RUN_S4001.is_dir()), reason="run del pilota G assenti")
def test_spec_b_dai_dati_veri(tmp_path):
    out = tmp_path / "spec.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "banco_G_d_spec.py"), "--out", str(out),
                        "--caso", f"guasta_s1001_w4:{tg.RUN_G_S1001}:4:B,B0",
                        "--caso", f"controllo_s1001_w3:{tg.RUN_G_S1001}:3:A",
                        "--caso", f"guasta_s4001_w3:{RUN_S4001}:3:B,B0"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "NO:" not in r.stdout and r.stdout.count("= history entro") == 5
    s = banco_g.load_spec(out)
    g1, _, g4 = s["casi"]
    hist = json.loads((tg.RUN_G_S1001 / "history.json").read_text())
    b, b0 = g1["bracci"]
    assert b["genome"] == hist[3]["genome"] and b0["genome"] == hist[3]["genome"]
    assert b["drift_init"] == pytest.approx(DERIVA_B, abs=1e-5) and b0["drift_init"] == [0.0, 0.0, 0.0]
    d = g4["bracci"][0]["drift_init"]
    assert abs(100 * math.hypot(d[0], d[1]) - 37.21) < 0.01 and abs(np.linalg.norm(d) - 0.40) < 1e-9
