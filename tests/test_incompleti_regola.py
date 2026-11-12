"""Incomplete-candidate rule: candidates with timeout or stopped_early classified by cause.
(a) guasto_catena from signals independent of the genome, (b) mola_perde_posa (rule 8),
calculation (calc_ok false), (c) attacco_valido with the damage from the baseline pose at
the end of the window. Synthetic data and, if present, older recorded logs read only."""

import difflib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.fronte_pareto import candidato_valido, motivi_non_valido  # noqa: E402
from src.optimization import window_record as wrec  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ORIG = ROOT.parent / "lavoro_incompleti_regola_originale"
REPO = Path.home() / "mola-adversarial-nsga3"
CAMP = REPO / "data" / "attack" / "campaign" / "straight"
BASE = REPO / "data" / "attack" / "baseline" / "rettilineo_direction_N5_media3semi_dmin140_sosta1800.csv"


def _frame_rows(stamps, t0=1000.0):
    # one line per input cloud, arrivals 0.1 s apart in wall time
    return [{"t_arr_wall": str(t0 + 0.1 * i), "stamp_in": str(s)} for i, s in enumerate(stamps)]


def _ok_controlli():
    return {"isaac": {"esito": "ok"}, "adattatore": {"esito": "ok"}, "nuvole_nodo": {"esito": "ok"}}


def _printf(stamps_good, stamps_bad=()):
    rows = [{"stamp_s": s, "is_good": 1} for s in stamps_good] + [{"stamp_s": s, "is_good": 0} for s in stamps_bad]
    return sorted(rows, key=lambda r: r["stamp_s"])


# --- (a) chain failure --------------------------------------------------------------------

def test_a_buco_nuvole_in_ingresso():
    st = [round(10 + 0.1 * i, 1) for i in range(20)] + [round(13.1 + 0.1 * i, 1) for i in range(10)]
    b = wrec.buchi_arrivi_nodo(_frame_rows(st), 999.0, 1010.0)
    assert b["esito"] == "guasto" and b["buco_max_s"] == pytest.approx(1.2)
    c = wrec.classe_incompleto("timeout", {**_ok_controlli(), "nuvole_nodo": b},
                               accorge={"accorge": False}, calc_ok=True)
    assert c["classe"] == "guasto_catena" and c["guasti"] == ["nuvole_nodo"]


def test_a_buco_sotto_soglia_e_salto_indietro():
    st = [10.0, 10.1, 10.2, 11.1, 5.0, 5.1]          # 0.9 s and restore (jump back)
    b = wrec.buchi_arrivi_nodo(_frame_rows(st), 999.0, 1010.0)
    assert b["esito"] == "ok" and b["buco_max_s"] == pytest.approx(0.9)
    assert wrec.buchi_arrivi_nodo(None, 0, 1)["esito"] == "non_determinato"


def test_a_stallo_isaac_e_adattatore():
    # stall with retry before the last attempt: does not count; stall after: failure
    ev = [{"t_wall": 100.0, "outcome": "retry", "recovered_after_s": 5.0}]
    assert wrec.stalli_isaac_rollout(ev, 90.0, 130.0, eta_posa_s=0.2, stall_sec=5.0)["esito"] == "ok"
    ev2 = ev + [{"t_wall": 110.0, "outcome": "run_abort"}]
    assert wrec.stalli_isaac_rollout(ev2, 90.0, 130.0, 0.2, 5.0)["esito"] == "guasto"
    assert wrec.stalli_isaac_rollout([], 90.0, 130.0, 7.0, 5.0)["esito"] == "guasto"
    ad = wrec.eventi_adattatore([{"t_prev_clean": 95.0, "t_restart": 120.0}], 90.0, 130.0)
    c = wrec.classe_incompleto("stopped_early(0.23m)", {**_ok_controlli(), "adattatore": ad},
                               accorge={"accorge": True, "motivo": "x"}, calc_ok=False)
    # (a) before (b) and before the calculation
    assert c["classe"] == "guasto_catena"


def test_a_non_determinato_non_basta():
    c = wrec.classe_incompleto("timeout", {"isaac": {"esito": "ok"}, "adattatore": {"esito": "non_determinato"},
                                           "nuvole_nodo": {"esito": "non_determinato"}},
                               accorge={"accorge": False}, calc_ok=True)
    assert c["classe"] == "attacco_valido" and c["non_determinati"] == ["adattatore", "nuvole_nodo"]


# --- (b) MOLA loses the pose, calculation -------------------------------------------------

def test_b_mola_senza_posa_oltre_1_s():
    # last pose at 10.9 s, rejected from 11.0 to 11.9, pose again at 12.1: 1.2 s without pose
    rec = _printf([round(10.0 + 0.1 * i, 1) for i in range(10)] + [12.1, 12.2],
                  stamps_bad=[round(11.0 + 0.1 * i, 1) for i in range(10)])
    acc = wrec.mola_accorge(rec)
    assert acc["accorge"] is True
    c = wrec.classe_incompleto("timeout", _ok_controlli(), accorge=acc, calc_ok=True)
    assert c["classe"] == "mola_perde_posa"


def test_calcolo_non_rifatto():
    c = wrec.classe_incompleto("stopped_early(0.23m)", _ok_controlli(), accorge={"accorge": False}, calc_ok=False)
    assert c["classe"] == "calcolo"


# --- (c) valid attack and damage at the end of the window ---------------------------------

def test_c_valido_e_danno_fine_finestra(tmp_path):
    acc = wrec.mola_accorge(_printf([10.0 + 0.1 * i for i in range(30)]))
    c = wrec.classe_incompleto("stopped_early(0.51m)", _ok_controlli(), accorge=acc, calc_ok=True)
    assert c["classe"] == "attacco_valido"
    csv = tmp_path / "b.csv"
    csv.write_text("s_m,x_m,y_m\n0,0,0\n1,1,0\n")
    (tmp_path / "b.json").write_text(json.dumps({"punti_fine_finestra_medi": [
        {"finestra": 0, "x_m": 1.0, "y_m": 0.0}, {"finestra": 1, "x_m": 2.0, "y_m": 0.0}]}))
    pf = wrec.punti_fine_finestra(csv)
    assert wrec.danno_fine_finestra([0.6, 0.3, 0.0], pf[0]) == pytest.approx(0.5)
    assert wrec.danno_fine_finestra(None, pf[0]) is None
    (tmp_path / "b.json").write_text("{}")
    with pytest.raises(ValueError):
        wrec.punti_fine_finestra(csv)


@pytest.mark.skipif(not BASE.exists(), reason="baseline della serie B assente")
def test_punti_fine_finestra_baseline_serie_b():
    pf = wrec.punti_fine_finestra(BASE)
    assert sorted(pf) == [0, 1, 2, 3, 4] and pf[4][0] == pytest.approx(5.1112)


def test_c_valido_in_selezione_e_invarianza_senza_campo():
    base = {"status": "stopped_early(0.40m)", "counted": True, "admissible": True, "damage": 0.3,
            "chamfer": 2.0, "budget_ok": True, "calc_ok": True, "stealth_ok": True, "f2_ok": True}
    # without the field (option off) stopped_early stays out
    assert candidato_valido(base, 5.0, True) is False
    assert candidato_valido({**base, "incompleto_classe": "attacco_valido"}, 5.0, True) is True
    for cl in ("guasto_catena", "mola_perde_posa", "calcolo"):
        assert candidato_valido({**base, "incompleto_classe": cl}, 5.0, True) is False
    assert motivi_non_valido({**base, "counted": False, "admissible": False,
                              "incompleto_classe": "guasto_catena"}, 5.0) != []
    assert wrec.classe_incompleto("ok", _ok_controlli()) is None
    assert wrec.classe_incompleto("est_capped", _ok_controlli()) is None


# --- recorded cases from older runs (read only) -------------------------------------------

def _rec(run, w, e):
    h = json.loads((CAMP / run / "rep_01" / "history.json").read_text())
    win = [x for x in (h["windows"] if isinstance(h, dict) else h) if x.get("window", x.get("k")) == w]
    return (win[0] if win else (h["windows"] if isinstance(h, dict) else h)[w])["evaluations"][e]


NOBUDGET = "nsga3_g3_p32x16_lag0.5_sel3_warm_H1.0_nobudget_dannobaseline"
SOGLIA10 = "nsga3_g3_p32x16_eps10_lag0.5_sel3_warm_H1.0_soglia10_dannobaseline"


@pytest.mark.skipif(not (CAMP / NOBUDGET / "rep_01" / "logs" / "mola_w3_e180.log").exists(),
                    reason="run della serie A assente")
def test_registrato_c_nobudget_w3_e180():
    acc = wrec.mola_accorge(wrec.read_printf(str(CAMP / NOBUDGET / "rep_01" / "logs" / "mola_w3_e180.log")))
    assert acc["accorge"] is False and acc["n_rifiutate"] == 0
    r = _rec(NOBUDGET, 3, 180)
    assert str(r["status"]).startswith("stopped_early")
    c = wrec.classe_incompleto(r["status"], _ok_controlli(), accorge=acc, calc_ok=r.get("calc_ok", True))
    assert c["classe"] == "attacco_valido"


@pytest.mark.skipif(not (CAMP / SOGLIA10 / "rep_01" / "logs" / "mola_w3_e20.log").exists(),
                    reason="run della serie A assente")
def test_registrato_calcolo_soglia10_w3_e20():
    acc = wrec.mola_accorge(wrec.read_printf(str(CAMP / SOGLIA10 / "rep_01" / "logs" / "mola_w3_e20.log")))
    r = _rec(SOGLIA10, 3, 20)
    assert r["calc_ok"] is False
    c = wrec.classe_incompleto(r["status"], _ok_controlli(), accorge=acc, calc_ok=r["calc_ok"])
    assert c["classe"] == "calcolo"


# --- invariance with the option off -------------------------------------------------------

@pytest.mark.skipif(not ORIG.exists(), reason="copia intatta assente")
@pytest.mark.parametrize("rel", ["src/optimization/attack_orchestrator.py", "src/analysis/fronte_pareto.py",
                                 "src/optimization/window_record.py"])
def test_righe_originali_tolte_solo_quelle_attese(rel):
    """All original lines remain, except those replaced by a version guarded by the
    option."""
    a = (ORIG / rel).read_text().splitlines()
    b = (ROOT / rel).read_text().splitlines()
    tolte = [l[1:].strip() for l in difflib.unified_diff(a, b, lineterm="", n=0)
             if l.startswith("-") and not l.startswith("---")]
    attese = {
        "src/optimization/attack_orchestrator.py": {
            'if re_end is not None and re_st in VALID and (not re_slow or args.search == "fixed" or args.admissibility == "none") and not re_mis:',
            '"search_gap0_cm": e.get("twin_gap0_cm"), "reeval_gap0_cm": re_gap0, "reeval_twin_misaligned": re_mis})'},
        "src/analysis/fronte_pareto.py": {"if stato not in STATI_VALIDI:"},
        "src/optimization/window_record.py": set(),
    }[rel]
    assert set(tolte) == attese


def test_opzione_spenta_di_default_nel_sorgente():
    src = (ROOT / "src/optimization/attack_orchestrator.py").read_text()
    assert 'ap.add_argument("--incompleti-regola", action="store_true"' in src
    # every new branch goes through the option guard
    assert src.count('getattr(args, "incompleti_regola", False)') == 3
    assert "if args.incompleti_regola:" in src


# --- run_campaign.sh ----------------------------------------------------------------------

def test_run_campaign_serie_b_passa_l_opzione(tmp_path):
    from tests.test_regole_notte import campagna
    r, _, orch, _ = campagna(ROOT, tmp_path, {"SERIE_B": "1", "SEEDS": "1001"}, "nsga3")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert orch.count("--incompleti-regola") == 1 and "--mola-se-ne-accorge" in orch
    r, _, orch, _ = campagna(ROOT, tmp_path, {"SERIE_B": "1", "SEEDS": "1001"}, "none")
    assert r.returncode == 0 and "--incompleti-regola" not in orch
    r, _, orch, _ = campagna(ROOT, tmp_path, {"SEEDS": "1001"}, "nsga3")
    assert r.returncode == 0 and "--incompleti-regola" not in orch


# --- (c) with the robot stopped: the q check stays ----------------------------------------

class _NodoQ:
    def __init__(self, q):
        self.last_quality_mean = q


@pytest.mark.parametrize("q, atteso", [(0.15, (False, "rilevabile (qualita' ICP bassa)", False)),
                                       (0.5, (True, "", False)), (None, (True, "", False))])
def test_c_robot_fermo_ricontrolla_q(q, atteso):
    from src.optimization import attack_orchestrator as ao
    nodo = _NodoQ(q)
    # stealth_check stops at "robot stopped" without looking at q
    assert ao.stealth_check(nodo, "stopped_early(0.05m)", 0.05, 1.0)[1] == "robot fermo"
    ok, motivo, _, dos = ao.stealth_robot_fermo_valido(nodo)
    assert (ok, motivo, dos) == atteso
    # same threshold and same output as stealth_check for a moving robot
    if q is not None and q < ao.STEALTH_MIN_QUALITY:
        assert ao.stealth_check(nodo, "stopped_early(0.50m)", 0.5, 1.0)[:2] == (ok, motivo)


def test_c_robot_fermo_nel_sorgente_usa_il_ricontrollo():
    src = (ROOT / "src/optimization/attack_orchestrator.py").read_text()
    i = src.index('if inc["classe"] == "attacco_valido" and dos_reason == "robot fermo":')
    blocco = src[i:i + 900]
    assert "stealth_robot_fermo_valido(node)" in blocco
    assert "True, False, \"\", \"\"" not in blocco
