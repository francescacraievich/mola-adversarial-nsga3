"""Per-meter and end-of-run damage quantities (--danno-metro, DANNO_METRO=1 with SERIE_B=1).

Synthetic cases with known values, only for the code: straight polyline y = 0 from (0,0)
to (5,0), true poses at known distances. With the option off the candidate records, the
fitness and the arguments of the campaign processes are unchanged."""

import json
import sys
import textwrap
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.baseline_distance import baseline_damage  # noqa: E402
from src.optimization import window_record as wrec  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_regole_notte as rn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ORCH = ROOT / "src" / "optimization" / "attack_orchestrator.py"
POLY = np.array([[0.0, 0.0], [5.0, 0.0]])


# --- danno_metro -------------------------------------------------------------------------

def test_metro_che_si_allontana():
    # distances 10, 20, 30 cm
    d = wrec.danno_metro([[1.0, 0.1], [1.5, 0.2], [2.0, 0.3]], POLY)
    assert d["n_points"] == 3
    assert d["start_cm"] == pytest.approx(10.0) and d["end_cm"] == pytest.approx(30.0)
    assert d["mean_cm"] == pytest.approx(20.0)              # (i)
    assert d["end_minus_start_cm"] == pytest.approx(20.0)   # (ii)
    assert d["max_cm"] == pytest.approx(30.0)               # (iii)
    assert d["max_minus_start_cm"] == pytest.approx(20.0)
    assert d["added_cm"] == pytest.approx(10.0)             # (iv) mean - start
    # baseline shifted by 10 cm: distances 0, 10, 20
    assert d["added_shift_cm"] == pytest.approx(10.0)


def test_metro_che_rientra():
    # the genome brings the robot back: distances 30, 20, 10
    d = wrec.danno_metro([[1.0, 0.3], [1.5, 0.2], [2.0, 0.1]], POLY)
    assert d["mean_cm"] == pytest.approx(20.0)
    assert d["end_minus_start_cm"] == pytest.approx(-20.0)
    assert d["max_cm"] == pytest.approx(30.0) and d["max_minus_start_cm"] == pytest.approx(0.0)
    assert d["added_cm"] == pytest.approx(-10.0)
    # shifted by 30 cm: distances 0, 10, 20, always >= 0
    assert d["added_shift_cm"] == pytest.approx(10.0)


def test_scostamento_conservato_e_picco_interno():
    # entry offset of 20 cm kept: zero added damage in both definitions
    d = wrec.danno_metro([[1.0, -0.2], [1.5, -0.2], [2.0, -0.2]], POLY)
    assert d["added_cm"] == pytest.approx(0.0) and d["added_shift_cm"] == pytest.approx(0.0)
    assert d["end_minus_start_cm"] == pytest.approx(0.0)
    # peak at mid-meter: end - start zero, maximum excursion 40 cm
    d = wrec.danno_metro([[1.0, 0.0], [1.5, 0.4], [2.0, 0.0]], POLY)
    assert d["end_minus_start_cm"] == pytest.approx(0.0)
    assert d["max_cm"] == pytest.approx(40.0)
    assert d["mean_cm"] == pytest.approx(40.0 / 3, abs=1e-3)   # the record rounds to 1e-3 cm


def test_oltre_l_estremo():
    # past (5,0) the distance from the endpoint applies: 100 and 150 cm
    d = wrec.danno_metro([[6.0, 0.0], [6.5, 0.0]], POLY)
    assert d["start_cm"] == pytest.approx(100.0) and d["end_cm"] == pytest.approx(150.0)
    assert d["added_shift_cm"] == pytest.approx(25.0)   # baseline shifted by 1 m: 0 and 50 cm


def test_coerente_con_fitness_e_baseline_window():
    rng = np.random.default_rng(7)
    path = np.column_stack([np.linspace(1.0, 2.0, 25), rng.normal(0.0, 0.05, 25)])
    d = wrec.danno_metro(path, POLY)
    bd = baseline_damage(path, POLY)
    bw = wrec.baseline_window_fields(path, POLY)
    assert d["mean_cm"] == pytest.approx(bd["dev_mean"] * 100, abs=1e-3)   # --damage mean fitness
    assert d["max_cm"] == pytest.approx(bd["dev_max"] * 100, abs=1e-3)
    assert d["end_cm"] == pytest.approx(bd["dev_end"] * 100, abs=1e-3)
    assert d["end_minus_start_cm"] == pytest.approx(bw["added_end_cm"], abs=1e-3)
    assert d["added_cm"] == pytest.approx(bw["added_mean_cm"], abs=1e-3)


def test_senza_pose_o_senza_baseline():
    assert wrec.danno_metro([[1.0, 0.1]], None) is None
    assert wrec.danno_metro(np.zeros((0, 2)), POLY) is None
    assert wrec.danno_metro([[float("nan"), 0.0]], POLY) is None
    d = wrec.danno_metro([[float("nan"), 0.0], [1.0, 0.1]], POLY)
    assert d["n_points"] == 1 and d["added_cm"] == 0.0


def test_definizioni_per_ogni_campo():
    d = wrec.danno_metro([[1.0, 0.1], [2.0, 0.2]], POLY)
    assert set(d) - {"n_points"} == set(wrec.DANNO_METRO_DEFINITIONS)


# --- danno_run ---------------------------------------------------------------------------

def _voce(w, n, mean, end, mx, added):
    return {"n_points": n, "mean_cm": mean, "end_cm": end, "max_cm": mx, "added_cm": added}


def test_danno_run_valori_noti():
    hist = [
        {"window": 0, "danno_metro_vittima": _voce(0, 10, 2.0, 3.0, 4.0, 1.0),
         "danno_metro_vincitore": {"mean_cm": 5.0}},
        {"window": 1, "danno_metro_vittima": _voce(1, 30, 6.0, 8.0, 9.0, 2.5),
         "danno_metro_vincitore": {"mean_cm": 7.0}},
        # attack discovered: victim damage excluded, winner recorded
        {"window": 2, "danno_metro_vittima": None, "danno_metro_vincitore": {"mean_cm": 9.0}},
    ]
    r = wrec.danno_run(hist)
    assert r["victim_end_cm"] == 8.0 and r["victim_end_window"] == 1
    assert r["victim_mean_cm"] == pytest.approx((10 * 2.0 + 30 * 6.0) / 40)
    assert r["victim_max_cm"] == 9.0
    assert r["victim_added_sum_cm"] == pytest.approx(3.5)
    assert r["winner_mean_cm"] == pytest.approx(7.0)
    assert r["n_windows"] == 2 and r["n_windows_winner"] == 3
    assert set(r["definitions"]) == set(wrec.DANNO_RUN_DEFINITIONS)


def test_danno_run_senza_voci():
    assert wrec.danno_run([]) is None
    assert wrec.danno_run([{"window": 0, "status": "isaac_stall"}]) is None


# --- winner_index ------------------------------------------------------------------------

def test_vincitore_per_indice_scelto_o_genoma_e_danno():
    log = [{"genome": [0.1, 0.2], "damage": 0.05},
           {"genome": [0.3, 0.4], "damage": 0.02},
           {"genome": [0.3, 0.4], "damage": 0.07}]   # same genome re-evaluated (warm start)
    assert wrec.winner_index(log, [0.3, 0.4], 0.07, chosen_idx=2) == 2
    assert wrec.winner_index(log, [0.3, 0.4], 0.07) == 2
    assert wrec.winner_index(log, [0.3, 0.4], 0.02) == 1
    # maxdamage: best_F[0] = -(dev_mean * 100) / 100
    x = 0.123456789
    log2 = [{"genome": [0.0], "damage": x}]
    assert wrec.winner_index(log2, [0.0], (x * 100.0) / 100.0) == 0
    assert wrec.winner_index(log, [0.9, 0.9], 0.07) is None
    assert wrec.winner_index(log, None, None) is None


# --- candidate record (eval_genome from the source) -------------------------------------

def _candidati(monkeypatch, danno_metro):
    """eval_genome with the harness of test_regole_notte; None = attribute missing from the
    arguments (as in runs recorded before the option)."""
    def sn(**kw):
        if "pop" in kw and danno_metro is not None:   # orchestrator arguments only
            kw["danno_metro"] = danno_metro
        return types.SimpleNamespace(**kw)
    monkeypatch.setattr(rn, "types", types.SimpleNamespace(SimpleNamespace=sn))
    return rn._candidati([None, None], ["ok", "ok"], [(1.0, 0.3, 0.0), (1.0, 0.3, 0.0)], accorge=False)


def _testo(e):
    return json.dumps({k: v for k, v in e.items() if k != "t_wall"}, sort_keys=True, default=str)


def test_candidato_opzione_accesa(monkeypatch):
    F, log = _candidati(monkeypatch, True)
    attesa = wrec.danno_metro(np.array([[0.0, 0.1], [0.5, 0.2], [1.0, 0.3]]), rn.POLY)
    for f, e in zip(F, log):
        assert e["danno_metro"] == attesa
        # fitness is the mean distance from the baseline along the meter
        assert -f[0] * 100 == pytest.approx(e["danno_metro"]["mean_cm"], abs=1e-3)


@pytest.mark.parametrize("valore", [None, False])
def test_candidato_opzione_spenta_come_prima(monkeypatch, valore):
    F_on, log_on = _candidati(monkeypatch, True)
    F, log = _candidati(monkeypatch, valore)
    assert all("danno_metro" not in e for e in log)
    assert F == F_on   # fitness unchanged
    for a, b in zip(log, log_on):
        b = {k: v for k, v in b.items() if k != "danno_metro"}
        assert _testo(a) == _testo(b)


# --- window entries (history block from the source) -----------------------------

def _blocco_finestra():
    src = ORCH.read_text().split("\n")
    i0 = next(i for i, r in enumerate(src) if "# --danno-metro: metre quantities for the victim" in r)
    i1 = next(i for i in range(i0, len(src)) if src[i].strip().startswith('print(f"  tempo finestra'))
    return textwrap.dedent("\n".join(src[i0:i1]))


def _esegui_blocco(acceso, dm_real, ch, real_end, sel_rec):
    trace = np.array([[1.0, 0.1], [1.5, 0.2], [2.0, 0.3]])
    eval_log = [{"genome": [0.1], "damage": 0.04, "chamfer": 12.0, "danno_metro": {"mean_cm": 4.0}},
                {"genome": [0.2], "damage": 0.06, "chamfer": 30.0, "danno_metro": {"mean_cm": 6.0}}]
    ns = {"args": types.SimpleNamespace(danno_metro=acceso), "wrec": wrec,
          "true_path": rn.ao.true_path, "node": types.SimpleNamespace(last_trace=trace),
          "baseline_poly": POLY, "eval_log": eval_log, "best_g": np.array([0.2]),
          "best_F": np.array([-0.06, 30.0]), "sel_rec": sel_rec, "dm_real": dm_real,
          "ch": ch, "real_end": real_end}
    exec(_blocco_finestra(), ns)
    return ns["danno_metro_rec"]


def test_finestra_opzione_accesa():
    r = _esegui_blocco(True, {"dev_mean": 0.2}, 15.5, (2.0, 0.3, 0.0), None)
    assert r["vincitore_idx"] == 1 and r["danno_metro_vincitore"] == {"mean_cm": 6.0}
    assert r["chamfer_vincitore_cm"] == 30.0 and r["chamfer_applicato_cm"] == 15.5
    assert r["danno_metro_vittima"]["mean_cm"] == pytest.approx(20.0)
    assert r["danno_metro_definizioni"] is wrec.DANNO_METRO_DEFINITIONS
    # pareto_maxdamage: the chosen index applies
    r = _esegui_blocco(True, {"dev_mean": 0.2}, 15.5, (2.0, 0.3, 0.0), {"chosen_idx": 0})
    assert r["vincitore_idx"] == 0 and r["chamfer_vincitore_cm"] == 12.0


def test_finestra_danno_escluso_o_applicazione_fallita():
    # attack discovered: victim damage excluded, Chamfer of the segment recorded
    r = _esegui_blocco(True, None, 15.5, (2.0, 0.3, 0.0), None)
    assert r["danno_metro_vittima"] is None and r["chamfer_applicato_cm"] == 15.5
    # application failed or Chamfer not measured
    r = _esegui_blocco(True, None, float("nan"), None, None)
    assert r["chamfer_applicato_cm"] is None


def test_finestra_opzione_spenta():
    assert _esegui_blocco(False, {"dev_mean": 0.2}, 15.5, (2.0, 0.3, 0.0), None) == {}


# --- arguments --------------------------------------------------------------------------

def test_richiede_damage_ref_baseline(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["x", "--danno-metro", "--out", str(tmp_path / "o")])
    assert rn.ao.main() == 2
    assert "--danno-metro richiede --damage-ref baseline" in capsys.readouterr().out


# --- run_campaign.sh ---------------------------------------------------------------------

@pytest.mark.parametrize("arm,atteso", [("nsga3", 1), ("random", 1), ("none", 0)])
def test_serie_b_accende_danno_metro(tmp_path, monkeypatch, arm, atteso):
    monkeypatch.delenv("DANNO_METRO", raising=False)
    r, _, orch, _ = rn.campagna(ROOT, tmp_path, {"SERIE_B": "1", "SEEDS": "1001"}, arm)
    assert r.returncode == 0, r.stdout[-3000:]
    assert orch.count("--danno-metro") == atteso
    r, _, orch, _ = rn.campagna(ROOT, tmp_path, {}, arm)
    assert r.returncode == 0, r.stdout[-3000:]
    assert "--danno-metro" not in orch


def test_danno_metro_sovrascrivibile_e_fuori_dal_nome(tmp_path, monkeypatch):
    monkeypatch.delenv("DANNO_METRO", raising=False)
    _, _, orch_on, out_on = rn.campagna(ROOT, tmp_path, {"SERIE_B": "1", "SEEDS": "1001"})
    r, _, orch, out = rn.campagna(ROOT, tmp_path, {"SERIE_B": "1", "SEEDS": "1001", "DANNO_METRO": "0"})
    assert r.returncode == 0 and "--danno-metro" not in orch
    assert Path(out).parent.name == Path(out_on).parent.name   # folder name unchanged
    r, _, orch, _ = rn.campagna(ROOT, tmp_path, {"DANNO_METRO": "1"})
    assert r.returncode == 0 and orch.count("--danno-metro") == 1
    r, _, orch, _ = rn.campagna(ROOT, tmp_path, {"DANNO_METRO": "2"})
    assert r.returncode != 0 and "DANNO_METRO sconosciuto" in r.stdout
