#!/usr/bin/env python3
"""Per-window log, random references and filter of anomalous twins, without Isaac or MOLA.

    python3 -m pytest tests/test_registro_serieB.py -q
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.optimization import window_record as wr  # noqa: E402
from src.analysis.fronte_pareto import candidato_valido, fronte_pareto  # noqa: E402


def riga(step, t, q=0.99, iters=1, term="Stalled", good=1):
    ts = datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y/%m/%d,%H:%M:%S.%f")
    return (f"[LidarOdometry] pathStep={step} timestamp={ts} goodness={q:.3f} minRequired=0.200 "
            f"iters={iters} termReason=IterTermReason::{term} isGood={good} "
            f"adapt_thres_sigma=0.1975 consecutive_bad=0\n")


def scrivi_log(path, righe, testa="mola-cli avviato\n"):
    path.write_text(testa + "".join(righe))
    return path


# ---------------------------------------------------------------------------
# Baseline, added deviation, signed lateral
# ---------------------------------------------------------------------------

POLY = np.array([[0.0, 0.0], [5.0, 0.0]])


def test_laterale_con_segno_e_prolungata():
    lat = wr.signed_lateral([[1.0, 0.2], [2.0, -0.3], [6.0, 0.1], [-1.0, -0.1]], POLY)
    assert lat == pytest.approx([0.2, -0.3, 0.1, -0.1])
    # past the end, the unsigned distance includes the overshoot, the lateral one does not
    from src.analysis.baseline_distance import dist_to_polyline
    assert dist_to_polyline([[6.0, 0.1]], POLY)[0] == pytest.approx(np.hypot(1.0, 0.1))


def test_laterale_polilinea_a_due_segmenti():
    poly = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    # left of the second segment (towards +y) means x < 1
    assert wr.signed_lateral([[0.9, 0.6]], poly)[0] == pytest.approx(0.1)
    assert wr.signed_lateral([[1.2, 0.6]], poly)[0] == pytest.approx(-0.2)


def test_campi_baseline_della_finestra():
    path = [[0.0, 0.01], [0.5, 0.02], [1.0, -0.04]]
    f = wr.baseline_window_fields(path, POLY)
    assert f["start_cm"] == pytest.approx(1.0)
    assert f["end_cm"] == pytest.approx(4.0)
    assert f["mean_cm"] == pytest.approx(7.0 / 3, abs=1e-3)
    assert f["added_end_cm"] == pytest.approx(3.0)
    assert f["added_mean_cm"] == pytest.approx(7.0 / 3 - 1.0, abs=1e-3)
    assert f["lat_start_cm"] == pytest.approx(1.0)
    assert f["lat_end_cm"] == pytest.approx(-4.0)
    assert f["lat_added_end_cm"] == pytest.approx(-5.0)
    assert f["lat_mean_cm"] == pytest.approx(-1.0 / 3, abs=1e-3)
    assert f["n_points"] == 3
    assert wr.baseline_window_fields(path, None) is None
    assert wr.baseline_window_fields([[np.nan, np.nan]], POLY) is None


# ---------------------------------------------------------------------------
# Scans from the MOLA log
# ---------------------------------------------------------------------------

def test_conteggio_scansioni(tmp_path):
    # slots 10.0 .. 11.0 (11 expected): 10.3 and 10.4 missing, 10.6 rejected
    stamps = [10.0, 10.1, 10.2, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1]
    righe = [riga(i, t, q=(0.15 if t == 10.6 else 0.98), good=(0 if t == 10.6 else 1),
                  term=("MaxIterations" if t == 10.6 else "Stalled"))
             for i, t in enumerate(stamps)]
    rows = wr.read_printf(scrivi_log(tmp_path / "mola_victim.log", righe))
    c = wr.scan_counts(rows, 10.0, 11.0)
    assert c["expected"] == 11 and c["processed"] == 9 and c["lost"] == 2
    assert c["rejected"] == 1 and c["n_maxiter"] == 1
    assert c["q_min"] == pytest.approx(0.15) and c["q_median"] == pytest.approx(0.98)
    assert c["log_complete"] is True
    # log stopped before the end of the interval: incomplete
    assert wr.scan_counts(rows, 10.0, 11.5)["log_complete"] is False
    assert wr.scan_counts(rows, None, 11.0) is None
    assert wr.scan_counts(rows, 11.0, 10.0) is None
    assert wr.read_printf(tmp_path / "manca.log") == []


def test_stamp_della_printf_letti_come_tempo_di_simulazione(tmp_path):
    rows = wr.read_printf(scrivi_log(tmp_path / "m.log", [riga(1, 2081.5)]))
    assert rows[0]["stamp_s"] == pytest.approx(2081.5)


# ---------------------------------------------------------------------------
# Anomalous twins
# ---------------------------------------------------------------------------

def test_flag_am2_e_al():
    rec = [{"stamp_s": 20.2, "goodness": 0.51, "iters": 8, "term_reason": "Stalled"},
           {"stamp_s": 20.3, "goodness": 0.99, "iters": 1, "term_reason": "Stalled"}]
    f = wr.twin_icp_flags(rec, prev_last_stamp=20.0)
    assert f["twin_icp_gap_s"] == pytest.approx(0.2) and f["twin_am2"] is True
    assert f["twin_al"] is True and f["twin_icp_first_q"] == pytest.approx(0.51)
    f = wr.twin_icp_flags(rec, prev_last_stamp=19.9)
    assert f["twin_icp_gap_s"] == pytest.approx(0.3) and f["twin_am2"] is False
    # MaxIterations in the first 0.2 s, even if not at the first ICP
    rec2 = [{"stamp_s": 30.0, "goodness": 0.99, "iters": 1, "term_reason": "Stalled"},
            {"stamp_s": 30.2, "goodness": 0.95, "iters": 25, "term_reason": "MaxIterations"},
            {"stamp_s": 30.4, "goodness": 0.10, "iters": 25, "term_reason": "MaxIterations"}]
    assert wr.twin_icp_flags(rec2, None)["twin_al"] is True
    assert wr.twin_icp_flags(rec2[:1] + rec2[2:], None)["twin_al"] is False
    # undeterminable
    f = wr.twin_icp_flags([], 20.0)
    assert f["twin_am2"] is None and f["twin_al"] is None
    assert wr.twin_icp_flags(rec, None)["twin_am2"] is None


def test_modi_del_filtro():
    f = {"twin_am2": True, "twin_al": False}
    assert wr.twin_filtered("none", f) is False
    assert wr.twin_filtered("am2", f) is True
    assert wr.twin_filtered("al", f) is False
    assert wr.twin_filtered("am2_al", f) is True
    assert wr.twin_filtered("am2", {"twin_am2": None, "twin_al": None}) is False
    with pytest.raises(ValueError):
        wr.twin_filtered("altro", f)


def _rec(idx, damage, chamfer, **kw):
    r = {"status": "ok", "counted": True, "admissible": True, "damage": damage,
         "chamfer": chamfer, "genome": [float(idx)]}
    r.update(kw)
    return r


def test_selezione_esclude_i_gemelli_filtrati():
    recs = [_rec(0, 0.05, 3.0), _rec(1, 0.09, 4.0, twin_filtered=True), _rec(2, 0.07, 4.5)]
    assert candidato_valido(recs[1]) is False
    assert candidato_valido(_rec(3, 0.01, 1.0, twin_filtered=False)) is True
    # without the key (older runs) nothing changes
    assert candidato_valido(_rec(4, 0.01, 1.0)) is True
    _, scelto = fronte_pareto(recs)
    assert scelto["idx"] == 2


def test_orchestratore_flag_dal_log_del_gemello(tmp_path):
    from src.optimization import attack_orchestrator as ao
    ao.TWIN_LOG_PREV["last_stamp"] = None
    node = SimpleNamespace(last_twin_flags=None)
    m1 = SimpleNamespace(log_path=scrivi_log(tmp_path / "mola_w0_nominal.log",
                                             [riga(2, 10.1), riga(3, 10.2), riga(4, 13.0)]))
    ao._note_twin_log(node, m1)
    assert node.last_twin_flags["twin_am2"] is None          # no previous
    assert ao.TWIN_LOG_PREV["last_stamp"] == pytest.approx(13.0)
    m2 = SimpleNamespace(log_path=scrivi_log(tmp_path / "mola_w0_e0.log",
                                             [riga(2, 13.2, q=0.5, iters=8), riga(3, 13.3)]))
    ao._note_twin_log(node, m2)
    f = ao._twin_flag_fields(node, "none")
    assert f["twin_am2"] is True and f["twin_al"] is True and f["twin_filtered"] is False
    assert ao._twin_flag_fields(node, "am2")["twin_filtered"] is True
    # log without printf rows (MOLA_LO_DEBUG_ICP_QUALITY off): unknown flags, no exclusion
    m3 = SimpleNamespace(log_path=scrivi_log(tmp_path / "mola_w0_e1.log", []))
    ao._note_twin_log(node, m3)
    f = ao._twin_flag_fields(node, "am2_al")
    assert f["twin_am2"] is None and f["twin_filtered"] is False
    assert ao.TWIN_LOG_PREV["last_stamp"] is None
    ao.TWIN_LOG_PREV["last_stamp"] = None


# ---------------------------------------------------------------------------
# Victim log in the orchestrator
# ---------------------------------------------------------------------------

def test_registro_della_vittima_e_ricalcolo(tmp_path):
    from src.optimization import attack_orchestrator as ao
    log = tmp_path / "mola_victim.log"
    scrivi_log(log, [riga(i, 10.0 + 0.1 * i) for i in range(6)])          # 10.0 .. 10.5
    trace = [{"true_x": -1.0 + 0.1 * i, "true_y": 0.01 * i} for i in range(5)]
    node = SimpleNamespace(last_trace=trace, last_drive_stamps=(10.2, 10.9),
                           last_victim_open_stamp=10.0)
    victim = SimpleNamespace(log_path=log)
    rec = ao._victim_window_record(node, victim, np.array([[-6.0, 0.0], [0.0, 0.0]]))
    assert rec["baseline"]["start_cm"] == pytest.approx(0.0)
    assert rec["baseline"]["lat_end_cm"] == pytest.approx(4.0)
    sd = rec["scans_drive"]
    assert sd["expected"] == 8 and sd["processed"] == 4 and sd["log_complete"] is False
    assert rec["scans_open"]["expected"] == 10
    assert rec["transport"]["mola_env_removed"] == ["FASTRTPS_DEFAULT_PROFILES_FILE"]
    # missing rows arrive later in the file: the window is recomputed
    scrivi_log(log, [riga(i, 10.0 + 0.1 * i) for i in range(12)])
    history = [{"window": 0, "victim_window": rec}]
    ao._refresh_victim_scans(history, victim)
    sd = history[0]["victim_window"]["scans_drive"]
    assert sd["processed"] == 8 and sd["lost"] == 0 and sd["log_complete"] is True
    assert history[0]["victim_window"]["refreshed"] is True
    # without a victim (vrestart) no counts, but the log exists
    rec2 = ao._victim_window_record(node, None, None)
    assert rec2["scans_drive"] is None and rec2["baseline"] is None


def test_stato_del_trasporto():
    env = {"RMW_IMPLEMENTATION": "rmw_fastrtps_cpp", "ROS_LOCALHOST_ONLY": "1",
           "MOLA_LO_DEBUG_ICP_QUALITY": "1"}
    t = wr.transport_state(env)
    assert t["RMW_IMPLEMENTATION"] == "rmw_fastrtps_cpp" and t["ROS_LOCALHOST_ONLY"] == "1"
    assert t["MOLA_LO_DEBUG_ICP_QUALITY"] == "1" and t["CYCLONEDDS_URI"] is None


# ---------------------------------------------------------------------------
# Perceptibility terms (generator, node, orchestrator)
# ---------------------------------------------------------------------------

def test_termini_uguali_alla_magnitudine():
    from src.perturbations.perturbation_generator import PerturbationGenerator
    gen = PerturbationGenerator()
    rng = np.random.default_rng(3)
    a = rng.uniform(-10, 10, size=(3000, 4))
    for b in (a + rng.normal(0, 0.02, a.shape),            # noise
              a[:2700],                                     # dropout
              np.vstack([a, rng.uniform(-10, 10, (150, 4))])):   # ghosts
        t = gen.compute_perturbation_terms(a, b, {})
        p = gen.compute_perturbation_magnitude(a, b, {})
        assert t["magnitude_cm"] == float(p)                      # identical, not approximate
        assert t["point_ratio"] == abs(len(b) - len(a)) / len(a)
        assert t["structural_cm"] == pytest.approx(20.0 * t["point_ratio"])
        assert np.hypot(t["chamfer_cm"], t["structural_cm"]) == pytest.approx(p)


def _nodo_finto(on):
    st = []
    zero = np.zeros(3)
    return SimpleNamespace(
        proc_times=[0.05], lat_times=[0.09], enabled=True, genome_id=1, n_in=10, n_out=10,
        _min_latency_ms=85.0, _chamfer=2.5, frame_idx=10, _max_drift_lag_s=0.5,
        generator=SimpleNamespace(accumulated_drift=zero),
        victim_generator=SimpleNamespace(accumulated_drift=zero),
        victim_stream=False, victim_attack=False, victim_genome_id=0, victim_frames=0,
        victim_frame_idx=0, _victim_offset=0.0, _victim_rec=[], _replay_seq=None,
        _replay_active=False, _replay_n=0, _replay_total=0, _replay_last_stamp=None,
        _twin_clock_on=False, _twin_offset=0.0,
        _perc={"shift_frac": 0.1, "shift_med_cm": 1.0, "n_added": 3, "n_removed": 4},
        _tc=None, status_pub=SimpleNamespace(publish=lambda m: st.append(json.loads(m.data))),
        _perc_terms_on=on,
        _perc_terms={"chamfer_cm": 2.0, "structural_cm": 1.5, "point_ratio": 0.075,
                     "magnitude_cm": 2.5}), st


def test_status_del_nodo_invariato_di_default():
    pn = pytest.importorskip("src.nodes.perturbation_node")
    n, st = _nodo_finto(False)
    pn.PerturbationNode._publish_status(n)
    assert "chamfer_pure_cm" not in st[0] and st[0]["chamfer_cm"] == 2.5
    n, st = _nodo_finto(True)
    pn.PerturbationNode._publish_status(n)
    assert st[0]["chamfer_pure_cm"] == 2.0 and st[0]["structural_cm"] == 1.5
    assert st[0]["point_ratio"] == 0.075 and st[0]["chamfer_cm"] == 2.5


def test_magnitudine_del_nodo_con_e_senza_termini():
    pn = pytest.importorskip("src.nodes.perturbation_node")
    from src.perturbations.perturbation_generator import PerturbationGenerator
    gen = PerturbationGenerator()
    a = np.random.default_rng(1).uniform(-5, 5, (500, 4))
    b = a[:450]
    off = SimpleNamespace(_perc_terms_on=False, _perc_terms=None)
    on = SimpleNamespace(_perc_terms_on=True, _perc_terms=None)
    p_off = pn.PerturbationNode._magnitude(off, gen, a, b, {})
    p_on = pn.PerturbationNode._magnitude(on, gen, a, b, {})
    assert p_off == p_on and off._perc_terms is None
    assert on._perc_terms["point_ratio"] == pytest.approx(0.1)


def test_orchestratore_media_dei_termini_nel_rollout():
    from src.optimization.attack_orchestrator import AttackNode
    n = SimpleNamespace(_n_cb=0, frames_out=0, node_frame_idx=-1, proc_ms_roll=[], chamfer=[],
                        perc_samples=[], victim_stamp_offset=0.0)
    for ch, pure in ((3.0, 2.0), (5.0, 4.0)):
        d = {"enabled": True, "victim_stream": False, "chamfer_cm": ch, "shift_frac": 0.1,
             "n_added": 1, "n_removed": 2, "chamfer_pure_cm": pure, "structural_cm": 1.0,
             "point_ratio": 0.05, "drift_cm": 0.0}
        AttackNode._status_cb(n, SimpleNamespace(data=json.dumps(d)))
    assert n.chamfer == [3.0, 5.0]
    assert [p["chamfer_pure_cm"] for p in n.perc_samples] == [2.0, 4.0]
    assert n.perc_samples[0]["structural_cm"] == 1.0 and n.perc_samples[0]["point_ratio"] == 0.05


# ---------------------------------------------------------------------------
# Sampling and single random with budget
# ---------------------------------------------------------------------------

DIR = (0, 1, 2, 7, 8, 9)


def _sparso_legacy(rng, n, n_search, off):
    # copy of the --random-sparse loop in the orchestrator
    X = np.tile(off, (n, 1))
    for i in range(n):
        kk = int(rng.integers(1, n_search + 1))
        idx = rng.choice(n_search, size=kk, replace=False)
        X[i, idx] = rng.uniform(-1.0, 1.0, size=kk)
    return X


def test_campionamento_sparso_uguale_al_legacy():
    off = np.array([0.0 if g in DIR else -1.0 for g in range(17)])
    a = wr.sample_genomes(np.random.default_rng(7), 20, 17, "sparse", off)
    b = _sparso_legacy(np.random.default_rng(7), 20, 17, off)
    assert np.array_equal(a, b)
    # one genome at a time = same sequence
    r = np.random.default_rng(7)
    c = np.vstack([wr.sample_genomes(r, 1, 17, "sparse", off) for _ in range(20)])
    assert np.array_equal(a, c)


def test_campionamento_uniforme_nei_limiti_e_sequenziale():
    a = wr.sample_genomes(np.random.default_rng(5), 64, 17, "uniform")
    assert a.shape == (64, 17) and a.min() >= -1.0 and a.max() <= 1.0
    assert np.array_equal(a, np.random.default_rng(5).uniform(-1.0, 1.0, size=(64, 17)))
    r = np.random.default_rng(5)
    c = np.vstack([wr.sample_genomes(r, 1, 17, "uniform") for _ in range(64)])
    assert np.array_equal(a, c)
    with pytest.raises(ValueError):
        wr.sample_genomes(r, 1, 17, "gauss")


def test_primo_ammissibile():
    log = []
    valid = [False, False, True, True]

    def evaluate(x):
        log.append({"ok": valid[len(log)]})
        return (-1.0, 2.0)

    it = iter(range(10))
    xs, fs, ok = wr.first_admissible(lambda: next(it), evaluate, lambda: log[-1]["ok"], 5)
    assert ok and xs == [0, 1, 2] and len(fs) == 3
    log.clear()
    valid[:] = [False] * 10
    it = iter(range(10))
    xs, fs, ok = wr.first_admissible(lambda: next(it), evaluate, lambda: log[-1]["ok"], 4)
    assert not ok and xs == [0, 1, 2, 3]


def test_primo_ammissibile_con_la_regola_della_selezione():
    # same rule as pareto_maxdamage: the first valid one within the budget
    log = []
    recs = [_rec(0, 0.2, 12.0), _rec(1, 0.1, 4.0, twin_filtered=True), _rec(2, 0.05, 8.0)]
    it = iter(range(3))
    xs, _, ok = wr.first_admissible(lambda: next(it), lambda x: log.append(recs[x]),
                                    lambda: candidato_valido(log[-1], 10.0), 5)
    assert ok and xs == [0, 1, 2]
    _, scelto = fronte_pareto(log, budget=10.0)
    assert scelto["idx"] == 2


# ---------------------------------------------------------------------------
# Campaign script: only the argument block (no start)
# ---------------------------------------------------------------------------

def _blocco_campagna(tmp_path, env_extra, arm="random", reps=3):
    src = (ROOT / "scripts" / "run_campaign.sh").read_text().splitlines()
    fine = next(i for i, l in enumerate(src) if l.startswith("if pgrep -f"))
    seme = next(l for l in src if "SEED_I=" in l).strip()
    dir_riga = next(l for l in src if l.strip().startswith("random|nsga3)") and "DIR=" in l)
    corpo = "\n".join(src[:fine])
    corpo = corpo.replace('cd "$ROOT"', ': cd')
    corpo += "\nSIGMA=-\ncase \"$ARM\" in\n" + dir_riga + "\n  *) DIR=\"${ARM}_g${GROUP}${MODE_SUFFIX}\" ;;\nesac\n"
    corpo += ("echo \"DIR=$DIR\"\necho \"FLAGS=$SERIEB_FLAGS\"\n"
              "echo \"MOLA=${MOLA_LO_DEBUG_ICP_QUALITY:-}\"\n"
              "echo \"NODE=${NODE_SB_FLAGS-}\"\necho \"DMIN=$MIN_LATENCY_MS\"\n"
              f"for i in $(seq 1 {reps}); do {seme}; echo \"SEED=$SEED_I\"; done\n")
    f = tmp_path / "blocco.sh"
    f.write_text(corpo)
    env = {k: v for k, v in os.environ.items()
           if k not in ("SERIE_B", "WINDOW_RNG", "SEEDS", "ICP_QUALITY_LOG", "PERC_TERMS",
                        "SAMPLING", "RANDOM_FIRST", "TWIN_FILTER", "MOLA_LO_DEBUG_ICP_QUALITY",
                        "CHAMFER_BUDGET", "MAX_DRIFT_LAG_S", "RUN_SUFFIX", "MIN_LATENCY_MS", "FAST_OPS",
                        "REPORT_MATCHED", "SYNC_WAIT", "SYNC_GUARD", "NODE_EXTRA", "ORCH_EXTRA")}
    env.update(env_extra)
    r = subprocess.run(["bash", str(f), "straight", arm, "3", str(reps), "32", "16"],
                       capture_output=True, text=True, env=env, cwd=tmp_path)
    out = dict(l.split("=", 1) for l in r.stdout.splitlines() if "=" in l and l.split("=", 1)[0] in
               ("DIR", "FLAGS", "MOLA", "NODE", "DMIN"))
    out["SEEDS"] = [l.split("=", 1)[1] for l in r.stdout.splitlines() if l.startswith("SEED=")]
    out["rc"] = r.returncode
    out["stdout"] = r.stdout
    return out


def test_campagna_default_invariato(tmp_path):
    o = _blocco_campagna(tmp_path, {})
    assert o["rc"] == 0
    assert o["DIR"] == "random_g3_p32x16_sel2_warm" and o["FLAGS"] == "" and o["MOLA"] == ""
    assert o["SEEDS"] == ["1001", "1002", "1003"]


def test_campagna_serie_b_au_pil3(tmp_path):
    # without SERIE_B: no new option, D_min 85
    o = _blocco_campagna(tmp_path, {}, arm="nsga3")
    assert o["rc"] == 0 and o["FLAGS"] == "" and o["NODE"] == "" and o["DMIN"] == "85"
    # with SERIE_B=1: D_min 140, --fast-ops and --report-matched to the node, publisher with guard
    o = _blocco_campagna(tmp_path, {"SERIE_B": "1"}, arm="nsga3")
    assert o["rc"] == 0 and o["DMIN"] == "140"
    assert o["NODE"] == " --fast-ops --report-matched"
    # then --select-only-valid --perceptibility chamfer
    assert " --sync-wait publisher --sync-guard" in o["FLAGS"]
    assert o["FLAGS"].endswith(" --select-only-valid --perceptibility chamfer")
    # kept drift (_rkeep), operator version I
    assert o["DIR"] == "nsga3_g3_p32x16_rkeep_sel2_warm_serieB_s1001-2001-3001_pt1_opinsieme"
    # every default can still be overridden
    o = _blocco_campagna(tmp_path, {"SERIE_B": "1", "MIN_LATENCY_MS": "85", "FAST_OPS": "0", "REPORT_MATCHED": "0",
                                    "SYNC_WAIT": "graph", "SYNC_GUARD": "0"}, arm="nsga3")
    assert o["rc"] == 0 and o["DMIN"] == "85" and o["NODE"] == ""
    assert "--sync-wait graph" in o["FLAGS"] and "--sync-guard" not in o["FLAGS"]
    # publisher without --report-matched to the node: rejected
    o = _blocco_campagna(tmp_path, {"SERIE_B": "1", "REPORT_MATCHED": "0"}, arm="nsga3")
    assert o["rc"] == 1 and "richiede REPORT_MATCHED=1" in o["stdout"]
    o = _blocco_campagna(tmp_path, {"SERIE_B": "1", "REPORT_MATCHED": "0", "NODE_EXTRA": "--report-matched"},
                         arm="nsga3")
    assert o["rc"] == 0
    o = _blocco_campagna(tmp_path, {"FAST_OPS": "si"}, arm="nsga3")
    assert o["rc"] == 1 and "FAST_OPS sconosciuto" in o["stdout"]


def test_campagna_serie_b(tmp_path):
    o = _blocco_campagna(tmp_path, {"SERIE_B": "1"})
    assert o["rc"] == 0
    # with SERIE_B=1 the name also carries seeds, terms and operators
    assert o["DIR"].endswith("_seedseq_serieB_s1001-2001-3001_pt1_opinsieme")
    assert "--window-rng seedseq" in o["FLAGS"]
    assert o["MOLA"] == "1" and o["SEEDS"] == ["1001", "2001", "3001"]
    o = _blocco_campagna(tmp_path, {"SERIE_B": "1"}, arm="nsga3")
    assert "--window-rng" not in o["FLAGS"] and o["SEEDS"] == ["1001", "2001", "3001"]
    o = _blocco_campagna(tmp_path, {"SERIE_B": "1", "RANDOM_FIRST": "20", "TWIN_FILTER": "am2",
                                    "SAMPLING": "uniform", "SELECT": "pareto_maxdamage"})
    assert "--random-first-admissible 20" in o["FLAGS"] and "--twin-filter am2" in o["FLAGS"]
    assert "--sampling uniform" in o["FLAGS"]
    assert o["DIR"].endswith("_sel3_warm_seedseq_sampuniform_first20_tfam2_serieB_s1001-2001-3001_pt1_opinsieme")


def test_campagna_errori(tmp_path):
    assert _blocco_campagna(tmp_path, {"RANDOM_FIRST": "5"}, arm="nsga3")["rc"] == 1
    assert _blocco_campagna(tmp_path, {"TWIN_FILTER": "xx"})["rc"] == 1
    assert _blocco_campagna(tmp_path, {"SEEDS": "1001 2001"}, reps=3)["rc"] == 1
    assert _blocco_campagna(tmp_path, {"ICP_QUALITY_LOG": "2"})["rc"] == 1
