#!/usr/bin/env python3
"""
Orchestrator integration checks: first-admissible random search, discard counts, transport,
warm start, frame origin, dropout limits, run outcomes. No ROS or Isaac.

    python3 -m pytest tests/test_integrazione_A.py -q
"""

import json
import math
import os
import shutil
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


def _ao():
    from src.optimization import attack_orchestrator as ao
    return ao


def riga(step, t, q=0.99, iters=1, term="Stalled", good=1):
    ts = datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y/%m/%d,%H:%M:%S.%f")
    return (f"[LidarOdometry] pathStep={step} timestamp={ts} goodness={q:.3f} minRequired=0.200 "
            f"iters={iters} termReason=IterTermReason::{term} isGood={good} "
            f"adapt_thres_sigma=0.1975 consecutive_bad=0\n")


def _rec(idx, damage, chamfer, status="ok", **kw):
    r = {"idx": idx, "status": status, "counted": True, "admissible": True,
         "twin_misaligned": False, "damage": damage, "chamfer": chamfer,
         "genome": [float(idx)] * 17}
    r.update(kw)
    return r


# ---------------------------------------------------------------------------
# Combinations of --random-first-admissible, from the real main
# ---------------------------------------------------------------------------

def _main(monkeypatch, argv):
    ao = _ao()
    monkeypatch.setattr(sys, "argv", ["attack_orchestrator.py"] + argv)
    return ao.main()


def test_main_rifiuta_random_first_senza_pareto(monkeypatch, capsys):
    assert _main(monkeypatch, ["--search", "random", "--random-first-admissible", "5"]) == 2
    assert "richiede --select pareto_maxdamage" in capsys.readouterr().out
    assert _main(monkeypatch, ["--search", "random", "--select", "ratio",
                               "--random-first-admissible", "5"]) == 2


def test_main_rifiuta_k_negativo_e_nsga3(monkeypatch, capsys):
    assert _main(monkeypatch, ["--search", "random", "--select", "pareto_maxdamage",
                               "--random-first-admissible", "-1"]) == 2
    assert ">= 0" in capsys.readouterr().out
    assert _main(monkeypatch, ["--search", "nsga3", "--select", "pareto_maxdamage",
                               "--random-first-admissible", "3"]) == 2


def test_controllo_combinazioni_ammesse():
    ao = _ao()
    a = SimpleNamespace(random_first_admissible=0, search="nsga3", select="maxdamage")
    assert ao.check_random_first_args(a) is None              # default unchanged
    a = SimpleNamespace(random_first_admissible=4, search="random", select="pareto_maxdamage")
    assert ao.check_random_first_args(a) is None


# ---------------------------------------------------------------------------
# Single random as main calls it (random_first_run, random_first_choice)
# ---------------------------------------------------------------------------

def _scenario(valid_seq):
    """eval_log and functions as in main: evaluate appends a record."""
    log = []
    it = iter(range(100))
    seq = iter(valid_seq)

    def evaluate(x):
        v = next(seq)
        log.append(_rec(len(log), 0.01 * (len(log) + 1), 1.0,
                        status="ok" if v else "stopped_early(0.1m)"))
        return (-0.01, 1.0)

    return log, (lambda: next(it)), evaluate


def test_random_first_primo_valido_e_scelta():
    ao = _ao()
    log, draw, ev = _scenario([False, False, True, True])
    is_valid = lambda e: candidato_valido(e, 10.0)  # noqa: E731
    rf, xs, fs = ao.random_first_run(draw, ev, log, is_valid, 5)
    assert rf == {"k_max": 5, "n_tries": 3, "found": True, "eval_idx": 2}
    assert xs == [0, 1, 2] and len(log) == 3
    fronte, scelto = fronte_pareto(log, budget=10.0)
    ch = ao.random_first_choice(fronte, log, rf)
    assert ch is scelto and ch["idx"] == 2
    # second call without re-evaluations: no extra attempt
    rf, xs, _ = ao.random_first_run(draw, ev, log, is_valid, 5, rf)
    assert xs == [] and rf["eval_idx"] == 2 and "n_tries_resumed" not in rf


def test_random_first_nessun_valido():
    ao = _ao()
    log, draw, ev = _scenario([False] * 4)
    rf, xs, _ = ao.random_first_run(draw, ev, log, lambda e: candidato_valido(e), 4)
    assert rf["found"] is False and rf["eval_idx"] is None and rf["n_tries"] == 4
    assert ao.random_first_choice([], log, rf) is None


def test_random_first_riprende_dopo_il_riavvio_dell_adattatore():
    ao = _ao()
    log, draw, ev = _scenario([False, True, False, False, True])
    is_valid = lambda e: candidato_valido(e)  # noqa: E731
    rf, _, _ = ao.random_first_run(draw, ev, log, is_valid, 5)
    assert rf["eval_idx"] == 1 and rf["n_tries"] == 2
    # --adapter-events block of main: the valid one is invalidated and re-evaluated
    # (invalid outcome)
    log[1]["admissible"] = False
    log[1]["counted"] = False
    log[1]["measured_during_adapter_degradation"] = True
    ev(None)                                  # re-evaluation: third value of the sequence (False)
    log[1]["superseded_by"] = 2
    rf, xs, _ = ao.random_first_run(draw, ev, log, is_valid, 5, rf)
    assert xs == [2, 3]                       # resume: two new attempts, the second valid
    assert rf["found"] and rf["eval_idx"] == 4 and rf["n_tries"] == 4 and rf["n_tries_resumed"] == 2
    fronte, _ = fronte_pareto(log)
    assert ao.random_first_choice(fronte, log, rf)["idx"] == 4


def test_random_first_primo_valido_fuori_dal_fronte():
    ao = _ao()
    log = [_rec(0, 0.02, 5.0), _rec(1, 0.05, 1.0)]       # the first is dominated
    fronte, scelto = fronte_pareto(log)
    ch = ao.random_first_choice(fronte, log, {"eval_idx": 0})
    assert scelto["idx"] == 1 and ch["idx"] == 0 and ch["outside_front"] is True
    assert ch["in_pymoo_front"] is None and ch["damage"] == 0.02


# ---------------------------------------------------------------------------
# Legacy sampling of main against window_record
# ---------------------------------------------------------------------------

DIR = (0, 1, 2, 7, 8, 9)


def test_legacy_di_main_uguale_al_campionamento_sparso_e_uniforme():
    ao = _ao()
    off = np.array([0.0 if g in DIR else -1.0 for g in range(17)])
    a = ao.legacy_random_genomes(np.random.default_rng(7), 20, 17, True, off)
    b = wr.sample_genomes(np.random.default_rng(7), 20, 17, "sparse", off)
    assert np.array_equal(a, b)
    a = ao.legacy_random_genomes(np.random.default_rng(7), 20, 17, False, off)
    assert np.array_equal(a, np.random.default_rng(7).uniform(-1.0, 1.0, size=(20, 17)))
    assert np.array_equal(a, wr.sample_genomes(np.random.default_rng(7), 20, 17, "uniform"))


def test_main_usa_le_funzioni_provate():
    # main calls exactly these functions (no copy of the loop left in main)
    import inspect
    src = inspect.getsource(_ao().main)
    assert "legacy_random_genomes(rng, args.pop * args.gen" in src
    # random_first_run is called by random_first_search and random_first_resume, which are
    # called by main
    assert "random_first_search(" in src and "random_first_resume(" in src
    assert "random_first_choice(" in src
    assert "rng.choice(n_search" not in src


# ---------------------------------------------------------------------------
# Separate discards and definitions
# ---------------------------------------------------------------------------

def _ts(t):
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y/%m/%d,%H:%M:%S.%f")


def test_scarti_separati_nel_conteggio(tmp_path):
    log = tmp_path / "mola_victim.log"
    righe = [riga(1, 10.0), riga(2, 10.1),
             "[DEBUG] onLidarImpl: dropping observation, for 0.050000< `min_time_between_scans`=0.080000.\n",
             f"[WARN] Observation discarded as non-valid for pathStep=3, timestamp={_ts(10.3)} UTC\n",
             riga(4, 10.4), riga(5, 10.5)]
    log.write_text("avvio\n" + "".join(righe))
    rows = wr.read_printf(log)
    disc = wr.read_discards(log)
    assert disc["invalid_stamps"] == [10.3] and disc["min_time_lines"] == [4]
    c = wr.scan_counts(rows, 10.0, 10.5, discards=disc)
    assert c["expected"] == 6 and c["processed"] == 4 and c["lost"] == 2
    assert c["discarded_invalid"] == 1 and c["dropped_min_time"] == 1
    assert c["lost_unexplained"] == 0 and c["excess"] == 0
    # without DEBUG lines in the log the min_time discard is not measurable: None
    log.write_text("avvio\n" + "".join(r for r in righe if "dropping" not in r))
    c = wr.scan_counts(wr.read_printf(log), 10.0, 10.5, discards=wr.read_discards(log))
    assert c["dropped_min_time"] is None and c["lost_unexplained"] == 1
    # without discards the previous fields, unchanged
    c0 = wr.scan_counts(rows, 10.0, 10.5)
    assert "discarded_invalid" not in c0 and c0["lost"] == 2


def test_definizioni_nel_registro_della_vittima(tmp_path):
    ao = _ao()
    log = tmp_path / "mola_victim.log"
    log.write_text("x\n" + "".join(riga(i, 10.0 + 0.1 * i) for i in range(6)))
    node = SimpleNamespace(last_trace=[], last_drive_stamps=(10.2, 10.5), last_victim_open_stamp=10.0)
    rec = ao._victim_window_record(node, SimpleNamespace(log_path=log), None)
    assert "expected" in rec["scans_definition"] and "lost" in rec["scans_definition"]
    assert rec["scans_drive"]["discarded_invalid"] == 0 and rec["scans_drive"]["dropped_min_time"] is None


# ---------------------------------------------------------------------------
# Campaign script, argument block and command, transport
# ---------------------------------------------------------------------------

CAMPAGNA = ROOT / "scripts" / "run_campaign.sh"
CAMPAGNA_BASE = ROOT / "tests" / "dati" / "run_campaign_base_2026-10-23.sh"
VARS_NUOVE = ("SERIE_B", "WINDOW_RNG", "SEEDS", "ICP_QUALITY_LOG", "PERC_TERMS", "SAMPLING",
              "RANDOM_FIRST", "TWIN_FILTER", "MOLA_LO_DEBUG_ICP_QUALITY", "CHAMFER_BUDGET",
              "MAX_DRIFT_LAG_S", "RUN_SUFFIX", "SELECT", "ORCH_EXTRA", "OPERATOR_MODE",
              "TRANSPORT", "LOCALHOST_ONLY", "MOLA_LIDAR_TOPIC_TYPE", "ROS_AUTOMATIC_DISCOVERY_RANGE",
              "SERIEB_TRANSPORT", "WINDOW_STOP", "ATTACK_FROM")


def _comando(script, tmp_path, env_extra, arm="random", reps=3, sigma="0.02"):
    """Run the script's argument block and, with the lines of its loop (DIR, seed,
    orchestrator command), print folder and command for each repetition. No process is
    started."""
    src = Path(script).read_text().splitlines()
    fine = next(i for i, l in enumerate(src) if l.startswith("if pgrep -f"))
    corpo = [l.replace('cd "$ROOT"', ': cd') for l in src[:fine]]
    dir_righe = [l for l in src if 'DIR="' in l and l.strip().split(")")[0] in
                 ("none", "gaussian", "random|nsga3", "fixed")]
    seme = next((l for l in src if "SEED_I=" in l), ":").strip()   # the base uses $((1000 + i))
    i0 = next(i for i, l in enumerate(src) if "exec python3 -u src/optimization/attack_orchestrator.py" in l)
    i1 = next(i for i in range(i0, len(src)) if "$ORCH_EXTRA )" in src[i])
    cmd = "\n".join(src[i0:i1 + 1])
    cmd = cmd.replace("( trap - INT; exec python3 -u src/optimization/attack_orchestrator.py", "echo CMD")
    cmd = cmd.replace("$ORCH_EXTRA ) \\", "$ORCH_EXTRA")
    corpo += [f'SIGMA="{sigma}"', 'case "$ARM" in', *dir_righe, "esac", 'echo "DIR=$DIR"',
              f'for i in $(seq 1 "$REPS"); do', seme, 'OUT="X/rep_$i"', cmd, "done",
              'echo "MOLA=${MOLA_LO_DEBUG_ICP_QUALITY:-}"', 'echo "PT=${PERC_TERMS:-}"',
              'echo "QOS=${MOLA_LIDAR_TOPIC_TYPE:-}" | tr "\\n" "~"; echo',
              'echo "RANGE=${ROS_AUTOMATIC_DISCOVERY_RANGE:-}"']
    f = tmp_path / ("blocco_" + Path(script).stem + ".sh")
    f.write_text("\n".join(corpo) + "\n")
    env = {k: v for k, v in os.environ.items() if k not in VARS_NUOVE}
    env.update(env_extra)
    r = subprocess.run(["bash", str(f), "straight", arm, "3", str(reps), "32", "16"],
                       capture_output=True, text=True, env=env, cwd=tmp_path)
    return r.returncode, r.stdout


AMBIENTI_SERIE_A = [
    {},
    {"CHAMFER_BUDGET": "70", "MAX_DRIFT_LAG_S": "0.5", "SELECT": "pareto_maxdamage",
     "ORCH_EXTRA": "--random-sparse --damage-ref baseline --baseline-path b.csv"},
    {"SELECT": "maxdamage", "MAX_DRIFT_LAG_S": "0.5", "ORCH_EXTRA": "--random-sparse --admissibility none"},
    {"ATTACK_FROM": "motion", "RUN_SUFFIX": "_x", "WARM_START": "false"},
]


@pytest.mark.parametrize("arm", ["none", "gaussian", "random", "nsga3", "fixed"])
@pytest.mark.parametrize("amb", range(len(AMBIENTI_SERIE_A)))
def test_campagna_uguale_alla_base_senza_variabili_nuove(tmp_path, arm, amb):
    env = AMBIENTI_SERIE_A[amb]
    rb, ob = _comando(CAMPAGNA_BASE, tmp_path, env, arm=arm)
    rn, on = _comando(CAMPAGNA, tmp_path, env, arm=arm)
    assert rb == 0 and rn == 0
    # base without SERIE_B repetitions: the MOLA/PT/QOS/RANGE line is empty in both
    assert on == ob
    assert "CMD" in on and "DIR=" in on


def test_campagna_serie_b_nel_nome(tmp_path):
    rc, o = _comando(CAMPAGNA, tmp_path, {"SERIE_B": "1"}, arm="nsga3")
    assert rc == 0
    assert "DIR=nsga3_g3_p32x16_rkeep_sel2_warm_serieB_s1001-2001-3001_pt1_opinsieme\n" in o
    assert "--seed 2001" in o and "MOLA=1" in o
    rc, o2 = _comando(CAMPAGNA, tmp_path, {"SERIE_B": "1", "PERC_TERMS": "0"}, arm="nsga3")
    assert "_pt0_opinsieme" in o2
    rc, o3 = _comando(CAMPAGNA, tmp_path, {"SERIE_B": "1", "SEEDS": "7 8 9"}, arm="nsga3")
    assert "_serieB_s7-8-9_" in o3
    # without SERIE_B seeds and explicit terms still enter the name
    rc, o4 = _comando(CAMPAGNA, tmp_path, {"SEEDS": "7 8 9", "PERC_TERMS": "1"}, arm="nsga3")
    assert "DIR=nsga3_g3_p32x16_sel2_warm_s7-8-9_pt1\n" in o4


def test_campagna_semi_validati(tmp_path):
    assert _comando(CAMPAGNA, tmp_path, {"SEEDS": "1001 10x 3001"})[0] == 1
    assert _comando(CAMPAGNA, tmp_path, {"SEEDS": "1001 1001 3001"})[0] == 1
    assert _comando(CAMPAGNA, tmp_path, {"SEEDS": "-5 1 2"})[0] == 1
    assert _comando(CAMPAGNA, tmp_path, {"SEEDS": "1001  2001 3001"})[0] == 0


def test_campagna_icp_quality_forzata(tmp_path):
    rc, o = _comando(CAMPAGNA, tmp_path, {"ICP_QUALITY_LOG": "1", "MOLA_LO_DEBUG_ICP_QUALITY": "0"})
    assert rc == 0 and "MOLA=1" in o and "portata a 1" in o


def test_campagna_random_first_richiede_pareto(tmp_path):
    assert _comando(CAMPAGNA, tmp_path, {"RANDOM_FIRST": "20"})[0] == 1
    rc, o = _comando(CAMPAGNA, tmp_path, {"RANDOM_FIRST": "20", "SELECT": "pareto_maxdamage"})
    assert rc == 0 and "--random-first-admissible 20" in o and "_first20" in o


def test_campagna_operatori(tmp_path):
    assert _comando(CAMPAGNA, tmp_path, {"OPERATOR_MODE": "coherent"})[0] == 1
    assert _comando(CAMPAGNA, tmp_path, {"OPERATOR_MODE": "random"})[0] == 0


QOS_ATTESA = "PointCloud2\n          qos:\n            reliability: reliable\n            depth: 50"


def test_campagna_trasporto(tmp_path):
    rc, o = _comando(CAMPAGNA, tmp_path, {"TRANSPORT": "reliable", "LOCALHOST_ONLY": "1"}, arm="nsga3")
    assert rc == 0
    assert "QOS=" + QOS_ATTESA.replace("\n", "~") in o              # exactly 10 and 12 spaces
    assert "RANGE=LOCALHOST" in o and "_reliable_localhost" in o
    rc, o = _comando(CAMPAGNA, tmp_path, {"TRANSPORT": "default"})
    assert rc == 0 and "QOS=~" in o and "RANGE=\n" in o
    assert _comando(CAMPAGNA, tmp_path, {"TRANSPORT": "best"})[0] == 1
    assert _comando(CAMPAGNA, tmp_path, {"LOCALHOST_ONLY": "2"})[0] == 1


def test_campagna_controlli_prima_del_lancio():
    src = CAMPAGNA.read_text()
    i_pre = src.index("scripts/controlla_qos_yaml.py --topic")
    assert src.index('if [[ "$TRANSPORT" == "reliable" ]]; then\n  for t in') < i_pre < src.index("isaac_start()")
    assert "/carter/lidar_victim /carter/lidar_attacker" in src


# ---------------------------------------------------------------------------
# Transport: offline parser check (mola-yaml-parser, no ROS)
# ---------------------------------------------------------------------------

PARSER = shutil.which("mola-yaml-parser")


def _qos_yaml(value, topic):
    env = {k: v for k, v in os.environ.items() if k != "MOLA_LIDAR_TOPIC_TYPE"}
    if value is not None:
        env["MOLA_LIDAR_TOPIC_TYPE"] = value
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "controlla_qos_yaml.py"), "--topic", topic],
                       capture_output=True, text=True, env=env)
    return r.returncode, r.stdout


@pytest.mark.skipif(PARSER is None, reason="mola-yaml-parser non installato")
def test_parser_qos_iniettata():
    assert _qos_yaml(QOS_ATTESA, "/carter/lidar_victim")[0] == 0
    assert _qos_yaml(QOS_ATTESA, "/carter/lidar_attacker")[0] == 0
    # 8 spaces: the parser truncates silently, the check stops it
    rc, o = _qos_yaml(QOS_ATTESA.replace("          qos", "        qos").replace("            ", "          "),
                      "/carter/lidar_attacker")
    assert rc == 1 and "nessuna voce" in o
    # 12 spaces: the parser fails
    assert _qos_yaml(QOS_ATTESA.replace("\n  ", "\n    "), "/carter/lidar_victim")[0] == 1
    # no injection: no qos
    rc, o = _qos_yaml(None, "/carter/lidar_victim")
    assert rc == 1 and "senza blocco qos" in o
    # different reliability
    assert _qos_yaml(QOS_ATTESA.replace("reliable", "best_effort"), "/carter/lidar_victim")[0] == 1


def test_confronto_qos_su_documento_sintetico():
    sys.path.insert(0, str(ROOT / "scripts"))
    import controlla_qos_yaml as cq
    ref = {"modules": [{"params": {"subscribe": [{"topic": "/a", "msg_type": "PointCloud2"}, {"topic": "/b"}]}},
                       {"params": None}, {"params": "x"}]}
    doc = json.loads(json.dumps(ref))
    doc["modules"][0]["params"]["subscribe"][0]["qos"] = {"reliability": "reliable", "depth": 50}
    assert cq.confronta(doc, ref, "/a", "reliable", 50) == []
    doc["modules"][0]["params"]["subscribe"][1]["topic"] = "/c"
    assert "resto del documento" in cq.confronta(doc, ref, "/a", "reliable", 50)[0]


# ---------------------------------------------------------------------------
# Transport: graph check with a fake ros2 (the chain is not queried)
# ---------------------------------------------------------------------------

def _blocco_endpoint(node, ns, ep, rel):
    return (f"Node name: {node}\nNode namespace: {ns}\nTopic type: sensor_msgs/msg/PointCloud2\n"
            f"Topic type hash: RIHS01_x\nEndpoint type: {ep}\nGID: 01.02\nQoS profile:\n"
            f"  Reliability: {rel}\n  History (Depth): KEEP_LAST (50)\n  Durability: VOLATILE\n"
            f"  Lifespan: Infinite\n  Deadline: Infinite\n  Liveliness: AUTOMATIC\n"
            f"  Liveliness lease duration: Infinite\n\n")


def _grafo(tmp_path, sub_rel, n_sub=1):
    out = ("Type: sensor_msgs/msg/PointCloud2\n\nPublisher count: 1\n\n"
           + _blocco_endpoint("perturbation_node", "/", "PUBLISHER", "RELIABLE")
           + f"Subscription count: {n_sub}\n\n"
           + "".join(_blocco_endpoint("mola_bridge_ros2", "/victim", "SUBSCRIPTION", sub_rel)
                     for _ in range(n_sub)))
    (tmp_path / "info.txt").write_text(out)
    fake = tmp_path / "bin" / "ros2"
    fake.parent.mkdir(exist_ok=True)
    fake.write_text(f"#!/usr/bin/env bash\ncat {tmp_path / 'info.txt'}\n")
    fake.chmod(0o755)
    env = dict(os.environ, PATH=f"{fake.parent}:{os.environ['PATH']}", OUT_DIR=str(tmp_path),
               TOPICS="/carter/lidar_victim")
    r = subprocess.run(["bash", str(ROOT / "scripts" / "controlla_qos_grafo.sh")],
                       capture_output=True, text=True, env=env)
    return r.returncode, r.stdout


def test_grafo_reliable_best_effort_assente(tmp_path):
    rc, o = _grafo(tmp_path, "RELIABLE")
    assert rc == 0 and "/victim/mola_bridge_ros2=RELIABLE" in o
    rc, o = _grafo(tmp_path, "BEST_EFFORT")
    assert rc == 1 and "non RELIABLE" in o
    rc, o = _grafo(tmp_path, "RELIABLE", n_sub=0)
    assert rc == 1 and "nessuna subscription" in o


# ---------------------------------------------------------------------------
# Warm-start composition (function called by main)
# ---------------------------------------------------------------------------

def test_partenza_calda_copie_e_perturbazioni():
    ao = _ao()
    rng = np.random.default_rng(0)
    pool = [(rng.uniform(-1, 1, 17), 1.0, float(c)) for c in (3.0, 1.0, 2.0)]
    rows, c = ao.warm_front_rows(pool, 8, np.random.default_rng(1001 + 1 + 7919), 17)
    assert len(rows) == 8
    assert c == {"warm_start_from_prev": 8, "warm_front_n": 3, "warm_mut_n": 5, "warm_mut_identical_n": 0}
    srt = sorted(pool, key=lambda t: t[2])
    assert all(np.array_equal(rows[i], srt[i][0]) for i in range(3))     # exact copies, for Chamfer
    assert not any(np.array_equal(r, t[0]) for r in rows[3:] for t in pool)
    # large front: copies only
    big = [(rng.uniform(-1, 1, 17), 1.0, float(i)) for i in range(40)]
    rows, c = ao.warm_front_rows(big, 16, np.random.default_rng(5), 17)
    assert c["warm_front_n"] == 16 and c["warm_mut_n"] == 0 and c["warm_start_from_prev"] == 16


def test_partenza_calda_perturbazione_uguale_al_fronte():
    # genes at the limit: the clipped noise may leave the row identical
    g = np.full(17, 1.0)
    c = wr.warm_start_counts([g], [np.clip(g + 0.2, -1, 1), g * 0.5], [g])
    assert c["warm_mut_identical_n"] == 1 and c["warm_mut_n"] == 2


def test_partenza_calda_main_non_azzera_piu():
    import inspect
    src = inspect.getsource(_ao().main)
    assert "warm_front_rows(warm_pool, half, rng_w, n_search)" in src
    assert '"warm_start_from_prev": (warm_counts or {})' in src


# ---------------------------------------------------------------------------
# Frame origin (--origin-from-estimate)
# ---------------------------------------------------------------------------

def test_origine_dalla_stima():
    true = (-5.99, -1.0, 0.3)
    est = (0.011, 0.001, 0.002)
    o = wr.origin_from_estimate(true, est)
    w = wr.compose_pose(o, est)
    assert w[0] == pytest.approx(true[0], abs=1e-12) and w[1] == pytest.approx(true[1], abs=1e-12)
    assert w[2] == pytest.approx(true[2], abs=1e-12)
    ao = _ao()
    assert ao.AttackNode.compose(o, est) == wr.compose_pose(o, est)     # same formula as main
    r_true = wr.origin_record("true", true, est, true)
    r_est = wr.origin_record("estimate", true, est, o)
    # legacy: the drift logged from the estimate becomes offset (1.1 cm); with estimate 0
    assert r_true["offset"]["d_cm"] == pytest.approx(100 * math.hypot(0.011, 0.001), abs=1e-3)
    assert r_true["offset_other_mode"]["d_cm"] == pytest.approx(0.0, abs=1e-9)
    assert r_est["offset"]["d_cm"] == pytest.approx(0.0, abs=1e-9)
    assert r_est["offset_other_mode"]["d_cm"] == r_true["offset"]["d_cm"]


class _NodoAvvio:
    def __init__(self, est, att_est, true):
        self.est_raw, self.attacker_est_raw, self.true = est, att_est, true
        self.est = (0.0, 0.0, 0.0)
        self.est_t = 0.0

    def set_victim_attack(self, on): pass
    def set_victim_stream(self, on): pass
    def drain(self): return True
    def spin(self, t): self.est = self.est or (0.0, 0.0, 0.0)
    def wait_converged(self, isaac, s): return 0.5, True
    def settled_pose(self, isaac): return self.true


def _avvio(mode):
    ao = _ao()
    v = ao.VictimRunner.__new__(ao.VictimRunner)
    v.run_origin = None
    v.origin_info = None
    v.spawn = lambda tag: None
    v.wait_ready = lambda check=None: None
    att = SimpleNamespace(run_origin=None, start=lambda node, o: None)
    node = _NodoAvvio((0.011, 0.0, 0.0), (0.004, 0.0, 0.0), (-5.99, -1.0, 0.0))
    # wait_converged / spin: the wait loop for the first estimate exits at once because est is set
    v.start(None, node, 5.0, attacker=att, origin_mode=mode)
    return v, att


def test_avvio_vittima_origine_legacy_e_dalla_stima(capsys):
    v, att = _avvio("true")
    assert v.run_origin == (-5.99, -1.0, 0.0) and att.run_origin == v.run_origin   # as before
    assert v.origin_info["victim"]["offset"]["d_cm"] == pytest.approx(1.1, abs=1e-6)
    assert v.origin_info["attacker"]["offset"]["d_cm"] == pytest.approx(0.4, abs=1e-6)
    v, att = _avvio("estimate")
    assert v.run_origin[0] == pytest.approx(-5.99 - 0.011) and att.run_origin[0] == pytest.approx(-5.99 - 0.004)
    assert v.origin_info["victim"]["offset"]["d_cm"] == pytest.approx(0.0, abs=1e-9)
    assert v.origin_info["victim"]["offset_other_mode"]["d_cm"] == pytest.approx(1.1, abs=1e-6)
    assert "origine victim (estimate)" in capsys.readouterr().out


def test_opzioni_nuove_spente_di_default():
    ao = _ao()
    src = Path(ao.__file__).read_text()
    assert '"--origin-from-estimate", action="store_true"' in src
    assert '"--fail-rc-on-no-pose", action="store_true"' in src
    assert 'origin_mode="estimate" if args.origin_from_estimate else "true"' in src


# ---------------------------------------------------------------------------
# Dropout with the node limits
# ---------------------------------------------------------------------------

def test_stampa_dei_parametri_con_il_tetto_del_nodo():
    ao = _ao()
    from src.perturbations.perturbation_generator import PerturbationGenerator
    g = np.full(17, -1.0)
    g[5] = -0.94261                       # value from a recorded run (w1)
    vecchio = PerturbationGenerator().encode_perturbation(g)["dropout_rate"]
    assert vecchio == pytest.approx(0.00086, abs=1e-5)                   # printed "+0,0009"
    # node without gen_limits in the status: node default, 0.15 limited to 0.05
    n = SimpleNamespace(node_gen_limits=None)
    assert ao.node_gen_limits(n)["max_dropout_rate"] == 0.05
    assert ao.print_generator(n).encode_perturbation(g)["dropout_rate"] == pytest.approx(0.00143, abs=1e-5)
    # limits read from the status
    nodo = PerturbationGenerator(max_dropout_rate=0.15)
    lim = {k: float(getattr(nodo, k)) for k in ("max_point_shift", "noise_std", "max_dropout_rate",
                                                 "max_ghost_points_ratio")}
    n = SimpleNamespace(node_gen_limits=lim)
    assert ao.node_gen_limits(n)["source"] == "status del nodo"
    p = ao.print_generator(n).encode_perturbation(g)
    assert p["dropout_rate"] == nodo.encode_perturbation(g)["dropout_rate"]
    assert PerturbationGenerator().get_genome_size() == ao.print_generator(n).get_genome_size() == 17


def test_status_del_nodo_con_i_limiti():
    pn = pytest.importorskip("src.nodes.perturbation_node")
    import importlib.util
    spec = importlib.util.spec_from_file_location("trs", ROOT / "tests" / "test_registro_serieB.py")
    trs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trs)
    from src.perturbations.perturbation_generator import PerturbationGenerator
    n, st = trs._nodo_finto(False)
    n.generator = PerturbationGenerator(max_dropout_rate=0.15)
    pn.PerturbationNode._publish_status(n)
    assert st[0]["gen_limits"]["max_dropout_rate"] == 0.05
    ao = _ao()
    a = SimpleNamespace(_n_cb=0, frames_out=0, node_frame_idx=-1, proc_ms_roll=[], chamfer=[],
                        perc_samples=[], victim_stamp_offset=0.0, node_gen_limits=None)
    ao.AttackNode._status_cb(a, SimpleNamespace(data=json.dumps(st[0])))
    assert a.node_gen_limits["max_dropout_rate"] == 0.05


# ---------------------------------------------------------------------------
# Victim without pose marked per window, exit code
# ---------------------------------------------------------------------------

def test_vittima_senza_posa_marcata_per_finestra():
    # sequence from run rep_03 (fixed_g3_sel2_H1.0_rif): w0-w2 applied ok,
    # w3-w7 nominal failed and victim victim_no_pose
    state = {"since": None, "any": False}
    recs = []
    for k, st in enumerate(["ok", "ok", "ok"] + ["victim_no_pose"] * 5):
        recs.append(wr.mark_victim_lost({"window": k, "victim_lost": False} if k < 3 else {"window": k},
                                        st, k, state))
    assert [r["victim_lost"] for r in recs] == [False] * 3 + [True] * 5
    assert all(r["victim_lost_since_window"] == 3 for r in recs[3:])
    assert "victim_lost_since_window" not in recs[0] and state["any"] is True
    # pose recovered: the series closes, a new loss has its own start window
    r = wr.mark_victim_lost({}, "ok", 8, state)
    assert r["victim_lost"] is False and state["since"] is None
    assert wr.mark_victim_lost({}, "victim_crash", 9, state)["victim_lost_since_window"] == 9
    # victim not used (vrestart): nothing changes
    assert wr.mark_victim_lost({"a": 1}, None, 10, state) == {"a": 1}


def test_codice_d_uscita():
    ao = _ao()
    lost = {"any": True}
    assert ao.exit_code(None, lost, False) == 0          # default unchanged
    assert ao.exit_code(None, lost, True) == 4
    assert ao.exit_code("stallo", lost, True) == 3
    assert ao.exit_code(None, {"any": False}, True) == 0


def test_main_marca_tutti_i_rami():
    import inspect
    src = inspect.getsource(_ao().main)
    # definition + 5 branches (skipped, none, watchdog, no_admissible, applied)
    assert src.count("_esito_finestra(") >= 6
    assert "return exit_code(stalled, lost_state, args.fail_rc_on_no_pose)" in src
