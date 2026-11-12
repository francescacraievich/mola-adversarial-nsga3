"""Campaign rules, with fake processes. With SERIE_B=1 run_campaign.sh uses D_min 140 ms, the
sync guard and the node per-frame log with delay (--frame-log, --frame-log-delay). "MOLA
notices" = the twin MOLA crashes or publishes no new pose for more than 1 s; consecutive
rejected ICPs are only logged. Chain check before closing a run as attack detected, with the
reason for excluded slow candidates. run_campaign.sh really runs, with fake python3, pgrep,
ros2 and sleep in PATH: no real process is started.
"""

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parents[1]
ORIGINALE = ROOT.parent / "originale"
PY = shutil.which("python3")
SLEEP = shutil.which("sleep")

TOLTE = ("SERIE_B", "WINDOW_RNG", "SEEDS", "ICP_QUALITY_LOG", "PERC_TERMS", "SAMPLING",
         "RANDOM_FIRST", "TWIN_FILTER", "MOLA_LO_DEBUG_ICP_QUALITY", "CHAMFER_BUDGET",
         "MAX_DRIFT_LAG_S", "RUN_SUFFIX", "MIN_LATENCY_MS", "FAST_OPS", "REPORT_MATCHED",
         "SYNC_WAIT", "SYNC_GUARD", "NODE_EXTRA", "ORCH_EXTRA", "SELECT", "TOPK",
         "TRANSPORT", "LOCALHOST_ONLY", "ISAAC_ARGS", "CAMPAIGN_DIR", "STRUCTURAL",
         "OPERATOR_MODE", "WARM_FRONT_AFTER_REEVAL", "WINDOW_STOP", "GOALS_OVERRIDE")


def _finti(bin_dir, reg):
    """Fake python3, pgrep, ros2 and sleep. python3 logs the arguments of the three
    campaign processes (one JSON line in reg) and passes the rest to the real one.
    """
    bin_dir.mkdir()
    py = f"""#!/bin/bash
case "$*" in
  *run_isaac_standalone.py*) echo "[standalone] pronto"; exec {SLEEP} 600 ;;
  *perturbation_node.py*) {PY} -c 'import json,sys; print(json.dumps(["nodo"]+sys.argv[1:]))' "$@" >> "{reg}"; exec {SLEEP} 600 ;;
  *attack_orchestrator.py*) {PY} -c 'import json,os,sys; a=sys.argv[1:]; os.makedirs(a[a.index("--out")+1], exist_ok=True); print(json.dumps(["orch"]+a))' "$@" >> "{reg}"; exit 0 ;;
esac
exec {PY} "$@"
"""
    files = {
        "python3": py,
        "pgrep": '#!/bin/bash\ncase "$*" in *add_intensity*) echo 1; exit 0 ;; esac\nexit 1\n',
        "ros2": '#!/bin/bash\necho "{\\"min_latency_ms\\": ${_LAT_FINTA}}"\n',
        "sleep": "#!/bin/bash\nexit 0\n",
    }
    for nome, testo in files.items():
        p = bin_dir / nome
        p.write_text(testo)
        p.chmod(p.stat().st_mode | stat.S_IEXEC)


def campagna(radice, tmp_path, env_extra, arm="nsga3"):
    """Runs run_campaign.sh of a copy with fake processes: returns the node and
    orchestrator arguments and the run folder.
    """
    tmp_path = tmp_path / f"c_{radice.name}_{len(list(tmp_path.iterdir()))}"
    tmp_path.mkdir()
    reg = tmp_path / "argomenti.jsonl"
    _finti(tmp_path / "bin", reg)
    env = {k: v for k, v in os.environ.items() if k not in TOLTE}
    env.update({"PATH": f"{tmp_path / 'bin'}:{env['PATH']}", "CAMPAIGN_DIR": str(tmp_path / "camp")})
    env.update(env_extra)
    lat = env.get("MIN_LATENCY_MS") or ("140" if env.get("SERIE_B") == "1" else "85")
    env["_LAT_FINTA"] = env.get("_LAT_FINTA", lat)
    r = subprocess.run(["bash", str(radice / "scripts" / "run_campaign.sh"), "straight", arm, "3", "1", "4", "2"],
                       capture_output=True, text=True, env=env, cwd=tmp_path, timeout=120)
    righe = [json.loads(l) for l in reg.read_text().splitlines()] if reg.exists() else []
    nodo = next((r_[1:] for r_ in righe if r_[0] == "nodo"), None)
    orch = next((r_[1:] for r_ in righe if r_[0] == "orch"), None)
    out = None
    if orch is not None:
        out = orch[orch.index("--out") + 1]
    return r, nodo, orch, out


def _valore(argv, opz):
    return argv[argv.index(opz) + 1] if opz in argv else None


# --- D_min 140, sync guard, per-frame log ---

@pytest.mark.parametrize("arm", ["nsga3", "none"])
def test_serie_b_dmin_140_guardia_e_frame_log(tmp_path, arm):
    r, nodo, orch, out = campagna(ROOT, tmp_path, {"SERIE_B": "1", "SEEDS": "1001"}, arm)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert _valore(nodo, "--min-latency-ms") == "140"
    assert "--sync-guard" in orch and _valore(orch, "--sync-wait") == "publisher"
    assert _valore(nodo, "--frame-log") == f"{out}/frame_log.csv"
    assert nodo.count("--frame-log-delay") == 1 and nodo.count("--frame-log") == 1
    # comando.txt equal to the arguments passed
    cmd = (Path(out) / "comando.txt").read_text()
    assert "--sync-guard" in cmd


def test_serie_b_min_latency_sovrascrivibile(tmp_path):
    r, nodo, _, _ = campagna(ROOT, tmp_path, {"SERIE_B": "1", "SEEDS": "1001", "MIN_LATENCY_MS": "130"})
    assert r.returncode == 0, r.stdout[-3000:]
    assert _valore(nodo, "--min-latency-ms") == "130"


def test_serie_b_frame_log_gia_in_node_extra(tmp_path):
    # a log already requested in NODE_EXTRA is not duplicated; missing delay: added
    env = {"SERIE_B": "1", "SEEDS": "1001", "NODE_EXTRA": f"--frame-log {tmp_path}/mio.csv"}
    r, nodo, _, _ = campagna(ROOT, tmp_path, env)
    assert r.returncode == 0, r.stdout[-3000:]
    assert nodo.count("--frame-log") == 1 and _valore(nodo, "--frame-log") == f"{tmp_path}/mio.csv"
    assert nodo.count("--frame-log-delay") == 1
    env["NODE_EXTRA"] += " --frame-log-delay"
    r, nodo, _, _ = campagna(ROOT, tmp_path, env)
    assert nodo.count("--frame-log") == 1 and nodo.count("--frame-log-delay") == 1


@pytest.mark.skipif(not (ORIGINALE / "scripts" / "run_campaign.sh").exists(), reason="copia originale assente")
@pytest.mark.parametrize("arm", ["nsga3", "random", "none"])
def test_senza_serie_b_processi_invariati(tmp_path, arm):
    for env in ({}, {"SERIE_B": "0"}, {"CHAMFER_BUDGET": "70", "MIN_LATENCY_MS": "100"}):
        rn, nodo_n, orch_n, out_n = campagna(ROOT, tmp_path, env, arm)
        ro, nodo_o, orch_o, out_o = campagna(ORIGINALE, tmp_path, env, arm)
        assert rn.returncode == ro.returncode == 0, rn.stdout[-2000:] + ro.stdout[-2000:]
        rel = lambda a, o: [x.replace(str(Path(o).parents[3]), "<C>") for x in a]  # noqa: E731
        assert rel(nodo_n, out_n) == rel(nodo_o, out_o)
        assert rel(orch_n, out_n) == rel(orch_o, out_o)
        assert _valore(nodo_n, "--min-latency-ms") == env.get("MIN_LATENCY_MS", "85")
        assert "--frame-log" not in nodo_n


# --- MOLA notices: crash or pose missing for more than 1 s ---

import types  # noqa: E402

import numpy as np  # noqa: E402

from src.analysis.fronte_pareto import motivi_non_valido  # noqa: E402
from src.optimization import attack_orchestrator as ao  # noqa: E402
from src.optimization import window_record as wrec  # noqa: E402

ORCH = ROOT / "src" / "optimization" / "attack_orchestrator.py"
POLY = np.array([[0.0, 0.0], [5.0, 0.0]])


def _riga_printf(i, stamp, good):
    s = f"1970/01/01,00:00:{stamp:09.6f}"
    return (f"[LidarOdometry] pathStep={i} timestamp={s} goodness={0.7 if good else 0.05:.3f} "
            f"minRequired=0.200 iters=12 termReason=IterTermReason::Stalled isGood={1 if good else 0} "
            f"adapt_thres_sigma=0.040 consecutive_bad=0")


def _log(tmp_path, nome, esiti, t0=20.0):
    """Fake twin log: one printf row per scan at 10 Hz, outcomes 1/0."""
    p = tmp_path / nome
    p.write_text("\n".join(_riga_printf(i, t0 + 0.1 * i, g) for i, g in enumerate(esiti)) + "\n")
    return p


def _eval_genome_src(path):
    import textwrap
    src = path.read_text().split("\n")
    i0 = next(i for i, r in enumerate(src) if r.strip() == "def eval_genome(g_active):")
    i1 = next(i for i in range(i0, len(src)) if src[i].strip() == "n_warm = 0")
    return textwrap.dedent("\n".join(src[i0:i1]))


def _candidati(logs, stati, ends, accorge=True):
    """eval_genome taken from the source, one candidate per log: outcome and final
    rollout pose given (pose None = discarded).
    """
    ns = dict(vars(ao))
    mola = types.SimpleNamespace(log_path=None)
    node = types.SimpleNamespace(
        last_loc_err_mean=float("nan"), last_loc_err_end=float("nan"), last_loc_err_mean_raw=float("nan"),
        last_track_ratio=0.95, last_perc=None, last_quality_mean=0.9, last_quality_min=0.8,
        last_stop_reason="h", last_tau=0.1, proc_ms_roll=[], last_replay=None, last_trace=[],
        last_twin_fp=None, last_drive_stamps=(float("nan"), float("nan")))
    n = [0]

    def rollout(genome, isaac, mola_, node_, goal, horizon, tag, **kw):
        i = n[0]
        n[0] += 1
        mola.log_path = str(logs[i]) if logs[i] is not None else None
        node.last_trace = np.array([[0.0, 0.1], [0.5, 0.2], [1.0, 0.3]])
        return (0.0, 0.0, 0.0), ends[i], 2.0, stati[i], 1.0, 10
    args = types.SimpleNamespace(
        pop=4, horizon=1.0, warmup=0.0, settle_sec=5.0, catchup_sec=0.8, attack_from="spawn",
        stop_on="estimate", untracked="discard", drift_reset_on="genome", trace=False,
        discard_policy="infra", no_nominal_rollout=False, g_nominale_senza_posa="trattieni",
        twin_filter="none", damage="mean", dir_weight=0.0, admissibility="none", chamfer_budget=None,
        max_proc_ms=None, f2_q_scale=0.0706, f2_it_scale=0.84, perceptibility="chamfer", f2_budget=None,
        verbose=False, objective="damage", mola_se_ne_accorge=accorge,
        accorge_icp_rifiutate=10, accorge_senza_posa_s=1.0)
    ns.update(
        rollout=rollout, TWIN_STATE={"mode": "clean"}, win_cap=None, t_win=0.0, eval_log=[],
        active_genes=list(range(3)), n_genes=3, expand_genome=lambda g, a, n_: np.asarray(g, float),
        isaac=None, mola=mola, node=node, wps=[np.array([1.0, 0.0])], k=2,
        nominal_ok=True, nom_ratio=1.0, nom_settle_s=None,
        frame_start=None, out=Path("/nonesiste"), nom_end=None, goal=np.array([1.0, 0.0]), nom_path=[],
        baseline_poly=POLY, nom_stats={}, nom_quality=None, nom_it=None, nom_assente=None,
        saved=(0.0, 0.0, 0.0), args=args,
        drift_fields=lambda n_: {}, _twin_flag_fields=lambda n_, m: {},
        _twin_start_fields=lambda n_: {}, _mola_csv_stats=lambda *a: {}, _gap0_from_trace=lambda t: None,
        _twin_misaligned=lambda n_, a, g: (False, None))
    exec(_eval_genome_src(ORCH), ns)
    F = [ns["eval_genome"](np.zeros(3)) for _ in logs]
    return F, ns["eval_log"]


def test_rifiutate_di_fila_solo_registrate(tmp_path):
    buone = [1] * 5
    # 9 rejected, threshold 5: past the rejection threshold, 1.0 s without pose: not excluded
    r = wrec.mola_accorge(wrec.read_printf(_log(tmp_path, "a.log", buone + [0] * 9 + buone)), n_rifiutate=5)
    assert r["rifiutate_max"] == 9 and r["rifiutate_oltre_soglia"] is True
    assert r["accorge"] is False and r["motivo"] is None
    # 10 rejected = 1.1 s without pose (pose published only with isGood=1): excluded for the pose
    r = wrec.mola_accorge(wrec.read_printf(_log(tmp_path, "b.log", buone + [0] * 10 + buone)))
    assert r["accorge"] is True and r["motivo"] == "1.1 s senza posa (soglia 1.0 s)"
    # lost scans: logged separately in senza_righe_max_s
    p = tmp_path / "e.log"
    p.write_text(_riga_printf(0, 20.0, 1) + "\n" + _riga_printf(1, 21.3, 1) + "\n")
    e = wrec.mola_accorge(wrec.read_printf(p))
    assert e["senza_righe_max_s"] == 1.3 and e["accorge"] is True


def test_crash_non_ammissibile_anche_senza_righe(tmp_path):
    r = wrec.mola_accorge([], crash=True)
    assert r["accorge"] is True and r["crash"] is True and "mola_crash" in r["motivo"]
    assert wrec.mola_accorge([])["accorge"] is None
    r = wrec.mola_accorge(wrec.read_printf(_log(tmp_path, "ok.log", [1] * 20)), crash=True)
    assert r["accorge"] is True and r["senza_posa_max_s"] == 0.1


def test_candidati_crash_e_senza_posa_non_ammissibili(tmp_path):
    buone = [1] * 20
    logs = [_log(tmp_path, "e0.log", buone),          # valid
            _log(tmp_path, "e1.log", buone[:5]),      # crash halfway
            _log(tmp_path, "e2.log", [0] * 30),       # never a pose
            None]                                     # crash, log missing
    stati = ["ok", "mola_crash", "no_pose", "mola_crash"]
    F, log = _candidati(logs, stati, [(1.0, 0.3, 0.0), None, None, None])
    assert log[0]["admissible"] is True and log[0]["accorge_ok"] is True and np.isfinite(F[0][0])
    for e in log[1:]:
        assert e["counted"] is False and e["admissible"] is False and e["accorge_ok"] is False
        assert e["inadmissible_reason"] == "mola_se_ne_accorge"
        m = motivi_non_valido(e)
        assert "mola_se_ne_accorge" in m and "non_ammissibile" not in m
    assert log[1]["mola_accorge"]["crash"] is True and log[3]["mola_accorge"]["n_icp"] == 0
    assert log[2]["mola_accorge"]["crash"] is False and log[2]["mola_accorge"]["senza_posa_max_s"] == 3.1
    assert all(f == (np.inf, np.inf, 1e6) for f in F[1:])


def test_candidati_scartati_opzione_spenta_come_prima(tmp_path):
    logs = [_log(tmp_path, "s1.log", [0] * 30), None]
    _, log = _candidati(logs, ["no_pose", "mola_crash"], [None, None], accorge=False)
    for e in log:
        assert not ({"admissible", "accorge_ok", "mola_accorge", "inadmissible_reason"} & set(e))


@pytest.mark.parametrize("arm,atteso", [("nsga3", 1), ("random", 1), ("none", 0)])
def test_serie_b_accende_mola_se_ne_accorge(tmp_path, arm, atteso):
    r, _, orch, _ = campagna(ROOT, tmp_path, {"SERIE_B": "1", "SEEDS": "1001"}, arm)
    assert r.returncode == 0, r.stdout[-3000:]
    assert orch.count("--mola-se-ne-accorge") == atteso
    r, _, orch, _ = campagna(ROOT, tmp_path, {}, arm)
    assert "--mola-se-ne-accorge" not in orch


# --- chain failure before attack detected, reason for slow candidates ---

import csv  # noqa: E402
import importlib.util  # noqa: E402
import textwrap  # noqa: E402
import time  # noqa: E402

from src.nodes.perturbation_node import DELAY_COLUMNS, FrameLog  # noqa: E402

COLONNE_FL = list(FrameLog.COLUMNS) + list(DELAY_COLUMNS)


def _fl(righe):
    """Per-frame log rows: (t_arr_wall, victim_stream, consegnata, stamp)."""
    out = []
    for t, aperto, cons, st in righe:
        r = {c: "" for c in COLONNE_FL}
        r.update(t_arr_wall=str(t), victim_stream="1" if aperto else "0", stamp_in=str(st),
                 stamp_victim=str(st) if cons else "", t_pub_victim_end_mono="5.0" if cons else "")
        out.append(r)
    return out


def test_buchi_consegna_vittima():
    regolari = [(100 + 0.1 * i, True, True, 20 + 0.1 * i) for i in range(30)]
    assert wrec.buchi_consegna_vittima(_fl(regolari), 100.0)["esito"] == "ok"
    # 15 clouds not delivered (1.6 s of sensor time) with the flow open: failure
    buco = [(100 + 0.1 * i, True, not (5 <= i < 20), 20 + 0.1 * i) for i in range(30)]
    b = wrec.buchi_consegna_vittima(_fl(buco), 100.0)
    assert b["esito"] == "guasto" and b["buco_max_s"] == 1.6 and abs(b["buco_da_stamp"] - 20.4) < 1e-9
    # Isaac pause: 30 s of wall time, continuous stamps: not a gap
    pausa = regolari[:10] + [(t + 30, a, c, s) for t, a, c, s in regolari[10:]]
    assert wrec.buchi_consegna_vittima(_fl(pausa), 100.0)["esito"] == "ok"
    # flow closed between two segments (twins): the stamp jump does not count
    tratti = (regolari[:10] + [(102.0, False, False, 25.0)]
              + [(103 + 0.1 * i, True, True, 40 + 0.1 * i) for i in range(10)])
    assert wrec.buchi_consegna_vittima(_fl(tratti), 100.0)["esito"] == "ok"
    # outside the window it does not count; without log or deliveries: undetermined
    assert wrec.buchi_consegna_vittima(_fl(buco), 102.5)["esito"] == "ok"
    assert wrec.buchi_consegna_vittima(None, 0.0)["esito"] == "non_determinato"
    assert wrec.buchi_consegna_vittima(_fl(regolari), 200.0)["esito"] == "non_determinato"


def _tracce(valori, t0=20.0):
    rows = [{"timestamp": str(t0), "icp_iterations": "0", "icp_quality": "0", "time_onLidar": "0"}]
    for i, v in enumerate(valori):
        rows.append({"timestamp": str(round(t0 + 0.1 * (i + 1), 3)), "icp_iterations": "10",
                     "icp_quality": "0.8", "time_onLidar": str(v)})
    return rows


def test_tempi_mola_vittima():
    assert wrec.CATENA_T_ON_MS == 200.0
    ok = wrec.tempi_mola_vittima(_tracce([0.02] * 50))
    assert ok["esito"] == "ok" and ok["n"] == 50 and ok["p99_ms"] == 20.0
    lento = wrec.tempi_mola_vittima(_tracce([0.02] * 40 + [0.25] * 10))
    assert lento["esito"] == "guasto" and lento["p99_ms"] == 250.0
    # only the window rows (stamp beyond the last one seen at the start)
    assert wrec.tempi_mola_vittima(_tracce([0.25] * 10 + [0.02] * 40), stamp_da=21.0)["esito"] == "ok"
    assert wrec.tempi_mola_vittima(None)["esito"] == "non_determinato"
    assert wrec.ultimo_stamp_tracce(_tracce([0.02] * 5)) == 20.5


def test_eventi_adattatore():
    ev = [{"t_check": 50, "p99": 35}, {"t_prev_clean": 10, "t_detect": 40, "t_restart": 45}]
    assert wrec.eventi_adattatore(ev, 46, 100)["esito"] == "ok"
    r = wrec.eventi_adattatore(ev, 30, 100)
    assert r["esito"] == "guasto" and len(r["riavvii"]) == 1
    # degradation started before the window and restart after: touches the window
    assert wrec.eventi_adattatore([{"t_prev_clean": 0, "t_restart": 500}], 30, 100)["esito"] == "guasto"
    r = wrec.eventi_adattatore([{"t_check": 60, "p99": 150}], 30, 100)
    assert r["esito"] == "guasto" and "p99" in r["motivo"]
    assert wrec.eventi_adattatore([{"t_check": 60, "p99": 100}], 30, 100)["esito"] == "ok"
    assert wrec.eventi_adattatore(None, 0, 1)["esito"] == "non_determinato"
    # no check in the window (monitoring off or short window): undetermined
    assert wrec.eventi_adattatore([{"t_check": 10, "p99": 30}], 30, 100)["esito"] == "non_determinato"


def _scrivi_csv(path, righe, colonne):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=colonne)
        w.writeheader()
        w.writerows(righe)
    return str(path)


def _catena(tmp_path, monkeypatch, stalli=(), pids=(4242,), status_dt=1.0, posa_dt=0.2,
            frame=None, eventi=None, tracce=None, ricevuta=False, log_path=None, opz=None):
    now = time.time()
    posa = tmp_path / "isaac_pose.json"
    posa.write_text(json.dumps({"wall": now - posa_dt}))
    monkeypatch.setattr(ao, "POSE_FILE", str(posa))
    monkeypatch.setattr(ao, "_perturbation_vivo", lambda: list(pids))
    monkeypatch.setitem(ao.STALL_POLICY, "events", [dict(window=2, **e) for e in stalli])
    t0 = now - 60
    fr = frame if frame is not None else [(t0 + 1 + 0.1 * i, True, True, 20 + 0.1 * i) for i in range(50)]
    fl = _scrivi_csv(tmp_path / "frame_log.csv", _fl(fr), COLONNE_FL)
    ev = tmp_path / "eventi.jsonl"
    ev.write_text("".join(json.dumps(e) + "\n" for e in (eventi or [{"t_check": t0 + 5, "p99": 30}])))
    tr = _scrivi_csv(tmp_path / "traces_victim.csv", tracce or _tracce([0.02] * 50),
                     ["timestamp", "icp_iterations", "icp_quality", "time_onLidar"])
    args = types.SimpleNamespace(adapter_events=str(ev), frame_log_nodo=fl, **(opz or {}))
    # last received pose: the last trace row (no pose lost), unless tested
    ultima = wrec.ultimo_stamp_tracce(wrec.leggi_csv(tr))
    node = types.SimpleNamespace(status_t=now - status_dt,
                                 victim_est_stamp=ultima if ricevuta is False else ricevuta)
    victim = types.SimpleNamespace(traces_path=tr, log_path=log_path)
    return ao.controlla_catena(args, node, victim, 2, {"wall": t0, "stamp_tracce": 20.0})


def test_controlla_catena(tmp_path, monkeypatch):
    c = _catena(tmp_path, monkeypatch)
    assert c["guasto"] is False and c["non_determinati"] == []
    assert set(c["controlli"]) == {"isaac", "adattatore", "nuvole_vittima", "posa_vittima",
                                   "mola_vittima", "perturbation"}
    casi = {
        "isaac": dict(stalli=[{"phase": "x", "error": "e"}]),
        "perturbation": dict(pids=()),
        "adattatore": dict(eventi=[{"t_prev_clean": time.time() - 100, "t_restart": time.time() - 30}]),
        "mola_vittima": dict(tracce=_tracce([0.3] * 20)),
    }
    for nome, kw in casi.items():
        c = _catena(tmp_path, monkeypatch, **kw)
        assert c["guasto"] is True and c["controlli"][nome]["esito"] == "guasto", nome
        assert c["motivi"][0].startswith(nome + ":")
    # Isaac loop pose stuck beyond STALL_SEC; node without state in the window
    assert _catena(tmp_path, monkeypatch, posa_dt=8.0)["controlli"]["isaac"]["esito"] == "guasto"
    assert _catena(tmp_path, monkeypatch, status_dt=120.0)["controlli"]["perturbation"]["esito"] == "guasto"
    t0 = time.time() - 60
    buco = [(t0 + 1 + 0.1 * i, True, not (10 <= i < 25), 20 + 0.1 * i) for i in range(50)]
    assert _catena(tmp_path, monkeypatch, frame=buco)["controlli"]["nuvole_vittima"]["esito"] == "guasto"


def test_perturbation_vivo_dal_proc():
    assert isinstance(ao._perturbation_vivo(), list)
    import re
    rx = re.compile(ao._RX_PERTURBATION)
    assert rx.search("python3 -u src/nodes/perturbation_node.py --min-latency-ms 140")
    assert not rx.search("bash -c tail perturbation_node.py")


def _chiusura(tmp_path, catena, danno=False):
    spec = importlib.util.spec_from_file_location("_t_scop", ROOT / "tests" / "test_attacco_scoperto.py")
    ts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ts)
    if not danno:
        return ts._chiudi(tmp_path, True, ["ok", "ok", "pose_stale", "ok"], catena=catena, controllo=True)
    src = ts._chiusura_src(ts.ORCH).replace(
        "_esito_finestra({'window': k}, statuses[k])",
        "_esito_finestra({'window': k, **({'danno_escluso': 'attacco_scoperto'} "
        "if statuses[k] == 'pose_stale' else {})}, statuses[k])")
    assert "danno_escluso" in src
    ns = dict(vars(ao))
    ns["controlla_catena"] = lambda *a: catena
    ns["out"] = tmp_path
    exec(src, ns)
    args = types.SimpleNamespace(chiudi_attacco_scoperto=True, victim_start="spawn",
                                 settle_clouds=0, victim_never_clean=False, seed=1001,
                                 controllo_catena=True)
    victim = types.SimpleNamespace(origin_info={}, log_path=None, last_est_end=None)
    node = types.SimpleNamespace(true=None, true_buf=[(30.1, 0.0, 0.0)])
    return ns["_prova"](args, victim, node, [], {"since": None, "any": False}, False,
                        ["ok", "ok", "pose_stale", "ok"])


def test_guasto_catena_chiude_la_run_al_posto_dell_attacco_scoperto(tmp_path):
    rotta = wrec.guasto_catena({"isaac": {"esito": "guasto", "motivo": "1 stalli di Isaac nella finestra"},
                                "adattatore": {"esito": "ok"}, "nuvole_vittima": {"esito": "non_determinato"}})
    esito, k, h = _chiusura(tmp_path, rotta)
    assert esito == "chiusa" and k == 2
    fine = h[2]["run_end"]
    assert fine["reason"] == "guasto_catena" and fine["window"] == 2 and fine["status"] == "pose_stale"
    assert fine["motivi"] == ["isaac: 1 stalli di Isaac nella finestra"]
    assert fine["da_rifare"] is True and fine["seme"] == 1001 and fine["non_determinati"] == ["nuvole_vittima"]
    assert (fine["windows_driven"], fine["windows_held"]) == (3, 0) and fine["sim_time_s"] == 30.1
    # applied window outside the damage: the reason becomes guasto_catena
    _, _, h = _chiusura(tmp_path, rotta, danno=True)
    assert h[2]["danno_escluso"] == "guasto_catena"
    # healthy chain: attack detected, with the check recorded
    sana = wrec.guasto_catena({"isaac": {"esito": "ok"}})
    _, _, h = _chiusura(tmp_path, sana)
    assert h[2]["run_end"]["reason"] == "attacco_scoperto"
    assert h[2]["run_end"]["controllo_catena"]["guasto"] is False
    _, _, h = _chiusura(tmp_path, sana, danno=True)
    assert h[2]["danno_escluso"] == "attacco_scoperto"
    # exit code 2: run_campaign.sh does not mark it complete and redoes it with the same seed
    assert "if foreign or sync_fail or guasto_catena or errore_conf:\n        return 2" in ORCH.read_text()


def _lenti_src():
    src = ORCH.read_text().split("\n")
    a = next(i for i, r in enumerate(src) if "# top_maxdamage (only with maxdamage" in r)
    b = next(i for i in range(a, len(src)) if "n_valid_sel = sum(" in src[i])
    return textwrap.dedent("\n".join(src[a:b + 1]))


@pytest.mark.parametrize("select,search,adm,atteso", [
    ("maxdamage", "nsga3", "stealth", True), ("maxdamage", "fixed", "stealth", False),
    ("maxdamage", "nsga3", "none", False), ("pareto_maxdamage", "nsga3", "stealth", False)])
def test_motivo_dei_lenti(select, search, adm, atteso):
    base = {"status": "ok", "counted": True, "admissible": True, "damage": 0.1, "chamfer": 1.0}
    eval_log = [dict(base, slow=True, genome=[0.1]), dict(base, slow=False, genome=[0.2])]
    ns = {"eval_log": eval_log, "motivi_non_valido": motivi_non_valido,
          "args": types.SimpleNamespace(select=select, search=search, admissibility=adm,
                                        chamfer_budget=None, no_nominal_rollout=False)}
    exec(_lenti_src(), ns)
    assert eval_log[0]["selection_invalid_reasons"] == (["lento"] if atteso else [])
    assert eval_log[1]["selection_invalid_reasons"] == []
    assert ns["n_valid_sel"] == (1 if atteso else 2)
    # the top_maxdamage filter excludes the same candidates
    if select == "maxdamage":
        args = types.SimpleNamespace(search=search, admissibility=adm, select_only_valid=True,
                                     chamfer_budget=None, topk=3)
        top = ao.top_maxdamage([dict(e) for e in eval_log], args)
        assert len(top) == (1 if atteso else 2)


def test_campagna_serie_b_controllo_catena(tmp_path):
    r, nodo, orch, out = campagna(ROOT, tmp_path, {"SERIE_B": "1", "SEEDS": "1001"})
    assert r.returncode == 0, r.stdout[-3000:]
    assert _valore(orch, "--frame-log-nodo") == _valore(nodo, "--frame-log") == f"{out}/frame_log.csv"
    assert _valore(orch, "--adapter-events") == "data/attack/verifica/adattatore_eventi.jsonl"
    assert "--frame-log-nodo" in (Path(out) / "comando.txt").read_text()
    # log and event file already given: those are used, no duplicates
    env = {"SERIE_B": "1", "SEEDS": "1001", "NODE_EXTRA": "--frame-log /x/fl.csv --frame-log-delay",
           "ORCH_EXTRA": "--adapter-events /x/ev.jsonl"}
    r, nodo, orch, out = campagna(ROOT, tmp_path, env)
    assert _valore(orch, "--frame-log-nodo") == "/x/fl.csv" and orch.count("--adapter-events") == 1
    assert _valore(orch, "--adapter-events") == "/x/ev.jsonl"
    # arm none: no chain check (no attack detected)
    r, nodo, orch, out = campagna(ROOT, tmp_path, {"SERIE_B": "1", "SEEDS": "1001"}, "none")
    assert "--frame-log-nodo" not in orch and "--adapter-events" not in orch
