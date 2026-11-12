"""Three options, off by default:
(1) with --no-nominal-rollout the selection reason comes from esito_gemello
    (gemello_perde_posa instead of guasto_gemello);
(2) --chiudi-attacco-scoperto: victim without pose = run closed with run_end
    attacco_scoperto, window with pose_stale excluded from the damage;
(3) --mola-se-ne-accorge: candidate infeasible with more than T s without pose in the twin
    log (consecutive rejected ICPs are only logged, see test_regole_notte.py).
Fake objects, no ROS, MOLA or Isaac; segments of main are taken from the source and run
as they are."""

import json
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.fronte_pareto import motivi_non_valido  # noqa: E402
from src.optimization import attack_orchestrator as ao  # noqa: E402
from src.optimization import window_record as wrec  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ORCH = ROOT / "src" / "optimization" / "attack_orchestrator.py"
# reference: the copy before these options or a later copy that already has them; the
# comparisons remove the options from both sides and count them only in this copy
ORIGINALE = ROOT.parent / "originale"
ORCH_ORIG = ORIGINALE / "src" / "optimization" / "attack_orchestrator.py"
NO_POSE = "no_pose_from_new_mola (publisher nel grafo: 0, processi vivi: 4)"
POLY = np.array([[0.0, 0.0], [5.0, 0.0]])
NUOVE = ("chiudi_attacco_scoperto", "mola_se_ne_accorge", "accorge_icp_rifiutate", "accorge_senza_posa_s",
         "frame_log_nodo", "controllo_catena", "rallentamento_mola")
# options added later, off by default
SUCCESSIVE = {"incompleti_regola": False,
              # limit-study options, off by default
              "moltiplicatore_limiti": 1.0, "famiglie_limiti": None, "blocca_gene": None}


def _riga_printf(i, stamp, good):
    """ICP printf line as in the MOLA log with MOLA_LO_DEBUG_ICP_QUALITY=1."""
    s = f"1970/01/01,00:00:{stamp:09.6f}"
    return (f"[LidarOdometry] pathStep={i} timestamp={s} goodness={0.7 if good else 0.05:.3f} "
            f"minRequired=0.200 iters=12 termReason=IterTermReason::Stalled isGood={1 if good else 0} "
            f"adapt_thres_sigma=0.040 consecutive_bad=0")


def _log(tmp_path, nome, esiti, t0=20.0):
    """Fake log: one line per scan at 10 Hz, outcomes = sequence of 1/0."""
    p = tmp_path / nome
    p.write_text("\n".join(_riga_printf(i, t0 + 0.1 * i, g) for i, g in enumerate(esiti)) + "\n")
    return p


# --- arguments and run_campaign.sh ------------------------------------------------------

def _default_args(modulo_dir):
    codice = textwrap.dedent("""
        import json, sys
        from src.optimization import attack_orchestrator as ao
        class Stop(Exception):
            pass
        orig = ao.argparse.ArgumentParser.parse_args
        preso = {}
        def parse(self, *a, **k):
            preso["a"] = orig(self, *a, **k)
            raise Stop
        ao.argparse.ArgumentParser.parse_args = parse
        sys.argv = ["attack_orchestrator.py"]
        try:
            ao.main()
        except Stop:
            pass
        print(json.dumps(vars(preso["a"]), default=str, sort_keys=True))
    """)
    r = subprocess.run([sys.executable, "-c", codice], cwd=modulo_dir, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(not ORCH_ORIG.exists(), reason="copia originale assente")
def test_argomenti_di_default_invariati():
    nuovo = _default_args(ROOT)
    vecchio = _default_args(ORIGINALE)
    attese = {
        "chiudi_attacco_scoperto": False, "mola_se_ne_accorge": False,
        "accorge_icp_rifiutate": 10, "accorge_senza_posa_s": 1.0, "frame_log_nodo": None,
        "controllo_catena": False, "rallentamento_mola": "guasto"}
    assert {k: nuovo.pop(k) for k in NUOVE} == attese
    # reference that already has the options: all present, same defaults
    gia = {k: vecchio.pop(k) for k in NUOVE if k in vecchio}
    assert gia in ({}, attese)
    for k, v in SUCCESSIVE.items():
        if k not in vecchio:
            assert nuovo.pop(k, v) == v
    assert nuovo == vecchio


TOLTE = ("SERIE_B", "WINDOW_RNG", "SEEDS", "ICP_QUALITY_LOG", "PERC_TERMS", "SAMPLING",
         "RANDOM_FIRST", "TWIN_FILTER", "MOLA_LO_DEBUG_ICP_QUALITY", "CHAMFER_BUDGET",
         "MAX_DRIFT_LAG_S", "RUN_SUFFIX", "MIN_LATENCY_MS", "FAST_OPS", "REPORT_MATCHED",
         "SYNC_WAIT", "SYNC_GUARD", "NODE_EXTRA", "ORCH_EXTRA", "SELECT", "TOPK")


def _flags_campagna(radice, tmp_path, env_extra, arm):
    """Orchestrator options built by run_campaign.sh of a copy (the part before the
    process checks), without starting anything."""
    import os
    src = (radice / "scripts" / "run_campaign.sh").read_text().splitlines()
    fine = next(i for i, l in enumerate(src) if l.startswith("if pgrep -f"))
    corpo = "\n".join(src[:fine]).replace('cd "$ROOT"', ": cd") + '\necho "FLAGS=$EXTRA $SERIEB_FLAGS"\n'
    f = tmp_path / f"blocco_{radice.name}.sh"
    f.write_text(corpo)
    env = {k: v for k, v in os.environ.items() if k not in TOLTE}
    env.update(env_extra)
    r = subprocess.run(["bash", str(f), "straight", arm, "3", "1", "4", "2"],
                       capture_output=True, text=True, env=env, cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    return next(l.split("=", 1)[1] for l in r.stdout.splitlines() if l.startswith("FLAGS="))


@pytest.mark.skipif(not (ORIGINALE / "scripts" / "run_campaign.sh").exists(), reason="copia originale assente")
@pytest.mark.parametrize("arm", ["nsga3", "random", "gaussian", "none"])
def test_campagna_senza_serie_b_invariata(tmp_path, arm):
    for env in ({}, {"CHAMFER_BUDGET": "70"}):
        assert _flags_campagna(ROOT, tmp_path, env, arm) == _flags_campagna(ORIGINALE, tmp_path, env, arm)


@pytest.mark.skipif(not (ORIGINALE / "scripts" / "run_campaign.sh").exists(), reason="copia originale assente")
@pytest.mark.parametrize("arm,atteso", [("nsga3", True), ("random", True), ("none", False)])
def test_campagna_serie_b(tmp_path, arm, atteso):
    env = {"SERIE_B": "1", "SEEDS": "1001"}
    nuovo = _flags_campagna(ROOT, tmp_path, env, arm).split()
    vecchio = _flags_campagna(ORIGINALE, tmp_path, env, arm).split()
    # extra options outside the none arm (also "MOLA notices" and the adapter events for
    # the chain check, and --incompleti-regola in every attack arm); removed from both
    # sides because the reference may already contain them
    ev = "data/attack/verifica/adattatore_eventi.jsonl"
    tolte = ("--chiudi-attacco-scoperto", "--mola-se-ne-accorge", "--adapter-events", ev,
             "--controllo-catena", "--incompleti-regola")
    assert [x for x in nuovo if x not in tolte] == [x for x in vecchio if x not in tolte]
    assert nuovo.count("--incompleti-regola") == (0 if arm == "none" else 1)
    assert nuovo.count("--adapter-events") == (1 if atteso else 0)
    assert nuovo.count("--controllo-catena") == (1 if atteso else 0)
    assert nuovo.count("--chiudi-attacco-scoperto") == (1 if atteso else 0)
    assert nuovo.count("--mola-se-ne-accorge") == (1 if atteso else 0)


def test_valori_non_validi_rifiutati():
    # main stops right after the argument checks (next line: parse_env_pairs)
    codice = textwrap.dedent("""
        import sys
        from src.optimization import attack_orchestrator as ao
        def ferma(*a, **k):
            raise SystemExit(99)
        ao.parse_env_pairs = ferma
        sys.argv = ["attack_orchestrator.py"] + sys.argv[1:]
        sys.exit(ao.main())
    """)
    for extra, rc in ((["--accorge-icp-rifiutate", "0"], 2), (["--accorge-senza-posa-s", "0"], 2),
                      (["--accorge-icp-rifiutate", "0"], 99)):
        opz = ["--mola-se-ne-accorge"] if rc == 2 else []
        r = subprocess.run([sys.executable, "-c", codice, *opz, *extra],
                           cwd=ROOT, capture_output=True, text=True, timeout=60)
        assert r.returncode == rc, r.stdout[-500:] + r.stderr[-500:]
        assert ("--accorge-icp-rifiutate deve essere >= 1" in r.stdout) == (rc == 2)


# --- (1) reason label --------------------------------------------------------------------

def _scartato(status, **extra):
    r = {"genome": [0.5, 0.5], "dev": None, "chamfer": 3.0, "status": status,
         "counted": False, "dos": False}
    r.update(extra)
    return r


def test_etichetta_gemello_perde_posa():
    perde = _scartato(NO_POSE, esito_gemello="gemello_perde_posa")
    guasto = _scartato(NO_POSE, esito_gemello="guasto_gemello", guasto_gemello=True)
    # without --no-nominal-rollout: unchanged
    assert motivi_non_valido(perde) == ["guasto_gemello", "scartato"]
    assert motivi_non_valido(guasto) == ["guasto_gemello", "scartato"]
    # with --no-nominal-rollout: reason from esito_gemello, still not valid
    assert motivi_non_valido(perde, da_esito_gemello=True) == ["gemello_perde_posa", "scartato"]
    assert motivi_non_valido(guasto, da_esito_gemello=True) == ["guasto_gemello", "scartato"]
    # pose_stale with good checks: esito_gemello controlli_ok, reason unchanged
    stale = _scartato("pose_stale", esito_gemello="controlli_ok")
    assert motivi_non_valido(stale, da_esito_gemello=True) == ["guasto_gemello", "scartato"]


def test_etichetta_nel_ciclo_di_selezione():
    src = ORCH.read_text()
    assert "motivi_non_valido(\n                        e, args.chamfer_budget, da_esito_gemello=args.no_nominal_rollout)" in src


# --- (3) MOLA notices --------------------------------------------------------------------

def test_mola_accorge_soglie(tmp_path):
    buone = [1] * 5
    r9 = wrec.mola_accorge(wrec.read_printf(_log(tmp_path, "a.log", buone + [0] * 9 + buone)))
    r10 = wrec.mola_accorge(wrec.read_printf(_log(tmp_path, "b.log", buone + [0] * 10 + buone)))
    assert (r9["rifiutate_max"], r9["senza_posa_max_s"], r9["accorge"]) == (9, 1.0, False)
    assert (r10["rifiutate_max"], r10["senza_posa_max_s"], r10["accorge"]) == (10, 1.1, True)
    # the same count applies at the head (before the first pose) and at the tail
    t9 = wrec.mola_accorge(wrec.read_printf(_log(tmp_path, "c.log", [0] * 9 + buone)))
    t10 = wrec.mola_accorge(wrec.read_printf(_log(tmp_path, "d.log", buone + [0] * 10)))
    assert t9["accorge"] is False and t10["accorge"] is True
    # lost scans: 12 slots without a line between two good ICPs = 1.3 s without pose
    p = tmp_path / "e.log"
    p.write_text(_riga_printf(0, 20.0, 1) + "\n" + _riga_printf(1, 21.3, 1) + "\n")
    e = wrec.mola_accorge(wrec.read_printf(p))
    assert e["rifiutate_max"] == 0 and e["senza_posa_max_s"] == 1.3 and e["accorge"] is True
    # no printf lines: undetermined
    assert wrec.mola_accorge([])["accorge"] is None
    # different thresholds: consecutive rejections over threshold do not exclude
    f = wrec.mola_accorge(wrec.read_printf(_log(tmp_path, "f.log", buone + [0] * 9 + buone)),
                          n_rifiutate=5, senza_posa_s=5.0)
    assert f["accorge"] is False and f["rifiutate_oltre_soglia"] is True


def _eval_genome_src(path):
    src = path.read_text().split("\n")
    i0 = next(i for i, r in enumerate(src) if r.strip() == "def eval_genome(g_active):")
    i1 = next(i for i in range(i0, len(src)) if src[i].strip() == "n_warm = 0")
    return textwrap.dedent("\n".join(src[i0:i1]))


def _esegui_candidati(path, logs, accorge, budget=None, chamfer=2.0):
    """One 'ok' candidate per twin log; returns objectives and eval_log."""
    ns = dict(vars(ao))
    mola = types.SimpleNamespace(log_path=None)
    node = types.SimpleNamespace(
        last_loc_err_mean=float("nan"), last_loc_err_end=float("nan"), last_loc_err_mean_raw=float("nan"),
        last_track_ratio=0.95, last_perc=None, last_quality_mean=0.9, last_quality_min=0.8,
        last_stop_reason="h", last_tau=0.1, proc_ms_roll=[], last_replay=None, last_trace=[],
        last_twin_fp=None, last_drive_stamps=(float("nan"), float("nan")))
    n = [0]

    def rollout(genome, isaac, mola_, node_, goal, horizon, tag, **kw):
        mola.log_path = str(logs[n[0]])
        n[0] += 1
        node.last_trace = np.array([[0.0, 0.1], [0.5, 0.2], [1.0, 0.3]])
        return (0.0, 0.0, 0.0), (1.0, 0.3, 0.0), chamfer, "ok", 1.0, 10
    args = types.SimpleNamespace(
        pop=4, horizon=1.0, warmup=0.0, settle_sec=5.0, catchup_sec=0.8, attack_from="spawn",
        stop_on="estimate", untracked="discard", drift_reset_on="genome", trace=False,
        discard_policy="infra", no_nominal_rollout=True, g_nominale_senza_posa="trattieni",
        twin_filter="none", damage="mean", dir_weight=0.0, admissibility="none", chamfer_budget=budget,
        max_proc_ms=None, f2_q_scale=0.0706, f2_it_scale=0.84, perceptibility="chamfer", f2_budget=None,
        verbose=False, objective="damage", mola_se_ne_accorge=accorge,
        accorge_icp_rifiutate=10, accorge_senza_posa_s=1.0)
    ns.update(
        rollout=rollout, TWIN_STATE={"mode": "clean"}, win_cap=None, t_win=0.0, eval_log=[],
        active_genes=list(range(3)), n_genes=3, expand_genome=lambda g, a, n_: np.asarray(g, float),
        isaac=None, mola=mola, node=node, wps=[np.array([1.0, 0.0])], k=2,
        nominal_ok=True, nom_ratio=1.0, nom_settle_s=None,
        frame_start=None, out=Path("/nonesiste"), nom_end=None, goal=np.array([1.0, 0.0]), nom_path=[],
        baseline_poly=POLY, nom_stats={}, nom_quality=None, nom_it=None,
        nom_assente={"window": 2, "senza_nominale": True},
        saved=(0.0, 0.0, 0.0), args=args,
        drift_fields=lambda n_: {}, _twin_flag_fields=lambda n_, m: {},
        _twin_start_fields=lambda n_: {}, _mola_csv_stats=lambda *a: {}, _gap0_from_trace=lambda t: None,
        _twin_misaligned=lambda n_, a, g: (False, None))
    exec(_eval_genome_src(path), ns)
    F = [ns["eval_genome"](np.zeros(3)) for _ in logs]
    return F, ns["eval_log"]


def test_candidato_10_rifiutate_non_ammissibile_9_si(tmp_path):
    buone = [1] * 20
    logs = [_log(tmp_path, "e9.log", buone + [0] * 9 + buone),
            _log(tmp_path, "e10.log", buone + [0] * 10 + buone)]
    F, log = _esegui_candidati(ORCH, logs, accorge=True)
    e9, e10 = log
    assert e9["admissible"] is True and e9["accorge_ok"] is True
    assert e9["mola_accorge"]["rifiutate_max"] == 9
    assert motivi_non_valido(e9, da_esito_gemello=True) == []
    assert e10["admissible"] is False and e10["accorge_ok"] is False
    assert e10["inadmissible_reason"] == "mola_se_ne_accorge" and e10["dos"] is False
    assert motivi_non_valido(e10, da_esito_gemello=True) == ["mola_se_ne_accorge"]
    # infeasible for NSGA-III like a non-stealthy candidate; the measurement still counts
    assert F[1][0] == np.inf and F[1][2] == 1e6 and e10["counted"] is True
    assert np.isfinite(F[0][0])
    # with the budget violated the two reasons stay distinct
    _, log_b = _esegui_candidati(ORCH, logs[1:], accorge=True, budget=70.0, chamfer=80.0)
    assert motivi_non_valido(log_b[0], 70.0) == ["mola_se_ne_accorge", "chamfer_oltre_budget"]


@pytest.mark.skipif(not ORCH_ORIG.exists(), reason="copia originale assente")
def test_candidati_opzione_spenta_come_prima(tmp_path):
    buone = [1] * 20
    logs = [_log(tmp_path, "x9.log", buone + [0] * 9 + buone),
            _log(tmp_path, "x10.log", buone + [0] * 10 + buone)]
    Fn, ln = _esegui_candidati(ORCH, logs, accorge=False)
    Fo, lo = _esegui_candidati(ORCH_ORIG, logs, accorge=False)
    for r in ln + lo:
        r.pop("t_wall")
    # proc_ms_* are nan: compare the JSON text
    assert Fn == Fo and json.dumps(ln, sort_keys=True, default=str) == json.dumps(lo, sort_keys=True, default=str)
    assert all(r["admissible"] is True and "accorge_ok" not in r for r in ln)


# --- (2) attack discovered ---------------------------------------------------------------

def test_vittima_persa_e_record(tmp_path):
    assert all(wrec.vittima_persa(s) for s in ("victim_no_pose", "victim_crash", "pose_stale"))
    assert not any(wrec.vittima_persa(s) for s in (None, "ok", "arrived", NO_POSE, "untracked(0.5)"))
    righe = wrec.read_printf(_log(tmp_path, "v.log", [1] * 5 + [0] * 4))
    r = wrec.attacco_scoperto_record(3, "pose_stale", righe, (1.0, 2.0, 0.1), (1.1, 2.0, 0.1),
                                     35.0, 1_800_000_000.0, driven=4, held=0)
    assert r["reason"] == "attacco_scoperto" and r["window"] == 3 and r["status"] == "pose_stale"
    assert r["stamp_ultima_posa"] == 20.4 and r["stamp_prima_icp_rifiutata"] == 20.5
    assert r["stamp_ultima_riga_log"] == 20.8 and r["sim_time_s"] == 35.0
    assert r["ultima_posa"] == [1.0, 2.0, 0.1] and r["posa_vera"] == [1.1, 2.0, 0.1]
    assert r["wall_iso"].startswith("2027-01-15") and r["separato_dai_successi"] is True
    assert (r["windows_driven"], r["windows_held"]) == (4, 0)
    vuoto = wrec.attacco_scoperto_record(0, "victim_no_pose")
    assert vuoto["stamp_ultima_posa"] is None and vuoto["ultima_posa"] is None


def _chiusura_src(path):
    """_esito_finestra and the check at the top of the window loop, taken from the source,
    in a fake loop that records one window per outcome."""
    src = path.read_text().split("\n")
    a0 = next(i for i, r in enumerate(src) if r.strip() == "def _esito_finestra(rec, v_status):")
    a1 = next(i for i in range(a0, len(src)) if src[i].strip() == "def _goal_mode_rec():")
    b0 = next(i for i, r in enumerate(src) if r.strip() == "while True:" and i > a1)
    b1 = next(i for i in range(b0, len(src)) if src[i].strip() == "k += 1")
    esito = textwrap.indent(textwrap.dedent("\n".join(src[a0:a1])), " " * 4)
    cima = textwrap.indent(textwrap.dedent("\n".join(src[b0 + 1:b1])), " " * 8)
    return ("def _prova(args, victim, node, history, lost_state, direction, statuses):\n"
            "    scoperto = None\n    victim_lost = False\n    nom_assente = None\n"
            "    shadow_rec = None\n    k = -1\n    catena_t0 = None\n"
            "    saved = (0.0, 0.0, 0.0)\n    goals_abs = [(5.0, 0.0)]\n"
            + esito + "\n"
            "    while True:\n" + cima + "\n"
            "        k += 1\n"
            "        if k >= len(statuses):\n"
            "            return 'finite', k, history\n"
            "        history.append(_esito_finestra({'window': k}, statuses[k]))\n"
            "        if statuses[k] is not None and str(statuses[k]).startswith('victim_'):\n"
            "            victim_lost = True\n"
            "    return 'chiusa', k, history\n")


def _chiudi(tmp_path, opzione, statuses, path=ORCH, catena=None, controllo=False):
    ns = dict(vars(ao))
    # chain check (rule 5, --controllo-catena): healthy, unless the test sets an outcome
    ns["controlla_catena"] = lambda *a: catena if catena is not None else wrec.guasto_catena({})
    ns["out"] = tmp_path
    exec(_chiusura_src(path), ns)
    args = types.SimpleNamespace(chiudi_attacco_scoperto=opzione, victim_start="spawn",
                                 settle_clouds=0, victim_never_clean=False, seed=1001,
                                 controllo_catena=controllo)
    victim = types.SimpleNamespace(origin_info={"o": 1}, log_path=str(_log(tmp_path, "vit.log", [1] * 6 + [0] * 3)),
                                   last_est_end=(1.0, 0.5, 0.0))
    node = types.SimpleNamespace(true=(1.2, 0.4, 0.0), true_buf=[(30.0, 1.2, 0.4), (30.1, 1.2, 0.4)])
    return ns["_prova"](args, victim, node, [], {"since": None, "any": False}, False, statuses)


def test_chiusura_con_record(tmp_path):
    esito, k, h = _chiudi(tmp_path, True, ["ok", "ok", "pose_stale", "ok", "ok"])
    assert esito == "chiusa" and k == 2 and len(h) == 3
    fine = h[2]["run_end"]
    assert fine["reason"] == "attacco_scoperto" and fine["window"] == 2 and fine["status"] == "pose_stale"
    assert fine["stamp_ultima_posa"] == 20.5 and fine["stamp_prima_icp_rifiutata"] == 20.6
    assert fine["sim_time_s"] == 30.1 and fine["ultima_posa"] == [1.0, 0.5, 0.0]
    assert (fine["windows_driven"], fine["windows_held"]) == (3, 0)
    assert all("run_end" not in r for r in h[:2])
    # existing victim_lost branch: with the option it closes with the record
    esito, k, h = _chiudi(tmp_path, True, ["ok", "victim_no_pose", "ok"])
    assert esito == "chiusa" and h[1]["run_end"]["status"] == "victim_no_pose" and h[1]["victim_lost"] is True


def test_chiusura_spenta_come_prima(tmp_path):
    # off: pose_stale does not stop the run, victim_no_pose stops it without run_end
    esito, k, h = _chiudi(tmp_path, False, ["ok", "pose_stale", "ok"])
    assert esito == "finite" and len(h) == 3 and all("run_end" not in r for r in h)
    esito, k, h = _chiudi(tmp_path, False, ["ok", "victim_no_pose", "ok"])
    assert esito == "chiusa" and len(h) == 2 and all("run_end" not in r for r in h)


def _applicato_src(path):
    src = path.read_text().split("\n")
    a0 = next(i for i, r in enumerate(src) if r.strip().startswith("danno_escluso = bool("))
    while not src[a0 - 1].strip().startswith("_refresh_victim_scans(history, victim)"):
        a0 -= 1
    a1 = next(i for i in range(a0, len(src)) if src[i].strip().startswith('print(f"  tempo finestra'))
    corpo = textwrap.indent(textwrap.dedent("\n".join(src[a0:a1])), " " * 4)
    return ("def _app(args, victim, real_end, st, node):\n    victim_lost = False\n"
            "    nom_end = goal = nom_path = baseline_poly = None\n    ref_lbl = 'x'\n"
            "    best_F = [-0.1, 2.0]\n    trav, frames = 1.0, 10\n"
            "    applied_quality = 0.9\n    applied_loc_mean = applied_loc_end = applied_loc_mean_raw = 1.0\n"
            "    applied_perc = None\n" + corpo + "\n    return locals()\n")


def test_finestra_pose_stale_fuori_dal_danno():
    ns = dict(vars(ao))
    ns["window_damage"] = lambda *a, **kw: {"dev_end": 0.2, "dev_mean": 0.1, "dev_max": 0.3, "damage": 0.15}
    ns["true_path"] = lambda t: t
    exec(_applicato_src(ORCH), ns)
    node = types.SimpleNamespace(last_trace=[])
    acc = types.SimpleNamespace(chiudi_attacco_scoperto=True, damage="mean", dir_weight=0.0)
    spe = types.SimpleNamespace(chiudi_attacco_scoperto=False, damage="mean", dir_weight=0.0)
    v = object()
    end = (1.0, 0.3, 0.0)
    loc = ns["_app"](acc, v, end, "pose_stale", node)
    assert loc["danno_escluso"] is True and loc["dm_real"] is None and loc["drift"] is None
    # off, or good outcome, or no victim: damage computed as usual
    for a, vv, st in ((spe, v, "pose_stale"), (acc, v, "ok"), (acc, None, "pose_stale")):
        loc = ns["_app"](a, vv, end, st, node)
        assert loc["danno_escluso"] is False and loc["dm_real"]["damage"] == 0.15
    # no final pose: usual branch (application failed, victim_lost)
    loc = ns["_app"](acc, v, None, "victim_no_pose", node)
    assert loc["danno_escluso"] is False and loc["dm_real"] is None and loc["victim_lost"] is True
    # the window record carries the exclusion reason
    assert '**({"danno_escluso": "attacco_scoperto"} if danno_escluso else {}),' in ORCH.read_text()


def test_vittima_persa_comprende_no_pose_e_mola_crash():
    # a victim with no estimate in the segment (no_pose) or that loses MOLA (mola_crash) is lost
    from src.optimization import window_record as wr
    for st in ("no_pose", "mola_crash", "no_pose (publisher nel grafo: 0)", "pose_stale"):
        assert wr.vittima_persa(st)
    for st in ("ok", "arrived", "untracked", "est_capped", None):
        assert not wr.vittima_persa(st)
