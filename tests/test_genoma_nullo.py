"""Null genome when a window has no admissible genome (--genoma-nullo-se-nessun-ammissibile)
and the queue conditions without budget (casuale_senzabudget, randpuro_senzabudget).

Option on: no hold, the null genome (directions 0, other genes -1) takes the winner path
(front, repeated winner checks, comparisons, window record) and the window counts as
driven; history with genoma_nullo, reason, attempts, victim |A| from the node status and
measured P. Option off: hold as before. Segments of the window loop are taken from the
source and run with fake objects, as in test_orchestratore_D31.py; no ROS node, Isaac or
MOLA."""

import json
import os
import re
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.optimization import attack_orchestrator as ao  # noqa: E402
from src.optimization import window_record as wrec  # noqa: E402
from src.perturbations.perturbation_generator import PerturbationGenerator  # noqa: E402

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("coda", ROOT / "scripts" / "coda_serieB_stato.py")
coda = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(coda)

ORCH = ROOT / "src" / "optimization" / "attack_orchestrator.py"
CODA_SH = ROOT / "scripts" / "coda_serieB.sh"
RUN_CAMPAIGN = ROOT / "scripts" / "run_campaign.sh"
NULLO = [0.0 if g in ao.DIRECTION_GENES else -1.0 for g in range(17)]
OPZ = "--genoma-nullo-se-nessun-ammissibile"


def _args_main(argv):
    """main arguments for a command line, without starting anything."""
    preso = {}

    class Stop(Exception):
        pass
    orig = ao.argparse.ArgumentParser.parse_args

    def parse(self, *a, **k):
        preso["a"] = orig(self, *a, **k)
        raise Stop
    ao.argparse.ArgumentParser.parse_args = parse
    vecchio = sys.argv
    sys.argv = ["attack_orchestrator.py", *argv]
    try:
        ao.main()
    except Stop:
        pass
    finally:
        ao.argparse.ArgumentParser.parse_args = orig
        sys.argv = vecchio
    return preso["a"]


def _righe():
    return ORCH.read_text().split("\n")


def _indice(src, testo, da=0):
    return next(i for i in range(da, len(src)) if src[i].strip() == testo)


def _in_giro(righe):
    """Segment of the window loop in a single pass (the continue stays valid)."""
    return "for _giro in range(1):\n" + textwrap.indent(textwrap.dedent("\n".join(righe)), "    ")


# --- option and null genome ---------------------------------------------------------

def test_opzione_spenta_di_default_e_documentata():
    assert _args_main([]).genoma_nullo_se_nessun_ammissibile is False
    assert _args_main([OPZ]).genoma_nullo_se_nessun_ammissibile is True
    src = ORCH.read_text()
    i = src.index(f'"{OPZ}"')
    assert "Spento di default." in src[i:i + 1200]


def test_genoma_nullo_e_il_vettore_off():
    g = ao.genoma_nullo(ao.GENOME_GROUPS[3], 17)
    assert g.tolist() == NULLO
    # same off vector as random sampling and the sparse population
    off = "off = np.array([0.0 if ag in DIRECTION_GENES else -1.0 for ag in active_genes])"
    assert ORCH.read_text().count(off) >= 3
    # decoded: no operator on, null directions
    p = PerturbationGenerator().encode_perturbation(g)
    for k, v in p.items():
        assert np.all(np.asarray(v, dtype=float) == 0.0), k
    # restricted groups: as expand_genome of the candidates (genes outside the group at -1)
    g1 = ao.genoma_nullo(ao.GENOME_GROUPS[1], 17)
    assert g1.tolist() == ao.expand_genome([0.0 if a in ao.DIRECTION_GENES else -1.0
                                            for a in ao.GENOME_GROUPS[1]], ao.GENOME_GROUPS[1]).tolist()


def test_deriva_vittima_dallo_status():
    node = types.SimpleNamespace(st_victim_drift_cm=1.234, st_victim_drift_xy_cm=1.2, st_drift_keep=True)
    d = ao.deriva_vittima_status(node)
    assert d["modulo_cm"] == 1.234 and d["xy_cm"] == 1.2 and d["drift_keep"] is True
    assert "victim_drift_cm" in d["fonte"]
    assert ao.deriva_vittima_status(types.SimpleNamespace())["modulo_cm"] is None
    # the field comes from the node status as victim_drift_xy_cm
    src = ORCH.read_text()
    assert 'self.st_victim_drift_cm = d.get("victim_drift_cm")' in src
    nodo = (ROOT / "src" / "nodes" / "perturbation_node.py").read_text()
    assert '"victim_drift_cm": round(float(np.linalg.norm(dv)) * 100.0, 3)' in nodo


# --- search without admissibles: single random (64 attempts), NSGA-III and random ---

def _record_non_valido(i):
    # good outcome but Chamfer over budget: not admissible
    return {"idx": i, "genome": [0.1] * 17, "status": "ok", "counted": True, "admissible": False,
            "budget_ok": False, "damage": 0.05, "chamfer": 30.0, "dev_mean": 0.05, "tau_s": 0.2, "slow": False}


def _randpuro_64():
    eval_log = []

    def eval_genome(x):
        eval_log.append(_record_non_valido(len(eval_log)))
        return (-0.05, 30.0, 25.0)
    off = np.array([0.0 if ag in ao.DIRECTION_GENES else -1.0 for ag in range(17)])
    rng = np.random.default_rng(np.random.SeedSequence([1001, 2]))
    rf_rec, X_all, F_all, _ = ao.random_first_search(rng, 17, "sparse", off, eval_genome, eval_log, 5.0, 64)
    return rf_rec, X_all, F_all, eval_log


def test_random_singolo_64_tentativi_senza_validi():
    rf_rec, X_all, F_all, eval_log = _randpuro_64()
    assert rf_rec["n_tries"] == 64 and rf_rec["found"] is False and rf_rec["eval_idx"] is None
    assert len(eval_log) == 64 and X_all.shape == (64, 17)


def _args(**kw):
    a = dict(genoma_nullo_se_nessun_ammissibile=False, victim_never_clean=True, select_only_valid=True,
             select="pareto_maxdamage", search="random", chamfer_budget=5.0, no_nominal_rollout=True,
             admissibility="chamfer", twin_gap0_max=30.0, twin_filter="none", topk=2, reeval=True,
             random_first_admissible=64)
    a.update(kw)
    return types.SimpleNamespace(**a)


def _segmento_scelta():
    """From genome selection to printing the front (excluded): pareto_maxdamage, branch
    without admissibles, null genome."""
    src = _righe()
    i0 = next(i for i, x in enumerate(src) if x.strip() == "# --- Selection of the genome to apply ---")
    i1 = _indice(src, "F = np.atleast_2d(res_F)", i0)
    return _in_giro(src[i0:i1])


class _Fermo:
    """Isaac, MOLA and node for the hold: records the calls."""

    def __init__(self):
        self.chiamate = []


def _esegui_scelta(tmp_path, args, eval_log, rf_rec, res_F, search):
    args.search = search
    hold = []
    ns = dict(ao.__dict__)
    ns.update(args=args, eval_log=eval_log, rf_rec=rf_rec, res_F=res_F, res_pop_F=None, history=[], k=2,
              goal=(3.0, -1.0), wp_idx=0, nom_end=None, t_win=0.0, out=tmp_path, isaac=_Fermo(), mola=_Fermo(),
              node=types.SimpleNamespace(st_victim_drift_cm=0.9, st_victim_drift_xy_cm=0.9, st_drift_keep=True),
              saved=(1.0, -1.0, 0.0), victim=types.SimpleNamespace(started=True), active_genes=list(range(17)),
              n_genes=17, best_g=None, best_F=None, ratio=None,
              deriva_inizio={"modulo_cm": 0.9, "xy_cm": 0.9, "drift_keep": True, "fonte": "x"},
              _esito_finestra=lambda rec, st: rec,
              hold_restore=lambda *a, **kw: hold.append(a[3]) or {"restored": True, "restore_pose": None})
    exec(_segmento_scelta(), ns)
    return ns, hold


@pytest.mark.parametrize("search", ["random_first", "nsga3", "random"])
def test_spenta_trattiene_come_prima(tmp_path, search):
    if search == "random_first":
        rf_rec, _, F_all, eval_log = _randpuro_64()
        res_F, cerca = np.zeros((0, 2)), "random"
    else:
        rf_rec, eval_log, res_F, cerca = None, [_record_non_valido(i) for i in range(512)], None, search
    ns, hold = _esegui_scelta(tmp_path, _args(), eval_log, rf_rec, res_F, cerca)
    h = ns["history"]
    assert len(h) == 1 and h[0]["status"] == "no_admissible" and h[0]["victim_held"] is True
    assert h[0]["victim_hold_reason"] == "nessun_ammissibile" and hold == [2]
    assert "genoma_nullo" not in h[0] and ns["genoma_nullo_rec"] is None and ns["best_g"] is None
    assert ("random_first" in h[0]) == (search == "random_first")
    assert json.loads((tmp_path / "history.json").read_text())[0]["status"] == "no_admissible"
    # hold: the window does not count as driven
    assert ao.direction_window_count(h, 3) == (2, 1)


@pytest.mark.parametrize("search,motivo,tentativi", [
    ("random_first", "nessun ammissibile in 64 tentativi", 64),
    ("nsga3", "nessun candidato ammissibile", 512),
    ("random", "nessun candidato ammissibile", 512)])
def test_accesa_applica_il_genoma_nullo(tmp_path, search, motivo, tentativi):
    if search == "random_first":
        rf_rec, _, F_all, eval_log = _randpuro_64()
        ok = np.all(np.isfinite(F_all), axis=1)
        res_F, _ = ao.pareto_front(F_all[ok], np.zeros((int(ok.sum()), 17)))
        cerca = "random"
    else:
        # pymoo without feasibles: res.F None (NSGA-III); random front on the finite ones
        rf_rec, eval_log, cerca = None, [_record_non_valido(i) for i in range(512)], search
        res_F = None if search == "nsga3" else np.array([[-0.05, 30.0]])
    ns, hold = _esegui_scelta(tmp_path, _args(genoma_nullo_se_nessun_ammissibile=True), eval_log, rf_rec,
                              res_F, cerca)
    assert ns["history"] == [] and hold == [] and not (tmp_path / "history.json").exists()
    assert ns["best_g"].tolist() == NULLO and np.all(np.isnan(ns["best_F"]))
    F = np.atleast_2d(ns["res_F"])
    assert F.ndim == 2 and F.shape[1] == 2 and (len(F) == 0) == (search == "nsga3")
    r = ns["genoma_nullo_rec"]
    assert r["motivo"] == motivo and r["tentativi"] == tentativi and r["n_valutazioni"] == len(eval_log)
    assert r["random_first"] == (search == "random_first") and r["genoma"] == NULLO
    assert r["deriva_vittima_inizio_finestra"]["modulo_cm"] == 0.9
    assert r["n_validi_selezione"] == 0


# --- winner path with the null genome: front, repeated checks, comparisons, record --

def _segmento_applicazione():
    src = _righe()
    i0 = _indice(src, "F = np.atleast_2d(res_F)")
    i1 = _indice(src, "if d31_errore:", i0)
    return "\n".join(textwrap.dedent("\n".join(src[i0:i1])).split("\n"))


class _Nodo:
    pause_start_wall = 10.0
    last_victim_open_wall = 12.0
    victim_stamp_offset = 0.0
    max_drift_lag_s = 0.5
    last_track_ratio = 1.0
    last_wps_reached = 0
    fresh_cloud = False
    last_motion = None

    def __init__(self):
        self.last_trace = []
        self.last_tau = 0.3
        self.last_loc_err_mean = self.last_loc_err_end = self.last_loc_err_mean_raw = float("nan")
        self.last_drift_start_cm = self.last_drift_end_cm = float("nan")
        self.last_perc = {"chamfer": 1.7}
        self.st_victim_drift_cm, self.st_victim_drift_xy_cm, self.st_drift_keep = 0.9, 0.9, True


def _esegui_applicazione(tmp_path, genoma_nullo_rec, best_g, best_F, res_F, ratio):
    tags = []
    node = _Nodo()

    def _rollout(genome, isa, mola_, nd, wps, horizon, tag, **kw):
        tags.append((tag, [float(v) for v in genome]))
        if tag.endswith("_applied"):
            # the victim drives the meter; the node status changes after the segment
            nd.last_trace = [{"x": 1.0, "y": -1.0, "pose_quality": 0.9}, {"x": 2.0, "y": -0.99, "pose_quality": 0.9}]
            nd.st_victim_drift_cm = nd.st_victim_drift_xy_cm = 0.6
            return (0, 0, 0), (2.0, -0.99, 0.0), 1.7, "ok", 1.0, 10
        nd.last_trace = [{"x": 1.0, "y": -1.0}, {"x": 2.0, "y": -1.0}]
        return (0, 0, 0), (2.0, -1.0, 0.0), 0.0, "ok", 1.0, 10

    def _prove(n, k, esegui, misura, nd, mola_, danno_ricerca_cm=None):
        for j in range(n):
            esegui(f"w{k}_prova{j}")
        return {"n": n, "danno_ricerca_cm": danno_ricerca_cm, "prove": []}

    args = types.SimpleNamespace(
        select="pareto_maxdamage", reeval=True, prove_vincitore=2, riallinea_dopo_assestamento=False,
        confronto_vincitore=True, warmup=0.0, settle_sec=2.0, catchup_sec=0.8, attack_from="spawn",
        stop_on="estimate", untracked="discard", drift_reset_on="keep", trace=False, horizon=1.0,
        chiudi_attacco_scoperto=True, danno_metro=True, damage="mean", dir_weight=0.0, frame_log_nodo=None,
        window_cap_factor=None, select_only_valid=True, perceptibility="chamfer", no_nominal_rollout=True,
        twin_state="ombra_g", replay_last_s=None, search="random", window_rng="seedseq", pop=1, gen=1,
        genome_group=3, untracked_policy="discard", discard_policy="x", chamfer_budget=5.0,
        damage_ref="baseline", baseline_path="b.csv", admissibility="chamfer", window_stop="ramp",
        victim="continuous", twin_align="true", warm_start=True, warm_front_after_reeval=True,
        twin_filter="none", twin_gap0_max=30.0, sampling="legacy", incompleti_regola=False,
        genoma_nullo_se_nessun_ammissibile=genoma_nullo_rec is not None)
    poly = np.array([[-6.0, -1.0], [0.0, -1.0], [6.0, -1.0]])
    ns = dict(ao.__dict__)
    ns.update(
        rollout=_rollout, prove_del_vincitore=_prove, args=args, node=node, isaac=None, mola=None,
        victim=types.SimpleNamespace(log_path=None, settle_s=1.0), attacker=None, shadow=None, out=tmp_path,
        history=[], eval_log=[_record_non_valido(i) for i in range(3)], k=2, wps=[(0.0, 0.0)],
        goal=(3.0, -1.0), goals_abs=[(5.0, 0.0)], nom_end=None, nom_path=[], baseline_poly=poly,
        frame_start=(1.0, -1.0, 0.0), best_g=best_g, best_F=best_F, res_F=res_F, ratio=ratio,
        genoma_nullo_rec=genoma_nullo_rec, sel_rec={"chosen_idx": None, "chosen": None}, rf_rec=None,
        seq_rec=None, n_genes=17, active_genes=list(range(17)), n_valid_sel=0, nominal_ok=True,
        nom_ratio=float("nan"), nom_settle_s=None, nom_assente={"senza_nominale": True}, nom_frames=0,
        nom_trav=0.0, nom_loc_mean=None, nom_loc_end=None, nom_loc_mean_raw=None, nom_loc_end_raw=None,
        nom_settle_ok=None, nom_quality=None, nom_it=None, nom_perc=None, nom_stats={}, nom_motion=None,
        nom_base_fields={}, nom_twin_start=None, n_warm=0, n_mut=0, warm_counts=None, warm_used=None,
        warm_src=None, win_timeout=False, win_cap=None, first_win_time=1.0, t_win=0.0, wp_idx=0,
        victim_lost=False, att_err_cm=None, att_err_deg=None, ref_lbl="dalla baseline", reeval=None,
        _esito_finestra=lambda rec, st: rec,
        window_damage=lambda *a, **kw: {"dev_end": 0.01, "dev_mean": 0.01, "dev_max": 0.01, "damage": 0.01},
        true_path=lambda t: [(r["x"], r["y"]) for r in t],
        _victim_window_record=lambda *a: {"baseline": None},
        _refresh_victim_scans=lambda *a, **kw: None, motion_vittima=lambda n: {},
        node_gen_limits=lambda n: {}, structural_fields=lambda n, a: {}, twin_config_fields=lambda a: {},
        subprocess=types.SimpleNamespace(run=lambda *a, **kw: (_ for _ in ()).throw(OSError("niente gpu"))),
        confronti_d31=lambda *a, **kw: None)
    exec(_segmento_applicazione(), ns)
    return ns, tags


def test_strada_del_vincitore_col_genoma_nullo(tmp_path):
    args = _args(genoma_nullo_se_nessun_ammissibile=True, search="nsga3")
    g, F, ratio, res_F, rec = ao.scelta_genoma_nullo(args, None, [_record_non_valido(i) for i in range(3)],
                                                      0, list(range(17)), 17, None)
    rec["deriva_vittima_inizio_finestra"] = {"modulo_cm": 0.9}
    ns, tags = _esegui_applicazione(tmp_path, rec, g, F, res_F, ratio)
    # repeated checks and application with the null genome, the victim drives
    assert [t for t, _ in tags] == ["w2_prova0", "w2_prova1", "w2_applied"]
    assert all(gg == NULLO for _, gg in tags)
    h = ns["history"]
    assert len(h) == 1 and "victim_held" not in h[0] and h[0]["genome"] == NULLO
    assert h[0]["genoma_nullo"] is True and h[0]["applied_status"] == "ok"
    d = h[0]["genoma_nullo_dettagli"]
    assert d["motivo"] == "nessun candidato ammissibile" and d["tentativi"] == 3
    assert d["P_misurato_cm"] == 1.7 and h[0]["chamfer_applicato_cm"] == 1.7
    assert d["deriva_vittima_inizio_finestra"]["modulo_cm"] == 0.9
    assert d["deriva_vittima_prima_applicazione"]["modulo_cm"] == 0.9
    assert d["deriva_vittima_dopo_applicazione"]["modulo_cm"] == 0.6
    # search objectives not measured: None; no winner in the twin
    assert h[0]["damage_cm"] is None and h[0]["perturbation"] is None and h[0]["ratio"] is None
    assert h[0]["vincitore_idx"] is None and h[0]["danno_metro_vincitore"] is None
    assert h[0]["chamfer_vincitore_cm"] is None and h[0]["gemello_scelto"] is None
    assert h[0]["prove_vincitore"]["danno_ricerca_cm"] is None
    assert h[0]["pareto_F"] == [] and h[0]["danno_metro_vittima"] is not None
    # the window counts as driven
    assert ao.direction_window_count(h + [{"window": 3}], 3) == (3, 0)
    json.loads((tmp_path / "history.json").read_text())   # written and readable


def test_strada_del_vincitore_spenta_invariata(tmp_path):
    """Without null genome the record of a normal winner has no new fields."""
    g = np.array([0.2] * 17)
    ns, tags = _esegui_applicazione(tmp_path, None, g, np.array([-0.05, 3.0]), np.array([[-0.05, 3.0]]), 1.0)
    h = ns["history"][0]
    assert "genoma_nullo" not in h and "genoma_nullo_dettagli" not in h
    assert h["damage_cm"] == pytest.approx(5.0) and h["perturbation"] == 3.0 and h["ratio"] == 1.0
    assert h["prove_vincitore"]["danno_ricerca_cm"] == pytest.approx(5.0)


def test_sorgente_ramo_spento_invariato():
    """With the option off the branch without admissibles is unchanged: same lines."""
    src = _righe()
    i = _indice(src, 'n_dos_win = sum(1 for e in eval_log if e.get("dos"))')
    j = next(x for x in range(i, len(src)) if src[x].strip().startswith(
        "if genoma_nullo_rec is None and (best_g is None or not np.all(np.isfinite(best_F))):"))
    assert src[j + 1].strip() == 'print(f"  nessun candidato ammissibile ({n_dos_win} DoS, "'
    blocco = "\n".join(src[j:j + 50])
    assert 'history[-1].update(victim_hold_fields("nessun_ammissibile", victim))' in blocco
    assert 'history[-1]["hold_restore"] = hold_restore(isaac, mola, node, k, saved)' in blocco


# --- new queue conditions -----------------------------------------------------------

def test_condizioni_senza_budget_nella_lista():
    ids = [c["id"] for c in coda.CONDIZIONI]
    assert len(ids) == 20
    assert ids.index("casuale_senzabudget") == ids.index("casuale_eps5") - 1
    assert ids.index("randpuro_senzabudget") == ids.index("randpuro_eps5") - 1
    cs, rs = coda.PER_ID["casuale_senzabudget"], coda.PER_ID["randpuro_senzabudget"]
    assert (cs["braccio"], cs["budget"], cs["tipo"]) == ("random", "", "ricerca")
    assert (rs["braccio"], rs["budget"], rs["tipo"]) == ("random", "", "randpuro")
    assert coda.espandi(["casuale_*"])[0] == "casuale_senzabudget"
    assert coda.espandi(["randpuro_*"])[0] == "randpuro_senzabudget"


def test_ore_delle_condizioni_senza_budget():
    ore = {"ricerca casuale soglia 5 cm": {5: (6.698, None), 8: (10.711, None)},
           "random puro soglia 50 cm": {5: (0.043, 0.858), 8: (0.062, 1.367)}}
    r = coda.stima_run("casuale_senzabudget", 5, 3, 32, 16, 2, 64, None, None, ore)
    assert r["ore"] == [coda.ORE_MISURATE_CASUALE, 10.711] and r["rollout_per_finestra"] == [514, 514]
    r = coda.stima_run("randpuro_senzabudget", 5, 3, 32, 16, 2, 64, None, None, ore)
    assert r["ore"] == [coda.ORE_STIMA_RANDPURO_SENZABUDGET, 1.367] and r["rollout_per_finestra"][1] == 64
    # maximum time as the sibling conditions; without a measured time as the others (12 h)
    assert coda.tempo_max_s(r["ore"][1]) == coda.tempo_max_s(1.367)
    assert coda.stima_run("randpuro_senzabudget", 5, 3, 32, 16, 2, 64, None, None, {})["ore"] == [None, None]
    # the measured mean declared in the source
    assert round((7.736 + 8.092 + 7.774) / 3, 3) == coda.ORE_MISURATE_CASUALE


def _blocco_run_campaign(env):
    """run_campaign.sh block from 'case "$ARM" in' (EXTRA) to the single random: EXTRA and
    SERIEB_FLAGS."""
    src = RUN_CAMPAIGN.read_text().splitlines()
    i0 = next(i for i, x in enumerate(src) if x.startswith('case "$ARM" in'))
    i1 = next(i for i in range(i0, len(src)) if src[i].startswith('case "$TWIN_FILTER" in'))
    codice = "\n".join(src[i0:i1]) + '\nprintf "%s|%s|%s" "$EXTRA" "$SERIEB_FLAGS" "$BUDGET_SUFFIX$MODE_SUFFIX"\n'
    base = {"PATH": "/usr/bin:/bin", "POP": "32", "GEN": "16", "GROUP": "3", "SELECT": "pareto_maxdamage",
            "WINDOW_RNG": "seedseq", "SAMPLING": "", "RANDOM_FIRST": "", "CHAMFER_BUDGET": "",
            "WARM_START": "true", "DRIFT_RESET_ON": "keep", "WINDOW_STOP": "ramp", "STOP_RAMP_K": "1.0"}
    r = subprocess.run(["bash", "-c", codice], capture_output=True, text=True, env={**base, **env})
    assert r.returncode == 0, r.stderr
    return r.stdout.split("|")


def _a_secco(tmp_path, voci, **extra):
    (tmp_path / "b.csv").write_text("x,y\n")
    env = {**os.environ, "RADICE": str(tmp_path / "serieB"), "PILOTA_DISCO": str(tmp_path / "manca"),
           "ORE_T6": str(tmp_path / "manca.csv"), "BASELINE_DISCO": str(tmp_path / "manca"),
           "BASELINE_SERIEB": str(tmp_path / "b.csv"), "SEMI": "1001", "SELECT": "pareto_maxdamage"}
    for v in ("GENOMA_NULLO", "CONTROLLO_OMBRA_VIVA", "ORIGINE_COERENTE"):
        env.pop(v, None)
    env.update(extra)
    p = subprocess.run(["bash", str(CODA_SH), "--a-secco", *voci], capture_output=True, text=True,
                       env=env, timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    assert not (tmp_path / "serieB").exists()
    return p.stdout


def _righe_env(out):
    return [r for r in out.splitlines() if "+ env " in r and "scripts/run_campaign.sh" in r]


def _righe_attese(out):
    return [r for r in out.splitlines() if "opzioni attese:" in r]


def test_comando_delle_condizioni_senza_budget(tmp_path):
    out = _a_secco(tmp_path, ["casuale_senzabudget", "randpuro_senzabudget"])
    env = _righe_env(out)
    att = _righe_attese(out)
    assert len(env) == 2 and len(att) == 2
    cs, rs = env
    assert " CHAMFER_BUDGET= " in cs and "RANDOM_FIRST" not in cs and cs.endswith(
        "scripts/run_campaign.sh straight random 3 1 32 16 > " + str(tmp_path / "serieB/casuale_senzabudget/s1001/campagna.log") + " 2>&1")
    assert " SELECT=pareto_maxdamage TOPK=2 " in cs
    assert " CHAMFER_BUDGET= " in rs and " SELECT=pareto_maxdamage TOPK=2 RANDOM_FIRST=64 " in rs
    assert "scripts/run_campaign.sh straight random 3 1 1 1 > " in rs
    assert "--chamfer-budget" not in att[0] and "--random-first-admissible" not in att[0]
    assert "--chamfer-budget" not in att[1] and "[--random-first-admissible 64]" in att[1]
    assert "braccio random, budget nessuno" in out
    # run_campaign.sh: no --chamfer-budget, single random with K = 64 only for randpuro
    e, f, suff = _blocco_run_campaign({"ARM": "random"})
    assert e == "--search random --pop 32 --gen 16 --genome-group 3" and "--random-first" not in f
    assert "_eps" not in suff
    e, f, suff = _blocco_run_campaign({"ARM": "random", "POP": "1", "GEN": "1", "RANDOM_FIRST": "64"})
    assert e == "--search random --pop 1 --gen 1 --genome-group 3" and "--random-first-admissible 64" in f
    assert "_eps" not in suff and "_first64" in suff
    # orchestrator: no budget, single random allowed with pareto_maxdamage
    a = _args_main(["--search", "random", "--pop", "1", "--gen", "1", "--genome-group", "3",
                    "--random-first-admissible", "64", "--select", "pareto_maxdamage"])
    assert a.chamfer_budget is None and ao.check_random_first_args(a) is None


def _senza_numeri(t):
    # times, free GB and watchdog heartbeat age change between two calls
    return re.sub(r"\d+", "N", t)


def test_coda_genoma_nullo_0_1_e_valore_errato(tmp_path):
    voci = ["baseline", "randpuro_eps5"]
    assente = _a_secco(tmp_path, voci)
    zero = _a_secco(tmp_path, voci, GENOMA_NULLO="0")
    uno = _a_secco(tmp_path, voci, GENOMA_NULLO="1")
    errato = _a_secco(tmp_path, voci, GENOMA_NULLO="si")
    # 0 and missing: same output, option never mentioned
    assert _senza_numeri(assente) == _senza_numeri(zero)
    assert OPZ not in assente and "GENOMA_NULLO" not in assente
    # 1: only in the attacked run, in ORCH_EXTRA and among the expected options
    env_b, env_r = _righe_env(uno)
    att_b, att_r = _righe_attese(uno)
    assert OPZ not in env_b and OPZ not in att_b
    assert ("\\ " + OPZ + " ") in env_r and f"[{OPZ}]" in att_r
    # apart from the option, lines equal to the off ones
    env0 = _righe_env(zero)
    assert env_r.replace("\\ " + OPZ, "") == env0[1] and env_b == env0[0]
    assert att_r.replace(f" [{OPZ}]", "") == _righe_attese(zero)[1]
    assert "CONTROLLO FALLITO: GENOMA_NULLO" not in uno
    # wrong value: check fails like the other 0/1 variables
    assert "CONTROLLO FALLITO: GENOMA_NULLO: 0 o 1 (dato: si)" in errato
    assert "a secco: il lancio vero si fermerebbe ai controlli falliti sopra" in errato


def test_coda_passa_l_opzione_come_le_altre():
    sh = CODA_SH.read_text()
    assert 'GENOMA_NULLO="${GENOMA_NULLO:-0}"' in sh
    assert f'[[ "$TIPO" != "baseline" && "$GENOMA_NULLO" == "1" ]] && ORCH_X="$ORCH_X {OPZ}"' in sh
    assert f'[[ "$TIPO" != "baseline" && "$GENOMA_NULLO" == "1" ]] && ATTESE+=("{OPZ}")' in sh
    assert "${OPZ_0311_DESC}${OPZ_0511_DESC} DDS_PROFILE=" in sh
    # run_campaign.sh unchanged: the option goes through ORCH_EXTRA
    assert "GENOMA_NULLO" not in RUN_CAMPAIGN.read_text()
