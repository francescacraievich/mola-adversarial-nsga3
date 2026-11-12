#!/usr/bin/env python3
"""Selection "only a valid candidate wins" (--select-only-valid): invalidity reasons, maxdamage
selection (top_maxdamage) and pareto_maxdamage (fronte_pareto with solo_validi).

    python3 -m pytest tests/test_selezione_solo_validi.py -q
"""

import random
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.fronte_pareto import (  # noqa: E402
    candidato_valido, fronte_pareto, motivi_non_valido)
from src.optimization.attack_orchestrator import top_maxdamage  # noqa: E402


def rec(damage, chamfer, status="ok", genome=None, **extra):
    """Counted record like those of eval_genome (history.json, evaluations)."""
    r = {"genome": genome if genome is not None else [damage, chamfer],
         "damage": damage, "dev_mean": damage, "chamfer": chamfer, "status": status,
         "counted": True, "admissible": True, "stealth_ok": True, "budget_ok": True,
         "calc_ok": True, "f2_ok": True, "slow": False, "tau_s": 0.3,
         "twin_misaligned": False}
    r.update(extra)
    return r


def scartato(status, **extra):
    """Discarded record like those of eval_genome (no damage, no admissibility)."""
    r = {"genome": [0.5, 0.5], "dev": None, "chamfer": 3.0, "status": status,
         "counted": False, "dos": False}
    r.update(extra)
    return r


def args_(only_valid, budget=None, topk=3, search="nsga3", admissibility="stealth"):
    return types.SimpleNamespace(select_only_valid=only_valid, chamfer_budget=budget,
                                 topk=topk, search=search, admissibility=admissibility)


def top_originale(eval_log, args):
    """Block of main before --select-only-valid (maxdamage selection), copied."""
    cands = [e for e in eval_log if e.get("admissible") and e.get("damage") is not None
             and (not e.get("slow") or args.search == "fixed" or args.admissibility == "none") and not e.get("twin_misaligned")
             and not e.get("twin_filtered")]
    cands.sort(key=lambda e: -e["damage"])
    seen, top = set(), []
    for e in cands:
        key = tuple(round(v, 4) for v in e["genome"])
        if key in seen:
            continue
        seen.add(key); top.append(e)
        if len(top) >= args.topk:
            break
    return top


# --- invalidity reasons ---

def test_motivi_della_decisione():
    assert motivi_non_valido(rec(0.1, 2.0)) == []
    assert motivi_non_valido(rec(0.1, 2.0, status="arrived")) == []
    assert motivi_non_valido(rec(0.1, 2.0, status="est_capped")) == ["est_capped"]
    assert motivi_non_valido(rec(0.1, 2.0, status="untracked(0.55)")) == ["untracked"]
    assert "guasto_gemello" in motivi_non_valido(scartato("no_pose_from_new_mola (publisher nel grafo: 1, x)"))
    assert "guasto_gemello" in motivi_non_valido(scartato("pose_stale"))
    assert "guasto_gemello" in motivi_non_valido(scartato("no_pose_from_new_mola", guasto_gemello=True))
    # twin failure even if the outcome were ok
    assert motivi_non_valido(rec(0.1, 2.0, guasto_gemello=True)) == ["guasto_gemello"]
    assert motivi_non_valido(rec(0.1, 80.0), budget=70.0) == ["chamfer_oltre_budget"]
    # run record with violated budget: a single reason, the Chamfer
    r = rec(0.1, 80.0, admissible=False, budget_ok=False)
    assert motivi_non_valido(r, budget=70.0) == ["chamfer_oltre_budget"]
    assert motivi_non_valido(r) == ["chamfer_oltre_budget"]
    r = rec(0.1, 2.0, admissible=False, stealth_ok=False, inadmissible_reason="posa assente")
    assert motivi_non_valido(r) == ["non_ammissibile"]
    assert motivi_non_valido(rec(0.1, 2.0, twin_misaligned=True)) == ["gemello_disallineato"]
    assert motivi_non_valido(rec(0.1, 2.0, twin_filtered=True)) == ["gemello_filtrato"]
    assert motivi_non_valido(scartato("mola_crash")) == ["scartato"]


def test_senza_budget_ammissibile_e_valido():
    # without a budget the Chamfer does not remove validity, whatever its value
    for ch in (0.0, 5.0, 70.0, 500.0, float("nan")):
        assert motivi_non_valido(rec(0.2, ch)) == []
    # and only admissible ok/arrived are valid: same choice with and without the option
    recs = [rec(0.1, 3.0), rec(0.4, 300.0), rec(0.3, 20.0, status="arrived")]
    assert top_maxdamage(recs, args_(True)) == top_maxdamage(recs, args_(False))
    assert fronte_pareto(recs, solo_validi=True) == fronte_pareto(recs)


# --- option off: same winners ---

def _log_casuale(rnd, n):
    stati = ["ok", "arrived", "est_capped", "untracked(0.41)", "timeout", "stopped_early(0.43m)"]
    log = []
    for _ in range(n):
        if rnd.random() < 0.15:
            log.append(scartato(rnd.choice(["no_pose_from_new_mola (x)", "pose_stale", "mola_crash"]),
                                guasto_gemello=rnd.random() < 0.5))
            continue
        g = [round(rnd.choice([0.1, 0.2, 0.3]), 1), round(rnd.random(), 4)] if rnd.random() < 0.2 else None
        r = rec(rnd.choice([0.05, 0.1, 0.2, 0.2, 0.35, 0.5]), rnd.uniform(0.5, 120.0),
                status=rnd.choice(stati), genome=g,
                slow=rnd.random() < 0.2, twin_misaligned=rnd.random() < 0.1,
                twin_filtered=rnd.random() < 0.1)
        if r["chamfer"] > 70.0 and rnd.random() < 0.5:
            r["admissible"] = False; r["budget_ok"] = False
        if rnd.random() < 0.1:
            r["admissible"] = False; r["stealth_ok"] = False
        log.append(r)
    return log


def test_spenta_stessi_candidati_di_prima():
    rnd = random.Random(26)
    for _ in range(400):
        log = _log_casuale(rnd, rnd.randint(0, 40))
        for a in (args_(False), args_(False, budget=70.0, topk=2),
                  args_(False, search="fixed"), args_(False, admissibility="none", topk=1)):
            assert top_maxdamage(log, a) == top_originale(log, a)
        for b in (None, 5.0, 70.0):
            fr, sc = fronte_pareto(log, budget=b)
            assert (fr, sc) == fronte_pareto(log, budget=b, solo_validi=False)


# --- option on: an invalid candidate never wins ---

def test_maxdamage_non_valido_col_danno_massimo_non_vince():
    for cattivo in (rec(0.9, 2.0, status="est_capped"),
                    rec(0.9, 2.0, status="untracked(0.52)"),
                    rec(0.9, 2.0, guasto_gemello=True),
                    rec(0.9, 2.0, status="pose_stale"),
                    rec(0.9, 90.0)):
        log = [rec(0.1, 2.0), cattivo, rec(0.2, 4.0)]
        spenta = top_maxdamage(log, args_(False, budget=70.0, topk=1))
        accesa = top_maxdamage(log, args_(True, budget=70.0, topk=1))
        if cattivo["chamfer"] <= 70.0:
            assert spenta[0] is cattivo        # would win without the option
        assert accesa[0]["damage"] == 0.2 and accesa[0] is not cattivo
        assert all(e is not cattivo for e in top_maxdamage(log, args_(True, budget=70.0, topk=5)))


def test_pareto_non_valido_non_entra_nel_fronte():
    log = [rec(0.1, 2.0), rec(0.9, 1.0, guasto_gemello=True), rec(0.3, 6.0)]
    fr, sc = fronte_pareto(log)
    assert sc["idx"] == 1                      # would win without the option
    fr_v, sc_v = fronte_pareto(log, solo_validi=True)
    assert sc_v["idx"] == 2 and all(p["idx"] != 1 for p in fr_v)
    assert not candidato_valido(log[1], solo_validi=True)
    # est_capped and untracked stay out (outcome not ok/arrived)
    log = [rec(0.1, 2.0), rec(0.9, 1.0, status="est_capped"), rec(0.8, 1.0, status="untracked(0.5)")]
    assert fronte_pareto(log, solo_validi=True)[1]["idx"] == 0
    # budget: beyond the threshold it is not admitted
    log = [rec(0.1, 2.0), rec(0.9, 80.0)]
    assert fronte_pareto(log, budget=70.0, solo_validi=True)[1]["idx"] == 0


def test_casuale_vincitore_sempre_valido():
    rnd = random.Random(1026)
    for _ in range(400):
        log = _log_casuale(rnd, rnd.randint(1, 40))
        for b in (None, 5.0, 70.0):
            for e in top_maxdamage(log, args_(True, budget=b, topk=3)):
                assert motivi_non_valido(e, b) == []
            _, sc = fronte_pareto(log, budget=b, solo_validi=True)
            if sc is not None:
                assert motivi_non_valido(log[sc["idx"]], b) == []


# --- no valid candidate: existing rule (window without genome) ---

def test_nessun_valido_nessun_vincitore():
    log = [rec(0.9, 2.0, status="est_capped"), rec(0.5, 2.0, status="untracked(0.3)"),
           scartato("no_pose_from_new_mola", guasto_gemello=True), rec(0.4, 90.0)]
    # with the option no candidate: best_g None and the window follows the existing
    # "no_admissible" branch (held with --victim-never-clean)
    assert top_maxdamage(log, args_(True, budget=70.0)) == []
    assert fronte_pareto(log, budget=70.0, solo_validi=True) == ([], None)
    # without the option the est_capped one would win
    assert top_maxdamage(log, args_(False, budget=70.0))[0]["status"] == "est_capped"


# --- run_campaign.sh: options with SERIE_B=1 and comando.txt (no start) ---

ROOT = Path(__file__).resolve().parent.parent
TOLTE = ("SERIE_B", "WINDOW_RNG", "SEEDS", "ICP_QUALITY_LOG", "PERC_TERMS", "SAMPLING",
         "RANDOM_FIRST", "TWIN_FILTER", "MOLA_LO_DEBUG_ICP_QUALITY", "CHAMFER_BUDGET",
         "MAX_DRIFT_LAG_S", "RUN_SUFFIX", "MIN_LATENCY_MS", "FAST_OPS", "REPORT_MATCHED",
         "SYNC_WAIT", "SYNC_GUARD", "NODE_EXTRA", "ORCH_EXTRA", "SELECT", "TOPK")


def _campagna(tmp_path, env_extra, arm="nsga3"):
    """Part of run_campaign.sh before the process checks, then the block
    that writes comando.txt with OUT in the temporary folder.
    """
    import os
    import subprocess
    src = (ROOT / "scripts" / "run_campaign.sh").read_text().splitlines()
    fine = next(i for i, l in enumerate(src) if l.startswith("if pgrep -f"))
    i0 = next(i for i, l in enumerate(src) if "comando.txt" in l)
    a = max(i for i in range(i0) if src[i].strip() == 'if [[ "$SERIE_B" == "1" ]]; then')
    b = next(i for i in range(i0, len(src)) if src[i].strip() == "fi")
    corpo = "\n".join(src[:fine]).replace('cd "$ROOT"', ': cd')
    corpo += ("\nOUT=\"$PWD/rep_01\"; SEED_I=1001\n" + "\n".join(src[a:b + 1])
              + "\necho \"FLAGS=$EXTRA $SERIEB_FLAGS\"\n")
    f = tmp_path / "blocco.sh"
    f.write_text(corpo)
    env = {k: v for k, v in os.environ.items() if k not in TOLTE}
    env.update(env_extra)
    r = subprocess.run(["bash", str(f), "straight", arm, "3", "1", "4", "2"],
                       capture_output=True, text=True, env=env, cwd=tmp_path)
    flags = next((l.split("=", 1)[1] for l in r.stdout.splitlines() if l.startswith("FLAGS=")), None)
    cmd = tmp_path / "rep_01" / "comando.txt"
    return r.returncode, flags, (cmd.read_text() if cmd.exists() else None), r.stdout


def test_campagna_senza_serie_b_invariata(tmp_path):
    rc, flags, cmd, _ = _campagna(tmp_path, {"CHAMFER_BUDGET": "70"})
    assert rc == 0 and "--select-only-valid" not in flags and "--perceptibility" not in flags
    assert flags.split() == ["--search", "nsga3", "--pop", "4", "--gen", "2", "--genome-group", "3",
                             "--chamfer-budget", "70"]
    assert cmd is None                          # no new file without SERIE_B


def test_campagna_serie_b_solo_validi_e_chamfer(tmp_path):
    rc, flags, cmd, _ = _campagna(tmp_path, {"SERIE_B": "1", "SEEDS": "1001", "CHAMFER_BUDGET": "70"})
    assert rc == 0
    assert flags.endswith(" --select-only-valid --perceptibility chamfer")
    assert "--chamfer-budget 70" in flags
    righe = cmd.splitlines()
    assert righe[0] == "CHAMFER_BUDGET=70 SELECT=maxdamage SERIE_B=1"
    assert "--chamfer-budget 70" in righe[1] and "--select-only-valid --perceptibility chamfer" in righe[1]
    assert '--seed "1001"' in righe[1] and '--select "maxdamage"' in righe[1]


def test_campagna_serie_b_senza_budget_e_ratio(tmp_path):
    rc, flags, cmd, _ = _campagna(tmp_path, {"SERIE_B": "1", "SEEDS": "1001"})
    assert rc == 0 and "--chamfer-budget" not in flags
    assert cmd.splitlines()[0] == "CHAMFER_BUDGET= SELECT=maxdamage SERIE_B=1"
    (tmp_path / "ratio").mkdir()
    rc, _, _, out = _campagna(tmp_path / "ratio", {"SERIE_B": "1", "SEEDS": "1001", "SELECT": "ratio"})
    assert rc == 1 and "SERIE_B=1 richiede SELECT=maxdamage o pareto_maxdamage" in out


if __name__ == "__main__":
    # the campaign script tests need tmp_path: pytest only
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and v.__code__.co_argcount == 0]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} test passati")
