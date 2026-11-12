#!/usr/bin/env python3
"""
Pareto front (max damage, min Chamfer) of the valid candidates of a window and
the point to apply.

Selection rule (--select pareto_maxdamage): among all candidates evaluated in
the window keep the valid ones, compute the non-dominated front on (damage to
maximise, Chamfer to minimise) and apply the front point with the largest
damage; ties go to the lower Chamfer, then the lower index. No re-evaluation and
no filter on "slow" candidates (tau and the slow flag are only logged).

A record (fields written by the orchestrator in eval_log) is valid if the
rollout status is ok or arrived and the measure is counted, admissible is true
(stealth, Chamfer budget, compute time, f2''), the twin is aligned at start and
not removed by the anomalous-twin filter, and damage and Chamfer are finite.
An optional budget adds Chamfer <= budget, to recompute the front with a
threshold other than the run's.

With solo_validi (--select-only-valid) the stricter checks of motivi_non_valido
also apply: within the Chamfer budget, tracked, no est_capped, no twin failure.

Side-effect free: used by the orchestrator for selection and by the analysis
scripts to draw the same front.
"""

import math

STATI_VALIDI = ("ok", "arrived")
# Twin outcomes without a pose: counted as twin failure by motivi_non_valido.
STATI_GUASTO_GEMELLO = ("no_pose_from_new_mola", "pose_stale")


def _finito(v):
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def motivi_non_valido(rec, budget=None, da_esito_gemello=False):
    """Reasons why a record cannot win with --select-only-valid (empty = valid).

    Checks: Chamfer within the budget (always true without a budget), tracked,
    no est_capped, no twin failure (guasto_gemello, no_pose_from_new_mola,
    pose_stale), plus the usual exclusions: discarded measure or missing damage,
    not admissible for other reasons (stealth, compute, f2''), twin misaligned
    or filtered. With da_esito_gemello (--no-nominal-rollout) a candidate that
    esito_gemello labels gemello_perde_posa (twin checks passed, no pose) gets
    that reason instead of guasto_gemello and stays invalid. With
    --mola-se-ne-accorge, accorge_ok false gives mola_se_ne_accorge.
    """
    if not isinstance(rec, dict):
        return ["record_assente"]
    motivi = []
    stato = str(rec.get("status", "")).split("(")[0].strip()
    if da_esito_gemello and rec.get("esito_gemello") == "gemello_perde_posa":
        motivi.append("gemello_perde_posa")
    elif rec.get("guasto_gemello") or stato in STATI_GUASTO_GEMELLO:
        motivi.append("guasto_gemello")
    if rec.get("accorge_ok") is False:
        motivi.append("mola_se_ne_accorge")
    if stato == "untracked":
        motivi.append("untracked")
    if stato == "est_capped":
        motivi.append("est_capped")
    if budget is not None and (not _finito(rec.get("chamfer"))
                               or float(rec["chamfer"]) > float(budget)):
        motivi.append("chamfer_oltre_budget")
    if rec.get("counted") is False:
        motivi.append("scartato")
    elif not _finito(rec.get("damage")):
        motivi.append("danno_assente")
    if rec.get("admissible") is False:
        # only the budget violated (or MOLA noticing): report that reason, not
        # a generic one
        altri = any(rec.get(c) is False for c in ("stealth_ok", "calc_ok", "f2_ok"))
        spiegato = rec.get("budget_ok") is False or rec.get("accorge_ok") is False
        if altri or not spiegato:
            motivi.append("non_ammissibile")
        elif rec.get("budget_ok") is False and "chamfer_oltre_budget" not in motivi:
            motivi.append("chamfer_oltre_budget")
    elif not rec.get("admissible") and rec.get("counted") is not False:
        motivi.append("non_ammissibile")
    if rec.get("twin_misaligned"):
        motivi.append("gemello_disallineato")
    if rec.get("twin_filtered"):
        motivi.append("gemello_filtrato")
    return motivi


def candidato_valido(rec, budget=None, solo_validi=False):
    """True if an evaluation record can enter the front."""
    if not isinstance(rec, dict):
        return False
    if solo_validi and motivi_non_valido(rec, budget):
        return False
    stato = str(rec.get("status", "")).split("(")[0]
    # --incompleti-regola: timeout/stopped_early classed attacco_valido are valid
    if stato not in STATI_VALIDI and rec.get("incompleto_classe") != "attacco_valido":
        return False
    if rec.get("counted") is False:
        return False
    if not rec.get("admissible"):
        return False
    if rec.get("twin_misaligned"):
        return False
    # Anomalous twin excluded by --twin-filter (field absent in older runs).
    if rec.get("twin_filtered"):
        return False
    if not _finito(rec.get("damage")) or not _finito(rec.get("chamfer")):
        return False
    if budget is not None and float(rec["chamfer"]) > float(budget):
        return False
    return True


def indici_non_dominati(danno, chamfer):
    """Non-dominated indices, maximising damage and minimising Chamfer.

    Two identical points do not dominate each other: both stay.
    """
    n = len(danno)
    out = []
    for i in range(n):
        dominato = False
        for j in range(n):
            if j == i:
                continue
            if (danno[j] >= danno[i] and chamfer[j] <= chamfer[i]
                    and (danno[j] > danno[i] or chamfer[j] < chamfer[i])):
                dominato = True
                break
        if not dominato:
            out.append(i)
    return out


def fronte_pareto(records, budget=None, solo_validi=False):
    """Front of the valid candidates and the point to apply.

    Args: records, list of evaluations (dicts as in history.json "evaluations"),
    the list index is the candidate id; budget, max Chamfer (cm) or None;
    solo_validi, keep only candidates without motivi_non_valido.
    Returns (front, chosen): front sorted by increasing Chamfer (then decreasing
    damage, then index) with keys idx, genome, damage, chamfer, tau_s, slow,
    status; chosen is the front point with max damage (ties: lower Chamfer, then
    lower index), or None if no candidate is valid.
    """
    validi = [i for i, r in enumerate(records) if candidato_valido(r, budget, solo_validi)]
    if not validi:
        return [], None
    danno = [float(records[i]["damage"]) for i in validi]
    cham = [float(records[i]["chamfer"]) for i in validi]
    fronte = []
    for j in indici_non_dominati(danno, cham):
        i = validi[j]
        r = records[i]
        fronte.append({"idx": i,
                       "genome": list(r.get("genome") or []),
                       "damage": danno[j],
                       "chamfer": cham[j],
                       "tau_s": r.get("tau_s"),
                       "slow": bool(r.get("slow")),
                       "status": r.get("status")})
    fronte.sort(key=lambda p: (p["chamfer"], -p["damage"], p["idx"]))
    scelto = min(fronte, key=lambda p: (-p["damage"], p["chamfer"], p["idx"]))
    return fronte, scelto
