"""Variant G, --g-twin-lost-ends-run: G twin that loads the shadow state and publishes no pose.
Pure functions, no ROS or MOLA; the last test reads a recorded pilot log, if present.
"""

import json
import re
import types
from pathlib import Path

import pytest

import src.optimization.twin_variants as tv

ORCH = Path(__file__).resolve().parents[1] / "src" / "optimization" / "attack_orchestrator.py"
PILOTA = (Path(__file__).resolve().parents[1] / "data/attack/verifica/2026-10-26_pilota_G/s1001/campagna/"
              "straight/nsga3_g3_p4x2_lag0.5_sel2_warm_stopramp_k1.0_reliable_serieB_s1001_pt1_oprandom_"
              "sgross_pilotaT5_dmin130/rep_01")
NO_POSE = "no_pose_from_new_mola (publisher nel grafo: 0, processi vivi: 4)"

LOG = """[04:10:00.0950|INFO |LidarOdometry] [full_state] caricato /x/ombra_w4.molastate (6654775 byte) [localmap]
[04:10:00.0998|INFO |LidarOdometry] [full_state] salvato /x/ombra_w4.molastate.risalvato_w4_nominal (6654775 byte, stamp 1) [l]
[LidarOdometry] pathStep=230 timestamp=x goodness=0.000 minRequired=0.200 iters=7 termReason=A isGood=0 adapt_thres_sigma=0.0400 consecutive_bad=0
[LidarOdometry] pathStep=230 timestamp=x goodness=0.000 minRequired=0.200 iters=6 termReason=A isGood=0 adapt_thres_sigma=0.0400 consecutive_bad=1
[LidarOdometry] pathStep=230 timestamp=x goodness=0.000 minRequired=0.200 iters=6 termReason=A isGood=0 adapt_thres_sigma=0.0400 consecutive_bad=2
[04:10:16.0223|INFO |LidarOdometry] Run totals: registrations=351 no_motion_model=115 (32.76%) icp_rejected=122 (34.76%)
"""


def _args(**kw):
    base = dict(twin_state="ombra_g", victim="continuous", twin_align="attacker",
                uniform_sim_time="true", twin_init="true", drift_reset_on="genome",
                attack_from="spawn", repeat_genome=None, repeat_spec=None,
                map_gate_test=False, replay_gate=0, g_overlay=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


def _overlay(tmp_path):
    root = tmp_path / "install_G1"
    for p in tv.G_OVERLAY_PKGS:
        (root / p / "lib" / "x86_64-linux-gnu").mkdir(parents=True)
    return str(root)


def test_righe_del_log():
    r = tv.g_twin_lost_lines(LOG)
    assert r == {"n_loaded": 1, "n_icp_lines": 3, "n_icp_good": 0, "n_icp_bad": 3,
                 "consecutive_bad_last": 2, "consecutive_bad_max": 2,
                 "registrations": 351, "no_motion_model": 115, "icp_rejected": 122}


def test_righe_assenti():
    r = tv.g_twin_lost_lines(None)
    assert r["n_loaded"] == 0 and r["n_icp_lines"] == 0
    assert r["consecutive_bad_last"] is None and r["icp_rejected"] is None
    # without end-of-run totals (MOLA killed) the per-scan rows remain
    r = tv.g_twin_lost_lines(LOG.rsplit("\n", 2)[0])
    assert r["icp_rejected"] is None and r["n_icp_bad"] == 3


def test_decisione():
    lines = tv.g_twin_lost_lines(LOG)
    assert tv.g_twin_lost(NO_POSE, "ombra_g", lines)
    # other variants, other nominal outcomes, state not loaded, log missing
    assert not tv.g_twin_lost(NO_POSE, "ombra_e", lines)
    assert not tv.g_twin_lost(NO_POSE, "clean", lines)
    assert not tv.g_twin_lost("mola_not_ready: x", "ombra_g", lines)
    assert not tv.g_twin_lost("ombra_g_state_missing: x", "ombra_g", lines)
    assert not tv.g_twin_lost(None, "ombra_g", lines)
    assert not tv.g_twin_lost(NO_POSE, "ombra_g", tv.g_twin_lost_lines("nessuna riga"))
    assert not tv.g_twin_lost(NO_POSE, "ombra_g", None)


def test_voce_della_history():
    lines = tv.g_twin_lost_lines(LOG)
    rec = tv.g_twin_lost_record(NO_POSE, lines, Path("/x/mola_w4_nominal.log"))
    assert rec["nominal_status"] == NO_POSE and rec["nominal_attempts"] == 2
    assert rec["mola_log"] == "/x/mola_w4_nominal.log" and rec["mola_log_attempt"] == "ultimo"
    assert rec["mola_lines"]["icp_rejected"] == 122 and rec["mola_lines"]["consecutive_bad_last"] == 2
    assert "[full_state] caricato" in rec["cause"]
    json.dumps(rec)
    assert tv.g_twin_lost_record(NO_POSE, None)["mola_lines"] == {}


def test_controllo_opzione(tmp_path):
    ov = _overlay(tmp_path)
    assert tv.check_twin_args(_args(g_overlay=ov, g_twin_lost_ends_run=True)) is None
    assert tv.check_twin_args(_args(g_overlay=ov, g_twin_lost_ends_run=False)) is None
    for ts in ("ombra_e", "clean", "replay"):
        msg = tv.check_twin_args(_args(twin_state=ts, g_twin_lost_ends_run=True))
        assert msg == "--g-twin-lost-ends-run richiede --twin-state ombra_g"
    # args without the attribute (older callers): no new error
    assert tv.check_twin_args(_args(twin_state="ombra_e")) is None


def test_orchestratore_default_spento():
    src = ORCH.read_text()
    m = re.search(r'ap\.add_argument\("--g-twin-lost-ends-run", action="store_true",', src)
    assert m, "opzione assente o non store_true"
    blocco = src[m.start():src.index("ap.add_argument", m.end())]
    assert "default=" not in blocco
    # this branch is under the option and precedes the hold branch, which is unchanged
    i_g = src.index("if args.g_twin_lost_ends_run and tv.g_twin_lost(st, args.twin_state, g_lines):")
    i_hold = src.index('hold = victim_hold_fields("nominale_fallito", victim)')
    assert i_g < i_hold
    ramo = src[i_g:src.index('print("  finestra saltata")', i_g)]
    assert '"reason": "victim_lost_g"' in ramo and "break" in ramo
    assert 'rec["victim_lost_g"] = tv.g_twin_lost_record(' in ramo


@pytest.mark.skipif(not (PILOTA / "logs" / "mola_w4_nominal.log").exists(),
                    reason="run pilota non presente")
def test_pilota_s1001_w4():
    h = json.loads((PILOTA / "history.json").read_text())
    w4 = next(r for r in h if r.get("window") == 4)
    st = w4["status"][len("nominal_failed("):-1]
    lines = tv.g_twin_lost_lines((PILOTA / "logs" / "mola_w4_nominal.log").read_text(errors="replace"))
    assert tv.g_twin_lost(st, "ombra_g", lines)
    assert lines["n_loaded"] == 1 and lines["n_icp_good"] == 0
    # windows driven before w4: with the option the run would have closed here
    assert not any(r.get("victim_held") for r in h if r.get("window", 99) < 4)
