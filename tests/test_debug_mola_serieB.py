"""MOLA DEBUG: with SERIE_B=1 run_campaign.sh exports MOLA_VERBOSITY_MOLA_LO=DEBUG and
MOLA_LO_DEBUG_ICP_QUALITY=1, which reach every MOLA launched by the orchestrator (same
configuration for victim, shadow and twins), also in the none arm. Without SERIE_B nothing
changes. run_campaign.sh actually runs with fake python3, pgrep, ros2 and sleep in PATH; the
fake orchestrator builds the real victim, shadow and twin runners and starts them on a
script that prints its own environment instead of mola-cli."""

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ORIGINALE = ROOT.parent / "originale"
# counter-check: needs a copy without DEBUG (no MOLA_DEBUG in run_campaign.sh);
# ROOT.parent/originale may be a later copy that already has it
CONTROPROVA = next((p for p in (Path(os.environ.get("DEBUG_MOLA_CONTROPROVA", ROOT.parent / "originale")),
                                Path.home() / "mola_lavoro/lavoro_regola5_debug/originale")
                    if (p / "scripts" / "run_campaign.sh").exists()
                    and "MOLA_DEBUG" not in (p / "scripts" / "run_campaign.sh").read_text()), None)
PY = shutil.which("python3")
SLEEP = shutil.which("sleep")
VARS = ("MOLA_VERBOSITY_MOLA_LO", "MOLA_LO_DEBUG_ICP_QUALITY")

TOLTE = ("SERIE_B", "WINDOW_RNG", "SEEDS", "ICP_QUALITY_LOG", "PERC_TERMS", "SAMPLING",
         "RANDOM_FIRST", "TWIN_FILTER", "MOLA_LO_DEBUG_ICP_QUALITY", "MOLA_VERBOSITY_MOLA_LO",
         "MOLA_DEBUG", "CHAMFER_BUDGET", "MAX_DRIFT_LAG_S", "RUN_SUFFIX", "MIN_LATENCY_MS",
         "FAST_OPS", "REPORT_MATCHED", "SYNC_WAIT", "SYNC_GUARD", "NODE_EXTRA", "ORCH_EXTRA",
         "SELECT", "TOPK", "TRANSPORT", "LOCALHOST_ONLY", "ISAAC_ARGS", "CAMPAIGN_DIR",
         "STRUCTURAL", "OPERATOR_MODE", "WARM_FRONT_AFTER_REEVAL", "WINDOW_STOP",
         "GOALS_OVERRIDE", "DANNO_METRO")

# Fake orchestrator: real runners (same environment functions as main) on a script that
# prints the MOLA variables.
ORCH_FINTO = r'''
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.environ["_RADICE"])
from src.optimization import attack_orchestrator as ao
from src.optimization import twin_variants as tv
a = sys.argv[1:]
out = Path(a[a.index("--out") + 1]); out.mkdir(parents=True, exist_ok=True)
script, logs = os.environ["_MOLA_FINTA"], out / "logs"
v = ao.VictimRunner(script, logs, ros_args=ao.VICTIM_ROS_ARGS, env={})
v.extra_env = ao.victim_extra_env({}, {}, ao.VICTIM_ROS_ARGS, shadow_env=True)   # --victim-shadow-env
s = ao.ShadowRunner(script, logs, ros_args='["--ros-args"]', env={}, g_root=str(out / "ovl"))
g = ao.MolaRunner(script, logs, extra_env={"ROS_ARGS": '["--ros-args"]'})
g.extra_env = {**g.extra_env, **tv.g_twin_env(str(out / "st.molastate"), None, root=str(out / "ovl"))}
amb = {}
for tag, r in (("victim", v), ("shadow", s), ("w0_e0", g)):
    r.spawn(tag)
    r.proc.wait(timeout=30)
    r._f.close()
    amb[tag] = dict(l.split("=", 1) for l in r.log_path.read_text().splitlines() if "=" in l)
amb["orch"] = {k: os.environ.get(k, "<assente>") for k in ("MOLA_VERBOSITY_MOLA_LO", "MOLA_LO_DEBUG_ICP_QUALITY")}
with open(os.environ["_REG"], "a") as f:
    f.write(json.dumps(["orch", a, amb]) + "\n")
'''


def _finti(d, reg):
    d.mkdir()
    (d / "orch_finto.py").write_text(ORCH_FINTO)
    mola = d / "mola_finta.sh"
    mola.write_text("#!/bin/bash\n" + "".join(f'echo "{v}=${{{v}-<assente>}}"\n' for v in VARS))
    py = f"""#!/bin/bash
case "$*" in
  *run_isaac_standalone.py*) echo "[standalone] pronto"; exec {SLEEP} 600 ;;
  *perturbation_node.py*) {PY} -c 'import json,sys; print(json.dumps(["nodo"]+sys.argv[1:]))' "$@" >> "{reg}"; exec {SLEEP} 600 ;;
  *attack_orchestrator.py*) [[ "$1" == "-u" ]] && shift; shift; exec {PY} "{d / 'orch_finto.py'}" "$@" ;;
esac
exec {PY} "$@"
"""
    files = {"python3": py,
             "pgrep": '#!/bin/bash\ncase "$*" in *add_intensity*) echo 1; exit 0 ;; esac\nexit 1\n',
             "ros2": '#!/bin/bash\necho "{\\"min_latency_ms\\": ${_LAT_FINTA}}"\n',
             "sleep": "#!/bin/bash\nexit 0\n"}
    for nome, testo in files.items():
        (d / nome).write_text(testo)
    for p in d.iterdir():
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return mola


def campagna(radice, tmp_path, env_extra, arm="nsga3"):
    """run_campaign.sh of `radice` with fake processes: (result, node arguments,
    orchestrator arguments, MOLA environment per role)."""
    tmp_path.mkdir(exist_ok=True)
    tmp_path = tmp_path / f"c{len(list(tmp_path.iterdir()))}"
    tmp_path.mkdir()
    reg = tmp_path / "reg.jsonl"
    mola = _finti(tmp_path / "bin", reg)
    env = {k: v for k, v in os.environ.items() if k not in TOLTE}
    env.update({"PATH": f"{tmp_path / 'bin'}:{env['PATH']}", "CAMPAIGN_DIR": str(tmp_path / "camp"),
                "_REG": str(reg), "_RADICE": str(radice), "_MOLA_FINTA": str(mola)})
    env.update(env_extra)
    env["_LAT_FINTA"] = env.get("MIN_LATENCY_MS") or ("140" if env.get("SERIE_B") == "1" else "85")
    r = subprocess.run(["bash", str(radice / "scripts" / "run_campaign.sh"), "straight", arm, "3", "1", "4", "2"],
                       capture_output=True, text=True, env=env, cwd=tmp_path, timeout=120)
    righe = [json.loads(x) for x in reg.read_text().splitlines()] if reg.exists() else []
    nodo = next((x[1:] for x in righe if x[0] == "nodo"), None)
    orch = next((x for x in righe if x[0] == "orch"), None)
    return r, nodo, (orch[1] if orch else None), (orch[2] if orch else None)


@pytest.mark.parametrize("arm", ["nsga3", "random", "none"])
def test_serie_b_debug_in_tutte_le_mola(tmp_path, arm):
    r, _, orch, amb = campagna(ROOT, tmp_path, {"SERIE_B": "1", "SEEDS": "1001"}, arm)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert set(amb) == {"victim", "shadow", "w0_e0", "orch"}
    for ruolo, e in amb.items():
        assert e["MOLA_VERBOSITY_MOLA_LO"] == "DEBUG", ruolo
        assert e["MOLA_LO_DEBUG_ICP_QUALITY"] == "1", ruolo
    # victim and shadow with the same MOLA environment; no --victim-mola-env
    assert amb["victim"] == amb["shadow"] == amb["w0_e0"]
    assert "--victim-mola-env" not in orch


def test_serie_b_debug_spegnibile_e_valore_esterno(tmp_path):
    r, _, _, amb = campagna(ROOT, tmp_path, {"SERIE_B": "1", "SEEDS": "1001", "MOLA_DEBUG": "0"})
    assert r.returncode == 0, r.stdout[-3000:]
    # the SERIE_B ICP printf (ICP_QUALITY_LOG=1) remains, without DEBUG
    assert {e["MOLA_VERBOSITY_MOLA_LO"] for e in amb.values()} == {"<assente>"}
    assert {e["MOLA_LO_DEBUG_ICP_QUALITY"] for e in amb.values()} == {"1"}
    env = {"SERIE_B": "1", "SEEDS": "1001", "MOLA_VERBOSITY_MOLA_LO": "INFO"}
    r, _, _, amb = campagna(ROOT, tmp_path, env)
    assert "MOLA_VERBOSITY_MOLA_LO era 'INFO', portata a DEBUG" in r.stdout
    assert {e["MOLA_VERBOSITY_MOLA_LO"] for e in amb.values()} == {"DEBUG"}
    r, _, _, _ = campagna(ROOT, tmp_path, {"SERIE_B": "1", "SEEDS": "1001", "MOLA_DEBUG": "2"})
    assert r.returncode == 1 and "MOLA_DEBUG sconosciuto" in r.stdout


@pytest.mark.parametrize("env", [{}, {"SERIE_B": "0"}, {"ICP_QUALITY_LOG": "1"}])
def test_senza_serie_b_nessun_debug(tmp_path, env):
    r, _, _, amb = campagna(ROOT, tmp_path, dict(env))
    assert r.returncode == 0, r.stdout[-3000:]
    assert {e["MOLA_VERBOSITY_MOLA_LO"] for e in amb.values()} == {"<assente>"}


@pytest.mark.skipif(not (ORIGINALE / "scripts" / "run_campaign.sh").exists(), reason="copia originale assente")
@pytest.mark.parametrize("arm", ["nsga3", "none"])
@pytest.mark.parametrize("env", [{}, {"SERIE_B": "0"}, {"ICP_QUALITY_LOG": "1", "SEEDS": "1001"}])
def test_senza_serie_b_identico_all_originale(tmp_path, arm, env):
    nuovo = campagna(ROOT, tmp_path / "nuovo", dict(env), arm)
    vecchio = campagna(ORIGINALE, tmp_path / "vecchio", dict(env), arm)
    assert nuovo[0].returncode == vecchio[0].returncode == 0
    # node and orchestrator arguments and MOLA environment equal, apart from the folder
    for i in (1, 2, 3):
        assert (json.dumps(nuovo[i]).replace(str(tmp_path / "nuovo"), "T")
                == json.dumps(vecchio[i]).replace(str(tmp_path / "vecchio"), "T"))


@pytest.mark.skipif(CONTROPROVA is None, reason="nessuna copia senza MOLA_DEBUG per la controprova")
def test_controprova_originale_senza_debug(tmp_path):
    # the check discriminates: without DEBUG export, with SERIE_B=1, only the ICP printf
    r, _, _, amb = campagna(CONTROPROVA, tmp_path, {"SERIE_B": "1", "SEEDS": "1001"})
    assert r.returncode == 0, r.stdout[-3000:]
    assert {e["MOLA_VERBOSITY_MOLA_LO"] for e in amb.values()} == {"<assente>"}
    assert {e["MOLA_LO_DEBUG_ICP_QUALITY"] for e in amb.values()} == {"1"}
