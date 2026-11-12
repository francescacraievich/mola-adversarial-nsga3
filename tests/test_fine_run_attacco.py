#!/usr/bin/env python3
"""
Twin attack turned off at the end of the run (spegni_attacco_fine_run in the finally of
main). finish() of the victim closes the victim stream and attack but leaves
/attack/enabled on with the last twin genome: without this step a late cloud goes through
the twin branch and comes out perturbed on the main topic.

    ROS_DOMAIN_ID=88 python3 -m pytest tests/test_fine_run_attacco.py -q
"""

import ast
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
QUI = Path(__file__).resolve().parent
ORCH = ROOT / "src/optimization/attack_orchestrator.py"
sys.path.insert(0, str(ROOT))


def _chiamata(nodo, nome):
    return (isinstance(nodo, ast.Call) and isinstance(nodo.func, (ast.Name, ast.Attribute))
            and (getattr(nodo.func, "id", None) == nome or getattr(nodo.func, "attr", None) == nome))


def _main_finally():
    albero = ast.parse(ORCH.read_text())
    main = next(n for n in albero.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    prove = [n for n in ast.walk(main) if isinstance(n, ast.Try)
             and any(_chiamata(c, "spegni_attacco_fine_run") for s in n.finalbody for c in ast.walk(s))]
    assert len(prove) == 1
    return albero, prove[0]


def test_spegnimento_solo_nel_finally_prima_di_close_e_play():
    """A single call, in the finally of main: after the last window and before node.close()
    (the orchestrator node is needed for the acknowledgement) and isaac.play() (which
    restarts the clouds)."""
    albero, prova = _main_finally()
    tutte = [n for n in ast.walk(albero) if _chiamata(n, "spegni_attacco_fine_run")]
    assert len(tutte) == 1
    fb = prova.finalbody

    def primo(nome):
        return next(i for i, s in enumerate(fb) if any(_chiamata(c, nome) for c in ast.walk(s)))

    i_sp = primo("spegni_attacco_fine_run")
    assert i_sp == 1 and ast.unparse(fb[0]) == "node.watch = None"
    assert i_sp < primo("close") < primo("play")
    # no new call in the try body (the windows)
    assert not any(_chiamata(c, "spegni_attacco_fine_run") for s in prova.body for c in ast.walk(s))


def test_chiamate_durante_la_run_invariate():
    """set_attack_enabled returns the acknowledgement outcome: rollout calls discard it
    (plain statements)."""
    albero = ast.parse(ORCH.read_text())
    usi = []
    for n in ast.walk(albero):
        for c in ast.iter_child_nodes(n):
            if _chiamata(c, "set_attack_enabled"):
                usi.append((type(n).__name__, ast.unparse(c)))
    rollout = [u for u in usi if u[1] != "node.set_attack_enabled(False)"]
    assert len(rollout) == 3 and all(t == "Expr" for t, _ in rollout)
    # the only new call: inside bool(...) in spegni_attacco_fine_run
    assert usi.count(("Call", "node.set_attack_enabled(False)")) == 1 and len(usi) == 4


def test_spegni_attacco_fine_run_esiti(capsys):
    ao = pytest.importorskip("src.optimization.attack_orchestrator")
    chiamate = []
    n = SimpleNamespace(set_attack_enabled=lambda on: chiamate.append(on) or True)
    assert ao.spegni_attacco_fine_run(n) is True and chiamate == [False]
    n.set_attack_enabled = lambda on: False
    assert ao.spegni_attacco_fine_run(n) is False

    def guasto(on):
        raise RuntimeError("contesto chiuso")
    n.set_attack_enabled = guasto
    assert ao.spegni_attacco_fine_run(n) is False
    assert "contesto chiuso" in capsys.readouterr().out


def test_set_attack_enabled_restituisce_la_conferma():
    ao = pytest.importorskip("src.optimization.attack_orchestrator")
    pytest.importorskip("std_msgs.msg")
    for esito in (True, False):
        n = SimpleNamespace(_enable_pub=object(), node_enabled=None,
                            _publish_confirmed=lambda pub, m, ok, what, esito=esito: esito)
        assert ao.AttackNode.set_attack_enabled(n, False) is esito


@pytest.mark.parametrize("modo", ["insieme", "random"])
def test_nuvola_dopo_fine_run_non_perturbata(tmp_path, modo):
    """Real node and AttackNode (ROS_DOMAIN_ID=88): after victim finish() the cloud comes out
    perturbed on the main topic ("gemello" row in the zone log); after
    spegni_attacco_fine_run it comes out identical bit for bit, with no log row."""
    pytest.importorskip("rclpy")
    esito = tmp_path / "esito.json"
    args = ["--frame-log", str(tmp_path / "frame_log.csv")]
    if modo == "insieme":
        args += ["--operator-mode", "insieme", "--structural", "gross", "--victim-never-clean",
                 "--struct-log", str(tmp_path / "struct_log.csv")]
    env = {**os.environ, "ROS_DOMAIN_ID": "88"}
    r = subprocess.run([sys.executable, str(QUI / "fine_run_nodo.py"), str(ROOT), str(tmp_path),
                        str(esito), "--"] + args, env=env, capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    d = json.loads(esito.read_text())
    p = {x["passo"]: x for x in d["passi"]}
    assert d["conferma_gemello_acceso"] and d["conferma_genoma_vittima"] and d["conferma_spento"] is True
    # during the run: twin perturbed, victim perturbed and main topic clean
    assert p["gemello"]["main"] is False
    assert p["vittima"]["victim"] is False and p["vittima"]["main"] is True
    # state left by finish(): twin attack still on
    assert p["dopo_finish"]["enabled"] is True and p["dopo_finish"]["main"] is False
    # after the end-of-run shutdown
    q = p["dopo_spegnimento"]
    assert q["enabled"] is False and q["victim_stream"] is False and q["victim_attack"] is False
    assert q["main"] is True and q["victim"] is None
    with open(tmp_path / "frame_log.csv") as f:
        fl = {round(float(x["stamp_in"]), 3): x for x in csv.DictReader(f)}
    assert fl[10.2]["enabled"] == "1" and fl[10.3]["enabled"] == "0"
    if modo == "insieme":
        with open(tmp_path / "struct_log.csv") as f:
            st = [(x["stream"], round(float(x["stamp"]), 3)) for x in csv.DictReader(f)]
        assert ("gemello", 10.2) in st
        assert not any(s == 10.3 for _, s in st)
