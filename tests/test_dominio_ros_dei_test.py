"""The suite runs in domain 88 (tests/conftest.py, which leaves other ROS_* variables alone)
and the real nodes used by the tests refuse to start outside that domain. No ROS node is
created: the nodes exit before importing rclpy."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

QUI = Path(__file__).resolve().parent


def test_ambiente_della_suite(dominio_ros_dei_test):
    assert dominio_ros_dei_test == "88"
    assert os.environ["ROS_DOMAIN_ID"] == "88"


def test_conftest_non_tocca_le_altre_variabili_ros():
    src = (QUI / "conftest.py").read_text()
    for v in ("DISCOVERY_RANGE", "ROS_STATIC_PEERS", "ROS_LOCALHOST_ONLY", "pop("):
        assert v not in src, v


def test_sottoprocesso_eredita_il_dominio():
    r = subprocess.run([sys.executable, "-c",
                        "import os; print(os.environ.get('ROS_DOMAIN_ID'))"],
                       capture_output=True, text=True, timeout=30)
    assert r.stdout.split() == ["88"]


@pytest.mark.parametrize("nodo", ["zone_nodo.py", "versioneI_nodo.py", "fine_run_nodo.py"])
@pytest.mark.parametrize("dominio", [None, "0", "87"])
def test_nodi_dei_test_rifiutano_altri_domini(nodo, dominio):
    env = {k: v for k, v in os.environ.items() if k != "ROS_DOMAIN_ID"}
    if dominio is not None:
        env["ROS_DOMAIN_ID"] = dominio
    # no arguments: if the check were missing, the node would fail at parsing, before ROS
    r = subprocess.run([sys.executable, str(QUI / nodo)], capture_output=True, text=True, env=env, timeout=30)
    assert r.returncode != 0
    assert "i nodi dei test girano solo nel dominio 88, non parto" in r.stderr
