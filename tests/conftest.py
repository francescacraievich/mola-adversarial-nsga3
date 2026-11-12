"""ROS domain of the suite: every node created by the tests, including subprocesses that
inherit os.environ (zone_nodo.py, versioneI_nodo.py, fine_run_nodo.py), runs in domain 88,
away from domain 0 used by bench runs. Set at conftest import, before test modules are
collected, and re-checked at every session. Other ROS_* variables (discovery range
included) are left as in the environment: some tests check them themselves."""
import os

import pytest

DOMINIO_TEST = "88"


def imponi_dominio_ros():
    os.environ["ROS_DOMAIN_ID"] = DOMINIO_TEST


imponi_dominio_ros()


def pytest_configure(config):
    imponi_dominio_ros()


@pytest.fixture(autouse=True, scope="session")
def dominio_ros_dei_test():
    imponi_dominio_ros()
    yield DOMINIO_TEST
    assert os.environ.get("ROS_DOMAIN_ID") == DOMINIO_TEST, "un test ha cambiato ROS_DOMAIN_ID della suite"
