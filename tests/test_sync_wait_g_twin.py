"""--sync-wait publisher for the variant-G twin: readers matched to the node main publisher
(main_pub_matched) before play. Fake objects, no ROS node.
"""
import pytest

from src.nodes import perturbation_node as pn
from src.optimization import attack_orchestrator as ao


@pytest.fixture
def sync_mode():
    saved = dict(ao.SYNC_WAIT)
    yield ao.SYNC_WAIT
    ao.SYNC_WAIT.clear()
    ao.SYNC_WAIT.update(saved)


class _Node:
    """The main publisher count goes up by one after `delay` spins."""
    def __init__(self, base=1, delay=5, rises=True):
        self.main_pub_matched = base
        self.delay, self.rises, self.spins = delay, rises, 0

    def drain(self):
        pass

    def spin(self, t):
        self.spins += 1
        if self.rises and self.spins == self.delay:
            self.main_pub_matched += 1


class _Runner:
    def __init__(self, alive=True):
        self._alive, self.log_path = alive, "mola_w2_e2.log"

    def alive(self):
        return self._alive


def test_bisogno_solo_con_g_e_publisher(sync_mode):
    sync_mode["mode"] = "graph"
    assert ao.g_publisher_need(True) is None
    sync_mode["mode"] = "publisher"
    assert ao.g_publisher_need(True) == {"main_pub_matched": 1}
    assert ao.g_publisher_need(False) is None


def test_default_invariato():
    assert ao.SYNC_WAIT["mode"] == "graph"
    assert ao.g_publisher_need(True) is None


def test_attesa_aggancio_del_gemello(monkeypatch):
    monkeypatch.setattr(ao, "SYNC_PUB_MARGIN_S", 0.0)
    node = _Node(base=1, delay=5)
    need = ao.g_publisher_need(True, mode="publisher")
    base = ao.matched_base(node, need)
    assert base == {"main_pub_matched": 1}
    rec = ao.sync_wait_publisher(node, need, base, (_Runner(),), None)
    assert rec["finale"] == {"main_pub_matched": 2}
    assert rec["base"] == {"main_pub_matched": 1} and rec["attesi_in_piu"] == {"main_pub_matched": 1}


def test_timeout_errore_chiaro(monkeypatch):
    monkeypatch.setattr(ao, "SYNC_SUB_TIMEOUT_S", 0.2)
    monkeypatch.setattr(ao, "SYNC_PUB_MARGIN_S", 0.0)
    node = _Node(base=1, rises=False)
    node.spin = lambda t: __import__("time").sleep(0.02)
    need = ao.g_publisher_need(True, mode="publisher")
    with pytest.raises(RuntimeError, match="main_pub_matched"):
        ao.sync_wait_publisher(node, need, {"main_pub_matched": 1}, (_Runner(),), None)


def test_mola_morta_prima_dell_aggancio():
    node = _Node(base=1, rises=False)
    need = ao.g_publisher_need(True, mode="publisher")
    with pytest.raises(RuntimeError, match="mola-cli terminato"):
        ao.sync_wait_publisher(node, need, {"main_pub_matched": 1}, (_Runner(alive=False),), None)


def test_base_assente_senza_report_matched():
    class _Vuoto(_Node):
        pass
    node = _Vuoto()
    del node.main_pub_matched
    t = [0.0]
    node.spin = lambda dt: t.__setitem__(0, t[0] + dt)
    with pytest.raises(RuntimeError, match="--report-matched"):
        ao.matched_base(node, {"main_pub_matched": 1}, timeout_s=1.0, clock=lambda: t[0])


def test_status_e_attesa_prima_del_play():
    src_n = open(pn.__file__, encoding="utf-8").read()
    i = src_n.index('if getattr(self, "_report_matched", False):\n            st["victim_pub_matched"]')
    assert src_n.index('st["main_pub_matched"] = self.pub.get_subscription_count()') > i
    src_o = open(ao.__file__, encoding="utf-8").read()
    assert 'self.main_pub_matched = d.get("main_pub_matched")' in src_o
    w = src_o.index("pub = sync_wait_publisher(node, g_pub_need, g_pub_base, (mola,), t_spawn)")
    s = src_o.rindex("mola.spawn(tag)", 0, w)
    assert src_o.rindex("g_pub_base = matched_base(node, g_pub_need)", 0, s) < s
    assert w < src_o.index("isaac.play()", w)
