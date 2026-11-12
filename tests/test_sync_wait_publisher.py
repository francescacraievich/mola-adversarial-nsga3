"""Synchronous start with --sync-wait publisher, first-stamp guard and node
--report-matched: fake objects and fake clock, no ROS node.
"""
from types import SimpleNamespace

import pytest

from src.nodes import perturbation_node as pn
from src.optimization import attack_orchestrator as ao


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def spin(self, dt):
        self.t += dt


@pytest.fixture
def sync_mode():
    saved = dict(ao.SYNC_WAIT)
    yield ao.SYNC_WAIT
    ao.SYNC_WAIT.clear()
    ao.SYNC_WAIT.update(saved)


# ----------------------------------------------------------------- pure wait
def test_publisher_needs():
    assert ao.publisher_needs({ao.VICTIM_TOPIC: ["/victim", "/ombra"]}) == {"victim_pub_matched": 2}
    assert ao.publisher_needs({ao.VICTIM_TOPIC: ["/victim"], "/carter/lidar_shadow": ["/ombra"]}) == \
        {"victim_pub_matched": 1, "shadow_pub_matched": 1}


def test_attesa_conteggio_che_sale():
    # base 1 (one foreign reader), rises to 2 at t=1.0 and to 3 at t=2.5
    clk = _Clock()
    read = lambda k: 1 if clk.t < 1.0 else 2 if clk.t < 2.5 else 3
    rec = ao.wait_publisher_matched(read, {"victim_pub_matched": 1}, {"victim_pub_matched": 2},
                                    clk.spin, timeout_s=30.0, margin_s=0.5, clock=clk)
    assert rec["aggancio_s"] == pytest.approx(2.5, abs=0.11)
    assert rec["finale"] == {"victim_pub_matched": 3}
    assert clk.t >= rec["aggancio_s"] + 0.5 - 1e-9      # margin after matching
    assert rec["base"] == {"victim_pub_matched": 1} and rec["margine_s"] == 0.5


def test_attesa_timeout():
    clk = _Clock()
    with pytest.raises(RuntimeError, match="sotto base"):
        ao.wait_publisher_matched(lambda k: 1, {"victim_pub_matched": 0}, {"victim_pub_matched": 2},
                                  clk.spin, timeout_s=3.0, margin_s=0.5, clock=clk)
    assert 3.0 < clk.t < 3.3


def test_attesa_conteggio_assente_va_in_timeout():
    clk = _Clock()
    with pytest.raises(RuntimeError):
        ao.wait_publisher_matched(lambda k: None, {"victim_pub_matched": 0}, {"victim_pub_matched": 2},
                                  clk.spin, timeout_s=1.0, margin_s=0.0, clock=clk)


def test_base_senza_report_matched():
    clk = _Clock()
    node = SimpleNamespace(spin=clk.spin)
    with pytest.raises(RuntimeError, match="--report-matched"):
        ao.matched_base(node, {"victim_pub_matched": 2}, timeout_s=1.0, clock=clk)
    node.victim_pub_matched = 0
    assert ao.matched_base(node, {"victim_pub_matched": 2}, timeout_s=1.0, clock=clk) == {"victim_pub_matched": 0}


# ------------------------------------------------- start with fake node
class _Graph:
    """Subscriptions visible in the orchestrator graph at once."""
    def get_subscriptions_info_by_topic(self, topic):
        return [SimpleNamespace(node_namespace=n) for n in ("/victim", ao.tv.SHADOW_NS)]


class _Node:
    """Fake node: the matched reader count rises by one per spin after
    `delay` spins from spawn (late publisher matching).
    """
    def __init__(self, log, delay):
        self.node, self.log, self.delay = _Graph(), log, delay
        self.victim_pub_matched = 0
        self.spins_after_spawn = None

    def set_victim_stream(self, on):
        self.log.append(("stream", on, self.victim_pub_matched))

    def drain(self):
        pass

    def spin(self, t):
        if self.spins_after_spawn is not None:
            self.spins_after_spawn += 1
            if self.spins_after_spawn > self.delay:
                self.victim_pub_matched = min(2, self.victim_pub_matched + 1)


class _Runner:
    def __init__(self, name, log, node):
        self.name, self.log, self.node = name, log, node
        self.args, self.log_path = [ao.VICTIM_TOPIC], f"{name}.log"

    def spawn(self, tag):
        self.log.append(("spawn", tag))
        if self.node.spins_after_spawn is None:
            self.node.spins_after_spawn = 0

    def alive(self):
        return True

    def wait_ready(self, check=None):
        self.log.append(("ready", self.name))

    def wait_pose(self, node):
        self.log.append(("pose", self.name))


def _deferred(mode, sync_mode, monkeypatch, delay=20):
    monkeypatch.setattr(ao, "SYNC_PUB_MARGIN_S", 0.0)
    sync_mode.update({"mode": mode, "guard": False, "recorder": None})
    log = []
    node = _Node(log, delay)
    v, s = _Runner("victim", log, node), _Runner("shadow", log, node)
    info = ao.VictimRunner.spawn_deferred(v, node, s)
    return info, node


def test_graph_non_aspetta_il_publisher(sync_mode, monkeypatch):
    info, node = _deferred("graph", sync_mode, monkeypatch)
    assert node.victim_pub_matched == 0              # flow can open with 0 matched readers
    assert "sync_wait" not in info
    assert set(info) == {"iscrizioni_s", "timeout_s", "topic", "rimandato"}   # as above


def test_publisher_aspetta_vittima_e_ombra(sync_mode, monkeypatch):
    info, node = _deferred("publisher", sync_mode, monkeypatch)
    assert node.victim_pub_matched == 2
    assert info["sync_wait"]["base"] == {"victim_pub_matched": 0}
    assert info["sync_wait"]["finale"] == {"victim_pub_matched": 2}
    assert node.spins_after_spawn >= 22


def test_publisher_timeout_errore_chiaro(sync_mode, monkeypatch):
    monkeypatch.setattr(ao, "SYNC_SUB_TIMEOUT_S", 0.0)
    with pytest.raises(RuntimeError, match="lettori agganciati"):
        _deferred("publisher", sync_mode, monkeypatch, delay=10 ** 6)


def test_start_with_shadow_apre_dopo_l_aggancio(sync_mode, monkeypatch):
    monkeypatch.setattr(ao, "SYNC_PUB_MARGIN_S", 0.0)
    rec = SimpleNamespace(poses={"victim": [{"stamp_ns": 11_000_000_000}],
                                 "shadow": [{"stamp_ns": 11_000_000_000}]})
    sync_mode.update({"mode": "publisher", "guard": False, "recorder": rec})
    log = []
    node = _Node(log, 5)
    v, s = _Runner("victim", log, node), _Runner("shadow", log, node)
    info = ao.VictimRunner._start_with_shadow(v, node, s)
    assert ("stream", True, 2) in log                # opened with 2 matched readers
    assert info["guardia"]["esito"] == "uguale"


def test_start_with_shadow_guardia_diverso(sync_mode, monkeypatch):
    monkeypatch.setattr(ao, "SYNC_PUB_MARGIN_S", 0.0)
    rec = SimpleNamespace(poses={"victim": [{"stamp_ns": 11_000_000_000}],
                                 "shadow": [{"stamp_ns": 11_900_000_000}]})
    sync_mode.update({"mode": "graph", "guard": True, "recorder": rec})
    log = []
    node = _Node(log, 0)
    v, s = _Runner("victim", log, node), _Runner("shadow", log, node)
    with pytest.raises(ao.ShadowSyncFail, match="11.0.*11.9"):
        ao.VictimRunner._start_with_shadow(v, node, s)
    assert v.sync_info["guardia"]["esito"] == "diverso"


# ----------------------------------------------------------------- guard
def test_guardia_attiva():
    saved = dict(ao.SYNC_WAIT)
    try:
        ao.SYNC_WAIT.update({"mode": "graph", "guard": False})
        assert not ao.sync_guard_active()
        ao.SYNC_WAIT.update({"mode": "publisher"})
        assert ao.sync_guard_active()
        ao.SYNC_WAIT.update({"mode": "graph", "guard": True})
        assert ao.sync_guard_active()
    finally:
        ao.SYNC_WAIT.clear(); ao.SYNC_WAIT.update(saved)


def test_primo_stamp_pose():
    v = [{"stamp_ns": 11_100_000_000}, {"stamp_ns": 11_000_000_000}]
    assert ao.first_stamp_compare(v, [{"stamp_ns": 11_000_000_001}])["esito"] == "uguale"
    g = ao.first_stamp_compare(v, [{"stamp_ns": 11_900_000_000}])
    assert (g["vittima"], g["ombra"], g["esito"]) == (11.0, 11.9, "diverso")
    assert ao.first_stamp_compare(v, []) is None


def test_primo_stamp_printf():
    rv = [{"pathstep": 2, "stamp_s": 11.1}, {"pathstep": 3, "stamp_s": 11.2}]
    rs = [{"pathstep": 2, "stamp_s": 12.0}]
    g = ao.first_printf_compare(rv, rs)
    assert (g["vittima"], g["ombra"], g["esito"], g["fonte"]) == (11.1, 12.0, "diverso", "printf pathStep=2")
    assert ao.first_printf_compare(rv, [{"pathstep": 2, "stamp_s": 11.1}])["esito"] == "uguale"
    assert ao.first_printf_compare([], rs) is None


def test_guardia_ripiego_sui_log(monkeypatch):
    clk = _Clock()
    rec = SimpleNamespace(poses={"victim": [{"stamp_ns": 11_000_000_000}], "shadow": []})
    rows = {"v.log": [{"pathstep": 2, "stamp_s": 11.1}], "s.log": [{"pathstep": 2, "stamp_s": 11.1}]}
    monkeypatch.setattr(ao.shc, "read_printf", lambda p: rows.get(p, []))
    g = ao.sync_guard(SimpleNamespace(spin=clk.spin), SimpleNamespace(log_path="v.log"),
                      SimpleNamespace(log_path="s.log"), recorder=rec, timeout_s=1.0, clock=clk)
    assert g["esito"] == "uguale" and g["fonte"] == "printf pathStep=2"
    rows["s.log"] = []
    g = ao.sync_guard(SimpleNamespace(spin=clk.spin), SimpleNamespace(log_path="v.log"),
                      SimpleNamespace(log_path="s.log"), recorder=rec, timeout_s=1.0, clock=clk)
    assert g["esito"] == "non verificabile"


# ----------------------------------------------------------------- node
def test_registro_crc_colonna_solo_con_opzione(tmp_path):
    a = pn.ShadowCrcLog(tmp_path / "a.csv")
    b = pn.ShadowCrcLog(tmp_path / "b.csv", extra_columns=("matched_victim",))
    a.close(); b.close()
    ha = (tmp_path / "a.csv").read_text().splitlines()[0].split(",")
    hb = (tmp_path / "b.csv").read_text().splitlines()[0].split(",")
    assert tuple(ha) == pn.ShadowCrcLog.COLUMNS
    assert hb == ha + ["matched_victim"]


def test_opzione_report_matched_spenta_di_default():
    import sys
    src = open(pn.__file__, encoding="utf-8").read()
    assert '"--report-matched", action="store_true"' in src
    assert sys.modules[pn.__name__] is pn


def test_cli_orchestratore_default_graph():
    src = open(ao.__file__, encoding="utf-8").read()
    assert '"--sync-wait", choices=["graph", "publisher"], default="graph"' in src
    assert ao.SYNC_WAIT["mode"] == "graph" and ao.SYNC_WAIT["guard"] is False
