"""Synchronous victim-shadow start (--shadow-check): fake objects, no ROS node."""
from types import SimpleNamespace

import pytest

from src.optimization import attack_orchestrator as ao
from src.optimization import shadow_check as shc


def test_missing_subscribers():
    infos = [SimpleNamespace(node_namespace="/victim"), SimpleNamespace(node_namespace="/")]
    assert shc.missing_subscribers(infos, ["/victim", "/ombra"]) == ["/ombra"]
    assert shc.missing_subscribers([], ["/victim"]) == ["/victim"]


class _Graph:
    """Subscribers visible after n queries."""

    def __init__(self, after, ns):
        self.calls, self.after, self.ns = 0, after, ns

    def get_subscriptions_info_by_topic(self, topic):
        self.calls += 1
        return [SimpleNamespace(node_namespace=n) for n in self.ns] if self.calls > self.after else []


class _Node:
    def __init__(self, graph, log):
        self.node, self.log = graph, log

    def set_victim_stream(self, on):
        self.log.append(("stream", on))

    def drain(self):
        pass

    def spin(self, t):
        pass


class _Runner:
    def __init__(self, name, log, topic=ao.VICTIM_TOPIC):
        self.name, self.log, self.args, self.log_path = name, log, [topic], f"{name}.log"

    def spawn(self, tag):
        self.log.append(("spawn", tag))

    def alive(self):
        return True

    def wait_ready(self, check=None):
        self.log.append(("ready", self.name))

    def wait_pose(self, node):
        self.log.append(("pose", self.name))


def test_avvio_sincrono_ordine():
    log = []
    node = _Node(_Graph(3, ["/victim", "/ombra"]), log)
    v, s = _Runner("victim", log), _Runner("shadow", log)
    info = ao.VictimRunner._start_with_shadow(v, node, s)
    assert log[:3] == [("stream", False), ("spawn", "victim"), ("spawn", "shadow")]
    assert log[3] == ("stream", True)          # opened only after the subscriptions
    assert log[4:] == [("ready", "victim"), ("ready", "shadow"), ("pose", "shadow")]
    assert node.node.calls == 4 and info["topic"] == {ao.VICTIM_TOPIC: ["/victim", "/ombra"]}


def test_avvio_sincrono_timeout(monkeypatch):
    monkeypatch.setattr(ao, "SYNC_SUB_TIMEOUT_S", 0.0)
    log = []
    node = _Node(_Graph(10 ** 6, ["/victim"]), log)
    with pytest.raises(RuntimeError, match="iscrizioni mancanti"):
        ao.VictimRunner._start_with_shadow(_Runner("victim", log), node, _Runner("shadow", log))
    assert ("stream", True) not in log


def test_avvio_sincrono_topic_separato():
    log = []
    g = _Graph(0, ["/victim", "/ombra"])
    info = ao.VictimRunner._start_with_shadow(_Runner("victim", log), _Node(g, log),
                                              _Runner("shadow", log, topic="/carter/lidar_shadow"))
    assert info["topic"] == {ao.VICTIM_TOPIC: ["/victim"], "/carter/lidar_shadow": ["/ombra"]}
