#!/usr/bin/env python3
"""Point cloud after the end of the run, with the real perturbation node and the
orchestrator's AttackNode in the same process (separate executors, DDS on the
environment's ROS_DOMAIN_ID, 88 in tests). No Isaac or MOLA: clouds are synthetic and
published on the node's input topic.

Sequence, as at the end of a run: twin rollout with genome and attack on, victim segment
with its genome, victim finish() (attack and victim stream off). Then a late cloud with
the state left by finish(), spegni_attacco_fine_run() and another late cloud.

Usage: fine_run_nodo.py <root> <folder> <out.json> -- <node arguments>
"""
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

import os  # noqa: E402
if os.environ.get("ROS_DOMAIN_ID") != "88":
    # real node: outside domain 88 it would reach the bench topics (domain 0)
    sys.exit(f"ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID')!r}: i nodi dei test girano solo nel dominio 88, non parto")

sep = sys.argv.index("--")
radice, cartella, uscita = sys.argv[1:sep]
node_args = sys.argv[sep + 1:]
sys.path.insert(0, radice)
cartella = Path(cartella)

import src.optimization.attack_orchestrator as ao  # noqa: E402
from src.nodes import perturbation_node as pn  # noqa: E402
from src.perturbations.perturbation_generator import PerturbationGenerator  # noqa: E402
from sensor_msgs.msg import PointCloud2  # noqa: E402

assert ao.__file__.startswith(radice), ao.__file__


def nuvola(seed, n=20000):
    """Floor and two walls, constant intensity."""
    rng = np.random.default_rng(seed)
    k = n // 3
    pav = np.c_[rng.uniform(-10, 10, k), rng.uniform(-10, 10, k), np.full(k, -0.4)]
    p1 = np.c_[np.full(k, 8.0), rng.uniform(-10, 10, k), rng.uniform(-0.4, 3, k)]
    p2 = np.c_[rng.uniform(-10, 10, n - 2 * k), np.full(n - 2 * k, 6.0), rng.uniform(-0.4, 3, n - 2 * k)]
    xyz = np.vstack([pav, p1, p2])
    return np.c_[xyz, np.full(len(xyz), 100.0)].astype(np.float32)


def messaggio(data, stamp):
    m = PointCloud2()
    m.header.frame_id = "base_link"
    m.header.stamp.sec = int(stamp)
    m.header.stamp.nanosec = int(round((stamp - int(stamp)) * 1e9))
    m.height, m.width = 1, len(data)
    m.fields, m.point_step = pn.FIELDS, pn.POINT_STEP
    m.row_step, m.is_bigendian, m.is_dense = pn.POINT_STEP * len(data), False, True
    m.data = data.tobytes()
    return m


def stamp_di(msg):
    return round(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, 3)


class _Cli(Exception):
    pass


def _cattura(cli):
    raise _Cli(cli)


# node arguments parsed by its own main, as from the command line
_orig = pn.PerturbationNode
pn.PerturbationNode = _cattura
sys.argv = ["perturbation_node.py"] + node_args
_init = pn.rclpy.init
pn.rclpy.init = lambda **k: None
try:
    pn.main()
    raise SystemExit("main non ha creato il nodo")
except _Cli as e:
    cli = e.args[0]
pn.PerturbationNode, pn.rclpy.init = _orig, _init

res = {"passi": []}
pnode = pex = th = None
try:
    node = ao.AttackNode(ao.FollowerModel())      # calls rclpy.init
    pnode = _orig(cli)
    from rclpy.executors import MultiThreadedExecutor
    pex = MultiThreadedExecutor(num_threads=4)
    pex.add_node(pnode)
    th = threading.Thread(target=pex.spin, daemon=True)
    th.start()
    uscite = {"main": {}, "victim": {}}
    node.node.create_subscription(PointCloud2, "/carter/lidar_perturbed",
                                  lambda m: uscite["main"].__setitem__(stamp_di(m), bytes(m.data)), 10)
    node.node.create_subscription(PointCloud2, "/carter/lidar_victim",
                                  lambda m: uscite["victim"].__setitem__(stamp_di(m), bytes(m.data)), 10)
    pub = node.node.create_publisher(PointCloud2, "/carter/lidar_with_intensity", 10)
    t0 = time.time()
    while pub.get_subscription_count() == 0 and time.time() - t0 < 30:
        node.spin(0.1)
    # status starts from the first processed cloud
    t0 = time.time()
    while node.node_enabled is None and time.time() - t0 < 30:
        pub.publish(messaggio(nuvola(0), 9.0))
        node.spin(0.5)
    if node.node_enabled is None:
        raise SystemExit("status del nodo assente")
    gen = PerturbationGenerator()
    rng = np.random.default_rng(7)
    g_gemello = rng.uniform(0.3, 0.9, gen.get_genome_size())
    g_vittima = rng.uniform(0.3, 0.9, gen.get_genome_size())

    def manda(nome, stamp, seed):
        data = nuvola(seed)
        pub.publish(messaggio(data, stamp))
        t = time.time()
        while time.time() - t < 10 and stamp not in uscite["main"] and stamp not in uscite["victim"]:
            node.spin(0.05)
        node.spin(0.5)      # possible output on the other topic
        rec = {"passo": nome, "stamp": stamp, "enabled": node.node_enabled,
               "victim_stream": node.victim_stream, "victim_attack": node.victim_attack,
               "main": None if stamp not in uscite["main"] else uscite["main"][stamp] == data.tobytes(),
               "victim": None if stamp not in uscite["victim"] else uscite["victim"][stamp] == data.tobytes()}
        res["passi"].append(rec)
        return rec

    # twin rollout
    node.set_genome(g_gemello)
    res["conferma_gemello_acceso"] = node.set_attack_enabled(True)
    manda("gemello", 10.0, 1)
    # victim segment (attack on from the first segment, as --attack-from spawn)
    res["conferma_genoma_vittima"] = node.set_victim_genome(g_vittima)
    node.set_victim_attack(True)
    node.set_victim_stream(True)
    manda("vittima", 10.1, 2)
    # victim finish() (attack_orchestrator.py, _rollout_victim)
    node.set_victim_attack(False)
    node.set_victim_stream(False)
    manda("dopo_finish", 10.2, 3)
    # end of run
    res["conferma_spento"] = ao.spegni_attacco_fine_run(node)
    manda("dopo_spegnimento", 10.3, 4)
finally:
    if pnode is not None:
        # stop the node executor and join its thread before shutting down rclpy,
        # otherwise interpreter exit can abort (code -6)
        if pex is not None:
            pex.shutdown(timeout_sec=5.0)
            th.join(timeout=5.0)
        for log in (pnode._frame_log, pnode._struct_log):
            if log is not None:
                log.close()
        pnode.destroy_node()
    Path(uscita).write_text(json.dumps(res, indent=1))
    try:
        node.close()
    except Exception:
        pass
