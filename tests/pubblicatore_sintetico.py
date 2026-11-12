#!/usr/bin/env python3
"""Synthetic PointCloud2 publisher for testing the probe (test domain only).

Publishes on each --topic a cloud of --byte serialized bytes at --hz for --durata s (node
QoS: reliable, depth 10) and writes one row per message to --registro (topic, stamp, wall
time before and after publish). Refuses to start if ROS_DOMAIN_ID is not the one given
with --dominio.
"""

import argparse
import csv
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", action="append", required=True)
    ap.add_argument("--byte", type=int, default=722864)
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--durata", type=float, default=20.0)
    ap.add_argument("--dominio", default="88")
    ap.add_argument("--attesa-lettori", type=float, default=10.0)
    ap.add_argument("--registro", required=True)
    a = ap.parse_args()
    if os.environ.get("ROS_DOMAIN_ID") != a.dominio:
        sys.exit(f"ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID')!r}: atteso {a.dominio}, non parto")
    import rclpy
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import PointCloud2, PointField

    rclpy.init()
    node = rclpy.create_node("pubblicatore_sintetico")
    pubs = [node.create_publisher(PointCloud2, t, 10) for t in a.topic]
    m = PointCloud2()
    m.header.frame_id = "base_link"
    m.height, m.point_step, m.is_dense = 1, 16, True
    m.fields = [PointField(name=n, offset=4 * i, datatype=7, count=1) for i, n in enumerate("xyzi")]
    m.width, m.row_step, m.data = 1, 16, bytes(16)
    extra = len(serialize_message(m)) - 16
    npunti = (a.byte - extra) // 16
    m.width, m.row_step = npunti, 16 * npunti
    m.data = os.urandom(16 * npunti)
    dim = len(serialize_message(m))
    t_lim = time.monotonic() + a.attesa_lettori
    while any(p.get_subscription_count() == 0 for p in pubs) and time.monotonic() < t_lim:
        rclpy.spin_once(node, timeout_sec=0.1)
    lettori = [p.get_subscription_count() for p in pubs]
    n = int(a.durata * a.hz)
    t0 = time.monotonic()
    with open(a.registro, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("topic", "stamp", "t_pub_inizio", "t_pub_fine", "size_b"))
        for k in range(n):
            st = 100.0 + k / a.hz
            m.header.stamp.sec = int(st)
            m.header.stamp.nanosec = int(round((st - int(st)) * 1e9))
            for t, p in zip(a.topic, pubs):
                t1 = time.time()
                p.publish(m)
                w.writerow((t, f"{st:.9f}", f"{t1:.6f}", f"{time.time():.6f}", dim))
            rclpy.spin_once(node, timeout_sec=0.0)
            pausa = t0 + (k + 1) / a.hz - time.monotonic()
            if pausa > 0:
                time.sleep(pausa)
    print(f"pubblicati {n} per topic, {dim} B, lettori all'avvio {lettori}")
    time.sleep(1.0)
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
