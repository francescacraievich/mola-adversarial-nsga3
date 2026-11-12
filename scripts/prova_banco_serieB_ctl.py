#!/usr/bin/env python3
"""
Helper commands for the bench scripts (prova_banco_serieB_*.sh).

Subcommands:
    isaac play|pause|pose               command to the Isaac server (file /tmp/isaac_cmd.json,
                                        same protocol as the orchestrator's IsaacClient)
    isaac set_pose X Y YAW_DEG          pause, set pose, play (as --start-pose of the orchestrator)
    status [--latency MS] [--timeout S] wait for the perturbation_node status and print the main
                                        fields; with --latency exit 1 if min_latency_ms differs
    stream on|off                       open or close the victim stream with the victim attack
                                        off; wait for confirmation in the status
    poses CSV                           record the /victim and /attacker poses (lidar_odometry/pose)
                                        with stamp and receive time, until SIGINT

No subcommand starts chain processes.
"""

import argparse
import csv
import json
import os
import signal
import sys
import time

CMD_FILE = "/tmp/isaac_cmd.json"
REPLY_FILE = "/tmp/isaac_reply.json"


def isaac_call(cmd, timeout=60.0, **kw):
    cid = int(time.time() * 1000) % 1_000_000_000
    if os.path.exists(REPLY_FILE):
        os.remove(REPLY_FILE)
    with open(CMD_FILE, "w") as f:
        json.dump({"cmd": cmd, "id": cid, **kw}, f)
    t_end = time.time() + timeout
    while time.time() < t_end:
        if os.path.exists(REPLY_FILE):
            try:
                with open(REPLY_FILE) as f:
                    r = json.load(f)
            except (OSError, ValueError):
                time.sleep(0.02)
                continue
            if r.get("id") == cid:
                if not r.get("ok"):
                    raise RuntimeError(f"Isaac '{cmd}': {r.get('error')}")
                return r
        time.sleep(0.02)
    raise TimeoutError(f"Isaac non risponde a '{cmd}' da {timeout:.0f} s")


def cmd_isaac(a):
    import math
    if a.op == "set_pose":
        isaac_call("pause")
        r = isaac_call("set_pose", x=float(a.args[0]), y=float(a.args[1]), yaw=math.radians(float(a.args[2])))
        isaac_call("play")
    else:
        r = isaac_call(a.op)
    print(json.dumps({k: v for k, v in r.items() if k != "id"}))
    return 0


class _Ros:
    def __init__(self, name):
        import rclpy
        from rclpy.node import Node
        rclpy.init()
        self.rclpy = rclpy
        self.node = Node(name)
        self.status = None
        from std_msgs.msg import String
        self.node.create_subscription(String, "/attack/status", self._st, 10)

    def _st(self, m):
        try:
            self.status = json.loads(m.data)
            self.status["_t"] = time.time()
        except ValueError:
            pass

    def spin(self, sec):
        t_end = time.time() + sec
        while time.time() < t_end:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)

    def wait_status(self, timeout):
        t_end = time.time() + timeout
        while time.time() < t_end:
            self.spin(0.2)
            if self.status is not None and time.time() - self.status["_t"] < 1.0:
                return self.status
        return None

    def close(self):
        self.node.destroy_node()
        self.rclpy.shutdown()


def cmd_status(a):
    r = _Ros("prova_banco_status")
    st = r.wait_status(a.timeout)
    r.close()
    if st is None:
        print("nessuno status del perturbation_node")
        return 1
    keys = ("enabled", "min_latency_ms", "frames_in", "frames_out", "proc_ms_mean", "latency_ms_mean",
            "latency_ms_max", "victim_stream", "victim_attack", "victim_frames")
    print(json.dumps({k: st.get(k) for k in keys}))
    if a.latency is not None and abs(float(st.get("min_latency_ms", -1)) - a.latency) > 1e-6:
        print(f"latenza minima {st.get('min_latency_ms')} ms invece di {a.latency} ms")
        return 1
    return 0


def cmd_stream(a):
    from std_msgs.msg import Bool
    on = a.op == "on"
    r = _Ros("prova_banco_stream")
    p_att = r.node.create_publisher(Bool, "/attack/victim_attack", 10)
    p_str = r.node.create_publisher(Bool, "/attack/victim_stream", 10)
    ok = False
    t_end = time.time() + a.timeout
    while time.time() < t_end:
        p_att.publish(Bool(data=False))
        p_str.publish(Bool(data=on))
        r.spin(0.5)
        st = r.status
        if st and time.time() - st["_t"] < 1.0 and st.get("victim_stream") is on and not st.get("victim_attack"):
            ok = True
            break
    print(json.dumps({"stream": a.op, "confermato": ok, "t_wall": time.time(), "t_mono": time.monotonic()}))
    r.close()
    return 0 if ok else 1


def cmd_poses(a):
    from nav_msgs.msg import Odometry
    r = _Ros("prova_banco_poses")
    f = open(a.csv, "w", newline="")
    w = csv.writer(f)
    w.writerow(["ns", "stamp", "t_recv_wall", "t_recv_mono", "x", "y"])

    def cb(ns):
        def _cb(m):
            w.writerow([ns, repr(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9), repr(time.time()),
                        repr(time.monotonic()), repr(m.pose.pose.position.x), repr(m.pose.pose.position.y)])
        return _cb
    for ns in ("victim", "attacker"):
        r.node.create_subscription(Odometry, f"/{ns}/lidar_odometry/pose", cb(ns), 100)
    stop = []
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    t_flush = time.monotonic()
    while not stop:
        r.rclpy.spin_once(r.node, timeout_sec=0.1)
        if time.monotonic() - t_flush >= 0.5:      # flush to disk at least once per second
            f.flush()
            t_flush = time.monotonic()
    f.close()
    r.close()
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    s = sub.add_parser("isaac"); s.add_argument("op", choices=["play", "pause", "pose", "set_pose"])
    s.add_argument("args", nargs="*")
    s = sub.add_parser("status"); s.add_argument("--latency", type=float, default=None)
    s.add_argument("--timeout", type=float, default=60.0)
    s = sub.add_parser("stream"); s.add_argument("op", choices=["on", "off"])
    s.add_argument("--timeout", type=float, default=10.0)
    s = sub.add_parser("poses"); s.add_argument("csv")
    a = ap.parse_args()
    return {"isaac": cmd_isaac, "status": cmd_status, "stream": cmd_stream, "poses": cmd_poses}[a.what](a)


if __name__ == "__main__":
    sys.exit(main())
