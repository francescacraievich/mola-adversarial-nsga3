#!/usr/bin/env python3
"""Point-cloud probe: one reader per topic with the same QoS as the MOLA instances.

For each message it records source_timestamp and received_timestamp from the
rclpy MessageInfo (two-argument callback), the stamp read from the CDR header
without deserializing the cloud (raw=True), the size in bytes and the wall and
monotonic times on arrival in the callback. One CSV per topic in --out, written
in blocks; sonda_meta.json with environment, QoS, counts and process CPU at
shutdown; the file "pronta" once every topic has at least one matched publisher.

Default QoS matches the MOLA LiDAR entry (MOLA_LIDAR_TOPIC_TYPE in
run_campaign.sh): reliable, keep_last, depth 50, volatile. The process uses the
environment it receives (domain, Fast DDS profile).

Limitation: the probe is a separate process from MOLA; arrival here does not
prove arrival at a MOLA reader.

Usage: python3 scripts/sonda_nuvole.py --out <dir> [--topic T ...] [--durata S]
"""

import argparse
import csv
import json
import os
import resource
import signal
import struct
import sys
import time
from pathlib import Path

TOPIC_DEFAULT = ("/carter/lidar_victim", "/carter/lidar_attacker", "/carter/lidar_perturbed")
COLONNE = ("t_arr_wall", "t_arr_mono", "stamp", "source_ts", "received_ts",
           "dds_ms", "size_b", "pub_seq", "rec_seq")
VARIABILI = ("ROS_DOMAIN_ID", "RMW_IMPLEMENTATION", "FASTRTPS_DEFAULT_PROFILES_FILE",
             "FASTDDS_DEFAULT_PROFILES_FILE", "ROS_AUTOMATIC_DISCOVERY_RANGE",
             "ROS_LOCALHOST_ONLY", "ROS_STATIC_PEERS", "RMW_FASTRTPS_USE_QOS_FROM_XML",
             "FASTDDS_BUILTIN_TRANSPORTS")


def stamp_cdr(buf):
    """Stamp (s) from the header of a serialized CDR message: 4 bytes of
    encapsulation, then int32 sec and uint32 nanosec. None if too short."""
    if len(buf) < 12:
        return None
    end = "<" if buf[1] & 1 else ">"   # 0x0001 CDR_LE, 0x0000 CDR_BE
    sec, nsec = struct.unpack_from(end + "iI", buf, 4)
    return sec + nsec * 1e-9


def nome_file(topic):
    return "sonda" + topic.replace("/", "_") + ".csv"


def _ts(info, chiave):
    v = info.get(chiave) if isinstance(info, dict) else getattr(info, chiave, None)
    return None if v is None else int(v)


class Registro:
    """Rows buffered in memory, written to file every `blocco` rows or on request."""

    def __init__(self, path, blocco):
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(COLONNE)
        self.righe = []
        self.blocco = blocco
        self.n = 0

    def aggiungi(self, riga):
        self.righe.append(riga)
        self.n += 1
        if len(self.righe) >= self.blocco:
            self.scrivi()

    def scrivi(self):
        if self.righe:
            self.w.writerows(self.righe)
            self.righe.clear()
            self.f.flush()

    def chiudi(self):
        self.scrivi()
        self.f.close()


def riga_messaggio(buf, info, t_wall, t_mono):
    src, rec = _ts(info, "source_timestamp"), _ts(info, "received_timestamp")
    st = stamp_cdr(buf)
    return (f"{t_wall:.6f}", f"{t_mono:.6f}",
            "" if st is None else f"{st:.9f}",
            "" if src is None else f"{src * 1e-9:.9f}",
            "" if rec is None else f"{rec * 1e-9:.9f}",
            "" if (src is None or rec is None) else f"{(rec - src) * 1e-6:.3f}",
            len(buf), _ts(info, "publication_sequence_number") or "",
            _ts(info, "reception_sequence_number") or "")


def qos_da_argomenti(a):
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    return QoSProfile(
        depth=a.depth, history=HistoryPolicy.KEEP_LAST,
        reliability=(ReliabilityPolicy.RELIABLE if a.reliability == "reliable"
                     else ReliabilityPolicy.BEST_EFFORT),
        durability=(DurabilityPolicy.VOLATILE if a.durability == "volatile"
                    else DurabilityPolicy.TRANSIENT_LOCAL))


def meta_ambiente():
    cwd = os.getcwd()
    return {"pid": os.getpid(), "argv": sys.argv, "cwd": cwd,
            "xml_in_cwd": os.path.exists(os.path.join(cwd, "DEFAULT_FASTRTPS_PROFILES.xml")),
            "env": {k: os.environ.get(k) for k in VARIABILI}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--topic", action="append", default=None,
                    help="topic da ascoltare (ripetibile); default vittima, attaccante, gemello")
    ap.add_argument("--depth", type=int, default=50)
    ap.add_argument("--reliability", choices=("reliable", "best_effort"), default="reliable")
    ap.add_argument("--durability", choices=("volatile", "transient_local"), default="volatile")
    ap.add_argument("--blocco", type=int, default=50, help="righe per scrittura su file")
    ap.add_argument("--durata", type=float, default=0.0, help="s; 0 = fino a INT/TERM")
    ap.add_argument("--nome-nodo", default="sonda_nuvole")
    a = ap.parse_args(argv)
    topics = a.topic or list(TOPIC_DEFAULT)

    import rclpy
    from rclpy.executors import SingleThreadedExecutor, ExternalShutdownException
    from sensor_msgs.msg import PointCloud2

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"inizio_wall": time.time(), "topic": topics,
            "qos": {"reliability": a.reliability, "history": "keep_last",
                    "depth": a.depth, "durability": a.durability},
            **meta_ambiente()}
    (out / "sonda_meta.json").write_text(json.dumps(meta, indent=1))

    from rclpy.signals import SignalHandlerOptions
    rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node(a.nome_nodo)
    qos = qos_da_argomenti(a)
    reg = {t: Registro(out / nome_file(t), a.blocco) for t in topics}

    def fa_callback(r):
        def cb(buf, info):
            r.aggiungi(riga_messaggio(buf, info, time.time(), time.monotonic()))
        return cb

    subs = [node.create_subscription(PointCloud2, t, fa_callback(reg[t]), qos, raw=True)
            for t in topics]
    pronta = {"fatto": False}

    def ogni_secondo():
        for r in reg.values():
            r.scrivi()
        if not pronta["fatto"] and all(node.count_publishers(t) > 0 for t in topics):
            pronta["fatto"] = True
            (out / "pronta").write_text(f"{time.time():.6f}\n")

    node.create_timer(1.0, ogni_secondo)
    fermo = {"v": False}

    def ferma(signum, frame):
        fermo["v"] = True

    signal.signal(signal.SIGINT, ferma)
    signal.signal(signal.SIGTERM, ferma)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    t0w, t0m = time.time(), time.monotonic()
    try:
        while not fermo["v"] and (a.durata <= 0 or time.monotonic() - t0m < a.durata):
            ex.spin_once(timeout_sec=0.2)
    except ExternalShutdownException:
        pass
    finally:
        for r in reg.values():
            r.chiudi()
        ru = resource.getrusage(resource.RUSAGE_SELF)
        durata = time.monotonic() - t0m
        meta.update({"fine_wall": time.time(), "durata_s": durata,
                     "spin_da_wall": t0w,
                     "messaggi": {t: reg[t].n for t in topics},
                     "pubblicatori_alla_fine": {t: node.count_publishers(t) for t in topics},
                     "cpu_user_s": ru.ru_utime, "cpu_sys_s": ru.ru_stime,
                     "cpu_frazione_core": (ru.ru_utime + ru.ru_stime) / durata if durata > 0 else None,
                     "maxrss_kb": ru.ru_maxrss})
        (out / "sonda_meta.json").write_text(json.dumps(meta, indent=1))
        del subs
        ex.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
