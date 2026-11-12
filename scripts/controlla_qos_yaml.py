#!/usr/bin/env python3
"""
Offline check of the LiDAR QoS injected through MOLA_LIDAR_TOPIC_TYPE.

Expands config/mola/lidar_odometry_ros2_local.yaml with mola-yaml-parser (same
loading as mola-cli, without ROS or DDS) twice: with the current value of
MOLA_LIDAR_TOPIC_TYPE and without it. Exits 0 if the LiDAR entry for the given
topic has the expected topic and qos values and the rest of the document is
identical to the expansion without injection; 1 otherwise (e.g. a wrong
indentation, which the parser may accept by truncating the document). The QoS
negotiated on the running chain is checked by scripts/controlla_qos_grafo.sh.

Usage:
    python3 scripts/controlla_qos_yaml.py --topic /carter/lidar_victim \\
        --reliability reliable --depth 50
"""

import argparse
import copy
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
YAML_MOLA = ROOT / "config" / "mola" / "lidar_odometry_ros2_local.yaml"


def espandi(yaml_path, topic, topic_type, parser="mola-yaml-parser"):
    """Return (parsed mola-yaml-parser output, error); the dict is None if the parser fails."""
    env = os.environ.copy()
    env["MOLA_LIDAR_TOPIC"] = topic
    if topic_type is None:
        env.pop("MOLA_LIDAR_TOPIC_TYPE", None)
    else:
        env["MOLA_LIDAR_TOPIC_TYPE"] = topic_type
    r = subprocess.run([parser, str(yaml_path)], capture_output=True, text=True, env=env, timeout=60)
    if r.returncode != 0:
        msg = (r.stderr.strip() or r.stdout.strip()).splitlines()
        return None, " | ".join(msg[:2]) + f" (rc {r.returncode})"
    try:
        return yaml.safe_load(r.stdout), None
    except yaml.YAMLError as e:
        return None, str(e)


def voce_lidar(doc, topic):
    """Subscribe entry of the ROS 2 bridge for the given topic (None if missing)."""
    mods = doc.get("modules") if isinstance(doc, dict) else None
    for m in mods if isinstance(mods, list) else []:
        par = m.get("params") if isinstance(m, dict) else None
        sub = par.get("subscribe") if isinstance(par, dict) else None
        for v in sub if isinstance(sub, list) else []:
            if isinstance(v, dict) and v.get("topic") == topic:
                return v
    return None


def confronta(doc, ref, topic, reliability, depth):
    """List of problems found (empty means the check passed)."""
    err = []
    v = voce_lidar(doc, topic)
    if v is None:
        return [f"nessuna voce subscribe con topic {topic} (documento troncato?)"]
    q = v.get("qos")
    if not isinstance(q, dict):
        err.append(f"voce {topic} senza blocco qos")
    else:
        if str(q.get("reliability")) != reliability:
            err.append(f"qos.reliability = {q.get('reliability')!r}, attesa {reliability!r}")
        if depth is not None and q.get("depth") != depth:
            err.append(f"qos.depth = {q.get('depth')!r}, attesa {depth}")
    if ref is not None:
        a, b = copy.deepcopy(doc), copy.deepcopy(ref)
        va, vb = voce_lidar(a, topic), voce_lidar(b, topic)
        if va is not None:
            va.pop("qos", None)
        if vb is not None:
            vb.pop("qos", None)
        if a != b:
            err.append("il resto del documento differisce dall'espansione senza iniezione")
    return err


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topic", required=True)
    ap.add_argument("--reliability", default="reliable")
    ap.add_argument("--depth", type=int, default=50)
    ap.add_argument("--yaml", default=str(YAML_MOLA))
    ap.add_argument("--parser", default="mola-yaml-parser")
    a = ap.parse_args(argv)
    if shutil.which(a.parser) is None:
        print(f"controllo QoS: {a.parser} non trovato")
        return 1
    val = os.environ.get("MOLA_LIDAR_TOPIC_TYPE")
    doc, e1 = espandi(a.yaml, a.topic, val, a.parser)
    ref, e2 = espandi(a.yaml, a.topic, None, a.parser)
    if doc is None:
        print(f"controllo QoS {a.topic}: il parser fallisce con il valore attuale ({e1})")
        return 1
    if ref is None:
        print(f"controllo QoS {a.topic}: il parser fallisce senza iniezione ({e2})")
        return 1
    err = confronta(doc, ref, a.topic, a.reliability, a.depth)
    if err:
        for e in err:
            print(f"controllo QoS {a.topic}: {e}")
        return 1
    print(f"controllo QoS {a.topic}: qos {voce_lidar(doc, a.topic)['qos']} presente, resto invariato")
    return 0


if __name__ == "__main__":
    sys.exit(main())
