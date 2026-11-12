#!/usr/bin/env python3
"""Fast DDS UDP counters and shared memory every --ogni s (reads /proc only).

- udp_snmp.csv: Udp line of /proc/net/snmp (InDatagrams, NoPorts, InErrors,
  OutDatagrams, RcvbufErrors, SndbufErrors) with deltas from the previous
  sample, plus count and size of the /dev/shm/fastrtps_* files;
- udp_socket.csv: for each UDP socket (IPv4 and IPv6) of the user's processes,
  pid and process name (from /proc/<pid>/fd), local port, rx_queue, tx_queue and
  drops. The inode -> pid map is rebuilt only when new inodes appear.
No ROS node. Usage: python3 scripts/sonda_udp.py --out <dir> [--ogni 1] [--durata S]
"""

import argparse
import csv
import os
import signal
import sys
import time
from pathlib import Path

CAMPI_SNMP = ("InDatagrams", "NoPorts", "InErrors", "OutDatagrams", "RcvbufErrors", "SndbufErrors")


def leggi_snmp(testo):
    """Udp line of /proc/net/snmp as (header, values); None if missing."""
    righe = [r.split() for r in testo.splitlines() if r.startswith("Udp:")]
    if len(righe) < 2:
        return None
    nomi, valori = righe[0][1:], righe[1][1:]
    d = dict(zip(nomi, (int(v) for v in valori)))
    return {k: d.get(k) for k in CAMPI_SNMP}


def leggi_udp(testo):
    """Sockets of /proc/net/udp or udp6: list of dicts with port, queues, uid, inode, drops."""
    out = []
    for r in testo.splitlines()[1:]:
        c = r.split()
        if len(c) < 13:
            continue
        loc = c[1]
        tx, rx = c[4].split(":")
        out.append({"porta": int(loc.rsplit(":", 1)[1], 16), "locale": loc,
                    "tx_queue": int(tx, 16), "rx_queue": int(rx, 16),
                    "uid": int(c[7]), "inode": int(c[9]),
                    "drops": int(c[12]) if len(c) > 12 else None})
    return out


def mappa_inode(uid, proc="/proc"):
    """Socket inode -> (pid, name) for the processes of user uid."""
    m = {}
    for p in os.listdir(proc):
        if not p.isdigit():
            continue
        base = os.path.join(proc, p)
        try:
            if os.stat(base).st_uid != uid:
                continue
            with open(os.path.join(base, "comm")) as f:
                nome = f.read().strip()
            for fd in os.listdir(os.path.join(base, "fd")):
                try:
                    dest = os.readlink(os.path.join(base, "fd", fd))
                except OSError:
                    continue
                if dest.startswith("socket:["):
                    m[int(dest[8:-1])] = (int(p), nome)
        except OSError:
            continue
    return m


def shm_fastrtps(cartella="/dev/shm"):
    """(count, apparent bytes, allocated bytes) of the fastrtps_* files."""
    n = dim = occ = 0
    try:
        for e in os.scandir(cartella):
            if e.name.startswith("fastrtps_"):
                try:
                    st = e.stat(follow_symlinks=False)
                except OSError:
                    continue
                n += 1
                dim += st.st_size
                occ += st.st_blocks * 512
    except OSError:
        pass
    return n, dim, occ


def _leggi(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


def campione(uid, mappa, proc="/proc", shm="/dev/shm"):
    """Return (snmp, user sockets with pid and name, shm, updated map)."""
    snmp = leggi_snmp(_leggi(os.path.join(proc, "net", "snmp")))
    socks = []
    for fam in ("udp", "udp6"):
        for s in leggi_udp(_leggi(os.path.join(proc, "net", fam))):
            if s["uid"] == uid:
                s["fam"] = fam
                socks.append(s)
    if any(s["inode"] not in mappa for s in socks):
        mappa = mappa_inode(uid, proc)
    for s in socks:
        s["pid"], s["nome"] = mappa.get(s["inode"], ("", ""))
    return snmp, socks, shm_fastrtps(shm), mappa


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--ogni", type=float, default=1.0)
    ap.add_argument("--durata", type=float, default=0.0, help="s; 0 = fino a INT/TERM")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fermo = {"v": False}
    signal.signal(signal.SIGINT, lambda *_: fermo.update(v=True))
    signal.signal(signal.SIGTERM, lambda *_: fermo.update(v=True))
    uid = os.getuid()
    f1 = open(out / "udp_snmp.csv", "w", newline="")
    f2 = open(out / "udp_socket.csv", "w", newline="")
    w1, w2 = csv.writer(f1), csv.writer(f2)
    w1.writerow(("t_wall", "t_mono") + CAMPI_SNMP + tuple("d_" + k for k in CAMPI_SNMP)
                + ("shm_n", "shm_byte", "shm_byte_occupati", "socket_utente", "drops_utente"))
    w2.writerow(("t_wall", "pid", "nome", "fam", "porta", "rx_queue", "tx_queue", "drops", "inode"))
    mappa, prec = {}, None
    t0 = time.monotonic()
    prossimo = t0
    try:
        while not fermo["v"] and (a.durata <= 0 or time.monotonic() - t0 < a.durata):
            tw, tm = time.time(), time.monotonic()
            snmp, socks, shm, mappa = campione(uid, mappa)
            snmp = snmp or {k: None for k in CAMPI_SNMP}
            diff = [(snmp[k] - prec[k]) if (prec and snmp[k] is not None and prec[k] is not None) else ""
                    for k in CAMPI_SNMP]
            w1.writerow((f"{tw:.6f}", f"{tm:.6f}") + tuple(snmp[k] for k in CAMPI_SNMP)
                        + tuple(diff) + shm
                        + (len(socks), sum(s["drops"] or 0 for s in socks)))
            for s in socks:
                w2.writerow((f"{tw:.6f}", s["pid"], s["nome"], s["fam"], s["porta"],
                             s["rx_queue"], s["tx_queue"], s["drops"], s["inode"]))
            f1.flush()
            f2.flush()
            prec = snmp
            prossimo += a.ogni
            attesa = prossimo - time.monotonic()
            if attesa > 0:
                time.sleep(attesa)
            else:
                prossimo = time.monotonic()
    finally:
        f1.close()
        f2.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
