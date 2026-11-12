"""UDP probe: reading /proc/net/snmp and /proc/net/udp from fake text."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sonda_udp as su  # noqa: E402

SNMP = """Ip: Forwarding DefaultTTL InReceives
Ip: 1 64 100
Udp: InDatagrams NoPorts InErrors OutDatagrams RcvbufErrors SndbufErrors InCsumErrors IgnoredMulti MemErrors
Udp: 1000 3 7 900 5 1 0 2 0
UdpLite: InDatagrams NoPorts InErrors OutDatagrams RcvbufErrors SndbufErrors InCsumErrors IgnoredMulti MemErrors
UdpLite: 0 0 0 0 0 0 0 0 0
"""

UDP = """   sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode ref pointer drops
  612: 00000000:1CE8 00000000:0000 07 00000000:00000A00 00:00000000 00000000  {uid}        0 4242 2 0000000000000000 17
  613: 0100007F:1CE9 00000000:0000 07 00000010:00000000 00:00000000 00000000     0        0 4343 2 0000000000000000 0
"""

UDP6 = """  sl  local_address                         remote_address                        st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode ref pointer drops
  100: 00000000000000000000000000000000:1D4C 00000000000000000000000000000000:0000 07 00000000:00000000 00:00000000 00000000  {uid}        0 5151 2 0000000000000000 3
"""


def test_leggi_snmp():
    d = su.leggi_snmp(SNMP)
    assert d == {"InDatagrams": 1000, "NoPorts": 3, "InErrors": 7, "OutDatagrams": 900,
                 "RcvbufErrors": 5, "SndbufErrors": 1}
    assert su.leggi_snmp("Ip: a\nIp: 1\n") is None


def test_leggi_udp():
    s = su.leggi_udp(UDP.format(uid=1000))
    assert s[0]["porta"] == 0x1CE8 and s[0]["rx_queue"] == 0xA00 and s[0]["drops"] == 17
    assert s[0]["uid"] == 1000 and s[0]["inode"] == 4242
    assert s[1]["tx_queue"] == 0x10 and s[1]["uid"] == 0


def test_campione_con_proc_finto(tmp_path):
    uid = os.getuid()
    proc = tmp_path / "proc"
    (proc / "net").mkdir(parents=True)
    (proc / "net" / "snmp").write_text(SNMP)
    (proc / "net" / "udp").write_text(UDP.format(uid=uid))
    (proc / "net" / "udp6").write_text(UDP6.format(uid=uid))
    p = proc / "4321"
    (p / "fd").mkdir(parents=True)
    (p / "comm").write_text("mola-cli\n")
    os.symlink("socket:[4242]", p / "fd" / "5")
    os.symlink("socket:[5151]", p / "fd" / "6")
    os.symlink("/dev/null", p / "fd" / "0")
    shm = tmp_path / "shm"
    shm.mkdir()
    (shm / "fastrtps_abc").write_bytes(b"x" * 100)
    (shm / "fastrtps_abc_el").write_bytes(b"")
    (shm / "altro").write_bytes(b"y" * 10)
    snmp, socks, shm_v, mappa = su.campione(uid, {}, proc=str(proc), shm=str(shm))
    assert snmp["RcvbufErrors"] == 5
    assert sorted((s["inode"], s["pid"], s["nome"], s["fam"]) for s in socks) == [
        (4242, 4321, "mola-cli", "udp"), (5151, 4321, "mola-cli", "udp6")]
    assert shm_v[0] == 2 and shm_v[1] == 100
    assert mappa[4242] == (4321, "mola-cli")


def test_main_breve(tmp_path):
    assert su.main(["--out", str(tmp_path), "--ogni", "0.2", "--durata", "0.5"]) == 0
    righe = (tmp_path / "udp_snmp.csv").read_text().splitlines()
    assert righe[0].startswith("t_wall,t_mono,InDatagrams")
    assert len(righe) >= 3
