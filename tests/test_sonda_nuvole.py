"""Cloud probe: stamp from the CDR header and log row (synthetic data)."""

import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sonda_nuvole as sn  # noqa: E402


def test_stamp_cdr_little_endian_sintetico():
    buf = b"\x00\x01\x00\x00" + struct.pack("<iI", 12, 400000000) + b"\x00" * 40
    assert sn.stamp_cdr(buf) == pytest.approx(12.4, abs=1e-9)


def test_stamp_cdr_big_endian_e_corto():
    buf = b"\x00\x00\x00\x00" + struct.pack(">iI", 7, 5)
    assert sn.stamp_cdr(buf) == pytest.approx(7 + 5e-9)
    assert sn.stamp_cdr(b"\x00\x01\x00") is None


def test_stamp_cdr_da_messaggio_serializzato():
    rclpy_ser = pytest.importorskip("rclpy.serialization")
    from sensor_msgs.msg import PointCloud2, PointField
    m = PointCloud2()
    m.header.stamp.sec, m.header.stamp.nanosec = 1234, 500000000
    m.header.frame_id = "base_link"
    m.height, m.width, m.point_step = 1, 10, 16
    m.row_step = 160
    m.fields = [PointField(name=n, offset=4 * i, datatype=7, count=1) for i, n in enumerate("xyzi")]
    m.data = bytes(160)
    buf = rclpy_ser.serialize_message(m)
    assert sn.stamp_cdr(buf) == pytest.approx(1234.5, abs=1e-9)


def test_riga_messaggio_con_info_dict():
    buf = b"\x00\x01\x00\x00" + struct.pack("<iI", 3, 100000000) + b"\x00" * 100
    info = {"source_timestamp": 1_700_000_000_000_000_000,
            "received_timestamp": 1_700_000_000_002_500_000,
            "publication_sequence_number": 9, "reception_sequence_number": 4}
    r = sn.riga_messaggio(buf, info, 1.5, 2.5)
    d = dict(zip(sn.COLONNE, r))
    assert d["stamp"] == "3.100000000"
    assert d["dds_ms"] == "2.500"
    assert d["size_b"] == len(buf)
    assert d["pub_seq"] == 9 and d["rec_seq"] == 4


def test_nome_file():
    assert sn.nome_file("/carter/lidar_victim") == "sonda_carter_lidar_victim.csv"


def test_registro_scrive_a_blocchi(tmp_path):
    reg = sn.Registro(tmp_path / "x.csv", blocco=3)
    for i in range(4):
        reg.aggiungi((i,) * len(sn.COLONNE))
    assert len((tmp_path / "x.csv").read_text().splitlines()) == 4   # header + 3
    reg.chiudi()
    assert len((tmp_path / "x.csv").read_text().splitlines()) == 5
