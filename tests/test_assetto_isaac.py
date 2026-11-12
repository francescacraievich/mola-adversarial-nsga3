"""Base attitude in the Isaac server: isaac/assetto.py and the assetto and set_state
commands (roll, pitch, keep_tilt) of run_isaac_standalone.py.

The server imports isaacsim at load time: its functions are extracted with ast and run
against a fake articulation that records the writes.
"""
import ast
import json
import math
import types
from pathlib import Path

import numpy as np
import pytest

RADICE = Path(__file__).resolve().parents[1]
SERVER = RADICE / "isaac" / "run_isaac_standalone.py"
MODULO = RADICE / "isaac" / "assetto.py"


def _carica():
    # as the server does: from the file, no sys.path, no .pyc
    m = types.ModuleType("assetto")
    m.__file__ = str(MODULO)
    exec(compile(MODULO.read_text(), str(MODULO), "exec"), m.__dict__)
    return m


assetto = _carica()

YAW = [0.0, -0.0, 1e-9, 0.3, -0.7, 1.5707963267948966, 3.0, -3.1, math.pi, -math.pi, 2.5e-3]


def _vecchio(yaw):
    # set_state formula without attitude (run_isaac_standalone.py)
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _rzyx(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch),
                              math.sin(pitch), math.cos(yaw), math.sin(yaw))
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def _stesso_orientamento(a, b, tol=1e-12):
    return abs(abs(float(np.dot(a, b))) - 1.0) < tol


# --- compatibility -----------------------------------------------------------

@pytest.mark.parametrize("yaw", YAW)
def test_sola_imbardata_identica_bit_per_bit(yaw):
    assert assetto.quat_da_yaw(yaw).tobytes() == _vecchio(yaw).tobytes()
    assert assetto.quat_da_euler(0.0, 0.0, yaw).tobytes() == _vecchio(yaw).tobytes()
    assert assetto.quat_da_euler(-0.0, 0.0, yaw).tobytes() == _vecchio(yaw).tobytes()


@pytest.mark.parametrize("yaw", YAW)
def test_set_state_senza_parametri_nuovi_come_prima(yaw):
    q_att = assetto.quat_da_euler(0.01, -0.02, 0.4)
    for cmd in ({"yaw": yaw}, {"yaw": yaw, "keep_tilt": False}):
        assert assetto.quat_set_state(cmd, q_att).tobytes() == _vecchio(yaw).tobytes()
    assert assetto.quat_set_state({"yaw": yaw, "roll": 0, "pitch": 0}, q_att).tobytes() \
        == _vecchio(float(yaw)).tobytes()


# --- Euler <-> quaternion ----------------------------------------------------

ANGOLI = [(r, p, y) for r in (-0.4, -1e-3, 0.0, 8.7e-4, 0.3)
          for p in (-0.45, -8.7e-4, 0.0, 1.7e-3, 0.2)
          for y in (-3.0, -0.5, 0.0, 0.9, 2.8)]


@pytest.mark.parametrize("r,p,y", ANGOLI)
def test_andata_e_ritorno(r, p, y):
    q = assetto.quat_da_euler(r, p, y)
    assert abs(np.linalg.norm(q) - 1.0) < 1e-14
    np.testing.assert_allclose(_mat(q), _rzyx(r, p, y), atol=1e-14)   # ZYX convention
    np.testing.assert_allclose(assetto.euler_da_quat(q), (r, p, y), atol=1e-12)
    assert _stesso_orientamento(assetto.quat_da_euler(*assetto.euler_da_quat(q)), q)


def test_imbardata_come_yaw_del_server():
    # same formula as _yaw (run_isaac_standalone.py), also with tilt
    for r, p, y in ANGOLI:
        q = assetto.quat_da_euler(r, p, y)
        w, x, yy, z = q
        atteso = float(np.arctan2(2.0 * (w * z + x * yy), 1.0 - 2.0 * (yy * yy + z * z)))
        assert assetto.euler_da_quat(q)[2] == atteso


def test_prodotto_e_composizione_di_rotazioni():
    a, b = assetto.quat_da_euler(0.1, -0.2, 0.7), assetto.quat_da_euler(-0.3, 0.05, -1.2)
    np.testing.assert_allclose(_mat(assetto.prodotto(a, b)), _mat(a) @ _mat(b), atol=1e-14)


# --- keep_tilt ----------------------------------------------------------------

@pytest.mark.parametrize("r,p,y", ANGOLI)
@pytest.mark.parametrize("y_nuovo", [-2.9, -0.01, 0.0, 1.3, math.pi])
def test_keep_tilt_conserva_tilt_e_sostituisce_yaw(r, p, y, y_nuovo):
    q_att = assetto.quat_da_euler(r, p, y)
    q = assetto.quat_keep_tilt(q_att, y_nuovo)
    rr, pp, yy = assetto.euler_da_quat(q)
    assert abs(rr - r) < 1e-12 and abs(pp - p) < 1e-12
    assert abs(math.remainder(yy - y_nuovo, 2 * math.pi)) < 1e-12
    assert _stesso_orientamento(q, assetto.quat_da_euler(r, p, y_nuovo))
    # q_att rotated about world z: the base z axis only changes azimuth
    za, zn = _mat(q_att)[:, 2], _mat(q)[:, 2]
    assert abs(za[2] - zn[2]) < 1e-14


def test_keep_tilt_su_imbardata_pura_e_sullo_stesso_yaw():
    for y in YAW:
        np.testing.assert_allclose(assetto.quat_keep_tilt(_vecchio(0.2), y), _vecchio(y), atol=1e-15)
    q_att = assetto.quat_da_euler(1.2e-3, -7e-4, 0.66)
    np.testing.assert_allclose(assetto.quat_keep_tilt(q_att, 0.66), q_att, atol=1e-15)


def test_set_state_keep_tilt_e_6d():
    q_att = assetto.quat_da_euler(1.0e-3, -2.0e-3, 0.5)
    q = assetto.quat_set_state({"x": 1, "y": 2, "yaw": -0.3, "keep_tilt": True}, q_att)
    np.testing.assert_allclose(assetto.euler_da_quat(q), (1.0e-3, -2.0e-3, -0.3), atol=1e-12)
    q = assetto.quat_set_state({"yaw": 0.25, "roll": 8.7e-4, "pitch": -1.7e-3}, q_att)
    np.testing.assert_allclose(assetto.euler_da_quat(q), (8.7e-4, -1.7e-3, 0.25), atol=1e-12)


@pytest.mark.parametrize("cmd,frase", [
    ({"yaw": 0.1, "roll": 0.0, "pitch": 0.0, "keep_tilt": True}, "alternativi"),
    ({"yaw": 0.1, "roll": 0.01, "keep_tilt": True}, "alternativi"),
    ({"yaw": 0.1, "roll": 0.01}, "sia roll sia pitch"),
    ({"yaw": 0.1, "pitch": 0.01}, "sia roll sia pitch"),
    ({"yaw": 0.1, "roll": "a", "pitch": 0.0}, "non numerico"),
    ({"yaw": 0.1, "roll": None, "pitch": 0.0}, "non numerico"),
    ({"yaw": 0.1, "roll": float("nan"), "pitch": 0.0}, "non finito"),
    ({"yaw": 0.1, "roll": 0.0, "pitch": 3.0}, "oltre"),
    ({"yaw": 0.1, "keep_tilt": "true"}, "true o false"),
    ({"yaw": 0.1, "keep_tilt": 1}, "true o false"),
])
def test_errori_espliciti(cmd, frase):
    with pytest.raises(ValueError, match=frase):
        assetto.quat_set_state(cmd, _vecchio(0.0))


# --- attitude reply ----------------------------------------------------------

def test_risposta_assetto_json():
    q = assetto.quat_da_euler(1e-3, -2e-3, 0.7)
    r = assetto.risposta_assetto(np.array([1.0, 2.0, 0.05]), q, np.array([0.1, 0.0, 0.0]),
                                 np.zeros(3), ["a", "b"], np.array([0.5, -0.5]), np.zeros(2))
    r2 = json.loads(json.dumps(r))
    assert r2 == r
    assert r["pose"] == [1.0, 2.0, r["yaw"]] and r["pos"] == [1.0, 2.0, 0.05]
    np.testing.assert_allclose([r["roll"], r["pitch"], r["yaw"]], (1e-3, -2e-3, 0.7), atol=1e-12)
    assert r["giunti"] == {"nomi": ["a", "b"], "pos": [0.5, -0.5], "vel": [0.0, 0.0]}
    assert "giunti" not in assetto.risposta_assetto(np.zeros(3), q, np.zeros(3), np.zeros(3))


# --- server: functions extracted from the source -----------------------------

class ArtFinta:
    def __init__(self, q):
        self.pos, self.q = np.array([3.0, -1.0, 0.0123]), np.asarray(q, dtype=np.float64)
        self.lin, self.ang = np.array([0.2, 0.01, -1e-4]), np.array([1e-4, -2e-4, 0.05])
        self.jpos, self.jvel = np.arange(7.0), np.arange(7.0) / 10
        self.dof_names = [f"g{i}" for i in range(7)]
        self.scritture = []

    def get_world_pose(self):
        return self.pos.copy(), self.q.copy()

    def set_world_pose(self, position, orientation):
        self.scritture.append(("pose", position, orientation))

    def get_linear_velocity(self): return self.lin.copy()
    def get_angular_velocity(self): return self.ang.copy()
    def get_joint_positions(self): return self.jpos.copy()
    def get_joint_velocities(self): return self.jvel.copy()
    def set_joint_velocities(self, v): self.scritture.append(("jvel", v))
    def set_joint_positions(self, v): self.scritture.append(("jpos", v))
    def set_linear_velocity(self, v): self.scritture.append(("lin", v))
    def set_angular_velocity(self, v): self.scritture.append(("ang", v))


class TimelineFinta:
    def is_playing(self): return False
    def play(self): raise AssertionError("play non atteso")
    def pause(self): raise AssertionError("pause non atteso")


def _server(art):
    albero = ast.parse(SERVER.read_text())
    nomi = {"_assetto", "_yaw", "_pose_of", "_snapshot", "_restore", "_handle"}
    corpo = [n for n in albero.body
             if (isinstance(n, ast.FunctionDef) and n.name in nomi)
             or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in ("_CHIAVI_ASSETTO",
                                                                            "_CAPS_ASSETTO")
                                                   for t in n.targets))]
    assert len(corpo) == len(nomi) + 2
    g = {"np": np, "Path": Path, "__file__": str(SERVER), "_mod_assetto": None,
         "_saved": None, "_timeline": TimelineFinta(), "_get_art": lambda: art}
    exec(compile(ast.Module(body=corpo, type_ignores=[]), str(SERVER), "exec"), g)
    return g


def test_server_set_state_di_sempre():
    for yaw in YAW:
        art = ArtFinta(assetto.quat_da_euler(1e-3, 2e-3, 0.1))
        g = _server(art)
        cmd = {"cmd": "set_state", "x": 1.5, "y": -0.5, "yaw": yaw, "vx": 0.1, "vy": 0.0,
               "wz": 0.02, "zero_joints": True}
        assert g["_handle"](cmd) == (None, 1)
        assert g["_mod_assetto"] is None            # the attitude module is not even loaded
        tipo, pos, q = art.scritture[0]
        assert tipo == "pose" and q.tobytes() == _vecchio(yaw).tobytes()
        assert pos.tolist() == [1.5, -0.5, 0.0123]
        assert [s[0] for s in art.scritture] == ["pose", "jvel", "lin", "ang"]


def test_server_set_state_6d_e_keep_tilt():
    art = ArtFinta(assetto.quat_da_euler(1e-3, -2e-3, 0.1))
    g = _server(art)
    assert g["_handle"]({"cmd": "set_state", "x": 1, "y": 2, "yaw": 0.4, "keep_tilt": True}) == (None, 1)
    np.testing.assert_allclose(assetto.euler_da_quat(art.scritture[0][2]), (1e-3, -2e-3, 0.4), atol=1e-12)
    g["_handle"]({"cmd": "set_state", "x": 1, "y": 2, "yaw": 0.4, "roll": 5e-4, "pitch": 7e-4})
    np.testing.assert_allclose(assetto.euler_da_quat(art.scritture[4][2]), (5e-4, 7e-4, 0.4), atol=1e-12)
    assert art.scritture[4][1].tolist() == [1.0, 2.0, 0.0123]   # z of the restore


def test_server_errore_senza_scritture():
    art = ArtFinta(_vecchio(0.0))
    g = _server(art)
    with pytest.raises(ValueError, match="alternativi"):
        g["_handle"]({"cmd": "set_state", "x": 1, "y": 2, "yaw": 0.4, "roll": 0.0, "pitch": 0.0,
                      "keep_tilt": True})
    assert art.scritture == []


def test_server_assetto_e_caps():
    q = assetto.quat_da_euler(-1e-3, 3e-3, -2.0)
    art = ArtFinta(q)
    g = _server(art)
    r, frame = g["_handle"]({"cmd": "assetto"})
    assert frame == 0 and art.scritture == [] and r["ok"] and r["playing"] is False
    assert r["quat_wxyz"] == q.tolist() and r["pos"] == art.pos.tolist()
    assert r["lin"] == art.lin.tolist() and r["ang"] == art.ang.tolist()
    assert r["giunti"]["nomi"] == art.dof_names and r["giunti"]["pos"] == art.jpos.tolist()
    np.testing.assert_allclose([r["roll"], r["pitch"], r["yaw"]], (-1e-3, 3e-3, -2.0), atol=1e-12)
    assert r["pose"] == g["_pose_of"](g["_snapshot"](art))     # same pose as the pose command
    json.dumps(r)
    caps = g["_handle"]({"cmd": "caps"})[0]["caps"]
    assert caps[0] == "set_state" and {"set_state_assetto", "assetto"} <= set(caps)


def test_caricamento_del_modulo_dal_server_senza_pyc():
    art = ArtFinta(_vecchio(0.0))
    g = _server(art)
    m = g["_assetto"]()
    assert isinstance(m, types.ModuleType) and m is g["_assetto"]()
    assert m.quat_da_yaw(0.3).tobytes() == _vecchio(0.3).tobytes()
    cache = RADICE / "isaac" / "__pycache__"
    assert not list(cache.glob("assetto*.pyc")) if cache.exists() else True
