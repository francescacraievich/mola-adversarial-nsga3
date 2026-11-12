"""
Scan structures for the coherent operator mode (--operator-mode coherent).

For each clean scan the points are voxelised (VOXEL_M) in the anchor frame (the
sensor frame at the first scan of the 1 m segment, carried to the current scan
with the estimated pose). A PCA over the K_NB nearest voxel centroids gives the
dimensionality (a1/a2/a3), main axis, normal and curvature, from which each voxel
gets a type: vertical edge, wall, floor, curvature or none. Connected components
per type (link LINK voxels; walls also need normals within WALL_LINK_DEG) are the
structures, identified by their smallest voxel key.

At the first scan of the segment one structure per type is chosen as anchor with
a fixed rule (R1 by size, R2 by azimuth from genes 7-8) and frozen in the anchor
frame. Rule "caso" draws one zone per operator uniformly among the candidates of
its type (sorted by id) with a local generator seeded by (genome seed, segment,
gene); without seed or segment it falls back to R1. With the "insieme" option each
operator gets a fixed set of zones, drawn the same way without replacement, until
its measure (points, removable points or voxels) reaches the number of points the
independent operator mode touches on the first scan with the same genome.

On later scans each anchor is associated with the structures of the same type that
fall in its voxels dilated by one (or with the nearest one within the gate); the
anchor grows only by contiguity, up to GROWTH times its initial extent. At the
first scan the same association is iterated to a fixed point.

Pose trackers for scan k in the anchor frame:
  scan  rigid SE(2) alignment of a fixed subset of scan k onto scan k-1, using only
        clean scans (same code in twin and victim). Alignment comes before the
        analysis, so the voxel grid does not inherit the prediction error.
  odom  base pose from the attacker's MOLA, passed by the node in the scan context;
        sensor-on-base pose is LIDAR_POSE.
With the scan tracker and a pose in the context, the association under the
attacker's pose is also computed and logged as a check.

No choice depends on point order: points are sorted by voxel key and coordinates
before every sum.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

# Speed-ups with byte-identical output. PERT_OTTIMIZZAZIONI: "spente" (default),
# "tutte" (O1-O5, O7, O9) or a comma-separated list; unknown names are an error.
# PERT_OTTIMIZZAZIONI_THREAD: pool size (default 4).
#   O1 centroid neighbours in threads          O2 blockwise eigh in threads
#   O3 single gather of cent[nb]                O4 target KD-tree built in a thread
#   O5 anchor voxel membership via searchsorted
#   O7 ICP queries in threads                   O9 edge score queries (g13) in threads
OTT_NOMI = ("O1", "O2", "O3", "O4", "O5", "O7", "O9")


def _leggi_ottimizzazioni(valore):
    v = (valore or "spente").strip()
    if v == "spente":
        return frozenset()
    if v == "tutte":
        return frozenset(OTT_NOMI)
    nomi = frozenset(x.strip() for x in v.split(",") if x.strip())
    ignoti = sorted(nomi - set(OTT_NOMI))
    if ignoti or not nomi:
        raise ValueError(f"PERT_OTTIMIZZAZIONI={valore!r}: nomi sconosciuti {ignoti} "
                         f"(validi: spente, tutte, {', '.join(OTT_NOMI)})")
    return nomi


OTT = _leggi_ottimizzazioni(os.environ.get("PERT_OTTIMIZZAZIONI"))
WORKERS = int(os.environ.get("PERT_OTTIMIZZAZIONI_THREAD", "4"))
if WORKERS < 1:
    raise ValueError(f"PERT_OTTIMIZZAZIONI_THREAD={WORKERS}: serve almeno 1")
_POOL = None
_POOL_LOCK = threading.Lock()


def _pool():
    global _POOL
    if _POOL is None:
        with _POOL_LOCK:
            if _POOL is None:
                _POOL = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="ott")
    return _POOL


def avvia_pool():
    """Create the pool and its threads at node start-up, so the first scan does not
    pay for it. No-op without optimisations."""
    if not OTT:
        return
    b = threading.Barrier(WORKERS)
    list(_pool().map(lambda _: b.wait(timeout=10.0), range(WORKERS)))


def _eigh_blocchi(cov):
    """np.linalg.eigh over blocks in threads. Each 3x3 matrix is decomposed on its own
    (same LAPACK call), so the result is bit-identical to a single call."""
    n = len(cov)
    if WORKERS <= 1 or n < 2048:
        return np.linalg.eigh(cov)
    lam = np.empty(cov.shape[:2])
    vec = np.empty(cov.shape)
    lim = np.linspace(0, n, WORKERS + 1).astype(int)

    def blocco(a, b):
        lam[a:b], vec[a:b] = np.linalg.eigh(cov[a:b])
    list(_pool().map(lambda ab: blocco(*ab), zip(lim[:-1], lim[1:])))
    return lam, vec


def _in_ordinati(x, s):
    """np.isin(x, s) for sorted, unique s (dilated keys)."""
    if len(s) == 0:
        return np.zeros(len(x), dtype=bool)
    i = np.searchsorted(s, x)
    np.minimum(i, len(s) - 1, out=i)
    return s[i] == x


VOXEL_M = 0.15            # voxel size, m (same as MOLA's decimation)
K_NB = 16                 # neighbours for the centroid PCA
LINK = 1.5                # component link distance, in voxels
WALL_LINK_DEG = 20.0      # max angle between normals of two linked wall voxels
A1_EDGE = 0.6             # min linearity of an edge
AXIS_Z_EDGE = 0.87        # min |axis_z| of a vertical edge (within 30 deg)
A2_PLANE = 0.6            # min planarity of wall and floor
WALL_NZ_MAX = 0.5         # max |n_z| of a wall
FLOOR_NZ_MIN = 0.87       # min |n_z| of the floor
A3_CURV = 0.3             # min scatter for curvature
MIN_VOX = 5               # min voxels of an eligible structure
MAX_VOX = 400             # max voxels of an eligible structure (~9 m2 of wall)
MIN_VOX_WALL = 20         # min voxels of an eligible wall (g6: one ghost per voxel, H >= 2 H_MIN)
RANGE_MIN_M = 1.0         # min L-infinity distance from the base (MOLA filter)
RANGE_MAX_M = 25.0        # max L-infinity distance from the base
DIST_MAX_M = 11.0         # max horizontal centroid range at selection (12 m minus 1 m of motion)
HIT_MAX_M = 11.5          # max horizontal range of hit voxels (12 m threshold)
OCCL_GAP_M = 1.0          # occluded edge: neighbour past the azimuth border closer by this (LOAM)
OVERLAP_RHO = 0.5         # min fraction of a structure's voxels inside the dilated anchor
GATE_M = 0.30             # centroid gate when no structure overlaps the anchor
GATE_DEG = 20.0           # max angle between anchor and structure axis/normal in the gate
GROWTH = 1.5              # max anchor extent along its main axis (relative)
R2_TIE_DEG = 2.0          # R2: within this azimuth gap R1 decides
PAV_BANDA_M = 2 * VOXEL_M  # rule "caso": voxel "on the floor" within this distance of the plane
PAV_QUOTA = 0.5           # rule "caso": on the floor if at least this fraction of voxels is in band
N_STRUCT = 1              # anchors per type
# LiDAR pose on the base (T_base<-sensor, translation only), as passed to MOLA by the
# launch script.
LIDAR_POSE = (-0.2317, 0.0, 0.526)
# Scan-to-scan alignment: decreasing gates (m), minimum pairs.
ICP_GATES = (0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.12, 0.1, 0.1, 0.1)
ICP_MIN_PAIRS = 30
ICP_SUB = 16              # one point in ICP_SUB (chosen per point, not by order) as source
ICP_EXTRA = 10            # extra iterations at the last gate, until the minimum step
ICP_STEP_M = 1e-9         # minimum step (m; rad for the angle)

NONE, EDGE_V, WALL, FLOOR, CURV = 0, 1, 2, 3, 4
TYPE_NAMES = {NONE: "nessuno", EDGE_V: "spigolo_v", WALL: "parete", FLOOR: "pavimento",
              CURV: "curvatura"}
TYPE_BY_NAME = {v: k for k, v in TYPE_NAMES.items()}
STRUCT_TYPES = (EDGE_V, WALL, CURV)

_OFF = 1 << 20
_DIL = np.array([(dx << 42) + (dy << 21) + dz for dx in (-1, 0, 1)
                 for dy in (-1, 0, 1) for dz in (-1, 0, 1)], dtype=np.int64)


def pack_keys(k3: np.ndarray) -> np.ndarray:
    """Pack integer voxel keys (n, 3) into one int64 (21 bits per axis)."""
    return ((k3[:, 0] + _OFF) << 42) | ((k3[:, 1] + _OFF) << 21) | (k3[:, 2] + _OFF)


def voxel_keys(xyz: np.ndarray, voxel: float = VOXEL_M) -> np.ndarray:
    return pack_keys(np.floor(xyz / voxel).astype(np.int64))


def dilate(keys: np.ndarray) -> np.ndarray:
    """Keys of the voxels within one voxel (27-neighbourhood), sorted and unique."""
    if len(keys) == 0:
        return keys
    return np.unique((keys[:, None] + _DIL[None, :]).ravel())


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def sub_mask(xyz: np.ndarray, sub: int = ICP_SUB) -> np.ndarray:
    """Point subset chosen by a hash of the coordinate bits: independent of point
    order and of the pose."""
    u = np.ascontiguousarray(xyz[:, :3], dtype=np.float64).view(np.uint64)
    with np.errstate(over="ignore"):
        h = (u[:, 0] * np.uint64(0x9E3779B97F4A7C15) + u[:, 1] * np.uint64(0xC2B2AE3D27D4EB4F)
             + u[:, 2] * np.uint64(0x165667B19E3779F9))
    h ^= h >> np.uint64(29)
    return (h >> np.uint64(40)) % np.uint64(sub) == 0


def se2_apply(pose, xyz: np.ndarray) -> np.ndarray:
    """p' = R(th) p + (x, y, 0), element-wise."""
    x, y, th = pose
    c, s = np.cos(th), np.sin(th)
    out = np.empty_like(xyz, dtype=np.float64)
    out[:, 0] = c * xyz[:, 0] - s * xyz[:, 1] + x
    out[:, 1] = s * xyz[:, 0] + c * xyz[:, 1] + y
    out[:, 2] = xyz[:, 2]
    return out


def se2_rot(th, v: np.ndarray) -> np.ndarray:
    c, s = np.cos(th), np.sin(th)
    v = np.asarray(v, dtype=np.float64)
    out = v.copy()
    out[..., 0] = c * v[..., 0] - s * v[..., 1]
    out[..., 1] = s * v[..., 0] + c * v[..., 1]
    return out


def se2_compose(a, b):
    """a o b."""
    c, s = np.cos(a[2]), np.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], float(_wrap(a[2] + b[2])))


def se2_inv(a):
    c, s = np.cos(a[2]), np.sin(a[2])
    return (-(c * a[0] + s * a[1]), -(-s * a[0] + c * a[1]), -a[2])


def se2_between(a, b):
    """inv(a) o b from differences: exactly (0, 0, 0) for a = b, whereas
    se2_compose(se2_inv(a), b) leaves a rounding residual."""
    c, s = np.cos(a[2]), np.sin(a[2])
    dx, dy = b[0] - a[0], b[1] - a[1]
    return (c * dx + s * dy, -s * dx + c * dy, float(_wrap(b[2] - a[2])))


IDENTITY = (0.0, 0.0, 0.0)


def analyse(xyz: np.ndarray, sensor_xy=(0.0, 0.0), voxel: float = VOXEL_M) -> Dict:
    """Voxels, descriptors, types and structures of scan xyz (predicted anchor
    frame). sensor_xy: sensor position in the same frame, used to orient wall
    normals."""
    n = len(xyz)
    pk = voxel_keys(xyz, voxel)
    order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0], pk))
    pks = pk[order]
    starts = np.flatnonzero(np.r_[True, pks[1:] != pks[:-1]])
    counts = np.diff(np.r_[starts, n])
    nv = len(starts)
    cent = np.add.reduceat(xyz[order], starts, axis=0) / counts[:, None]
    vox_sorted = np.repeat(np.arange(nv), counts)
    vox_of_point = np.empty(n, dtype=np.int64)
    vox_of_point[order] = vox_sorted
    vkeys = pks[starts]

    types = np.zeros(nv, dtype=np.int8)
    curv = np.zeros(nv)
    axis = np.zeros((nv, 3))
    normal = np.zeros((nv, 3))
    k = min(K_NB, nv)
    nb = np.arange(nv)[:, None]
    if k >= 4:
        _, nb = cKDTree(cent).query(cent, k=k, workers=WORKERS if "O1" in OTT else 1)
        if "O3" in OTT:
            cn = cent[nb]
            c = cn - cn.mean(axis=1, keepdims=True)
            del cn
        else:
            c = cent[nb] - cent[nb].mean(axis=1, keepdims=True)
        x, y, z = c[..., 0], c[..., 1], c[..., 2]
        inv = 1.0 / (k - 1)
        cov = np.empty((nv, 3, 3))
        cov[:, 0, 0] = (x * x).sum(1) * inv
        cov[:, 1, 1] = (y * y).sum(1) * inv
        cov[:, 2, 2] = (z * z).sum(1) * inv
        cov[:, 0, 1] = cov[:, 1, 0] = (x * y).sum(1) * inv
        cov[:, 0, 2] = cov[:, 2, 0] = (x * z).sum(1) * inv
        cov[:, 1, 2] = cov[:, 2, 1] = (y * z).sum(1) * inv
        lam, vec = _eigh_blocchi(cov) if "O2" in OTT else np.linalg.eigh(cov)   # ascending
        lam = np.maximum(lam, 0.0)
        s1, s2, s3 = np.sqrt(lam[:, 2]), np.sqrt(lam[:, 1]), np.sqrt(lam[:, 0])
        s1e = np.where(s1 > 0, s1, 1.0)
        a1 = np.where(s1 > 0, (s1 - s2) / s1e, 0.0)
        a2 = np.where(s1 > 0, (s2 - s3) / s1e, 0.0)
        a3 = np.where(s1 > 0, s3 / s1e, 0.0)
        tot = lam.sum(axis=1)
        curv = np.where(tot > 0, lam[:, 0] / np.where(tot > 0, tot, 1.0), 0.0)
        curv = np.where(curv < 1e-6, 0.0, curv)
        axis = vec[:, :, 2]
        normal = vec[:, :, 0]
        az, nz = np.abs(axis[:, 2]), np.abs(normal[:, 2])
        t_edge = (a1 >= A1_EDGE) & (az >= AXIS_Z_EDGE)
        t_wall = (a2 >= A2_PLANE) & (nz <= WALL_NZ_MAX)
        t_floor = (a2 >= A2_PLANE) & (nz >= FLOOR_NZ_MIN)
        t_curv = a3 >= A3_CURV
        types[t_curv] = CURV
        types[t_floor] = FLOOR
        types[t_wall] = WALL
        types[t_edge] = EDGE_V
        # drop curvature voxels at the floor junction (a floor neighbour within LINK voxels)
        cv = np.flatnonzero(types == CURV)
        if len(cv):
            dn = np.linalg.norm(cent[nb[cv]] - cent[cv][:, None, :], axis=2)
            pav = ((types[nb[cv]] == FLOOR) & (dn <= LINK * voxel)).any(axis=1)
            types[cv[pav]] = NONE
        # canonical signs: axis with z >= 0, normal towards the sensor
        axis = np.where(axis[:, 2:3] < 0, -axis, axis)
        to_s = np.c_[sensor_xy[0] - cent[:, 0], sensor_xy[1] - cent[:, 1], np.zeros(nv)]
        normal = np.where(((normal * to_s).sum(1) < 0)[:, None], -normal, normal)

    fr = {"n": n, "order": order, "starts": starts, "counts": counts, "cent": cent,
          "vkeys": vkeys, "vox_of_point": vox_of_point, "types": types, "curv": curv,
          "axis": axis, "normal": normal, "nb": nb}
    _components(fr)
    return fr


def _components(fr: Dict):
    """Connected components per type and vectorised structure records."""
    cent, types, vkeys = fr["cent"], fr["types"], fr["vkeys"]
    nv = len(cent)
    vox_struct = np.full(nv, -1, dtype=np.int64)
    s_type, s_first = [], []
    labels_all = []
    base = 0
    for t in STRUCT_TYPES:
        idx = np.flatnonzero(types == t)
        if len(idx) == 0:
            continue
        if len(idx) > 1:
            pairs = cKDTree(cent[idx]).query_pairs(LINK * VOXEL_M, output_type="ndarray")
        else:
            pairs = np.zeros((0, 2), dtype=np.int64)
        if t == WALL and len(pairs):
            nn = fr["normal"][idx]
            dots = np.abs((nn[pairs[:, 0]] * nn[pairs[:, 1]]).sum(1))
            pairs = pairs[dots >= np.cos(np.radians(WALL_LINK_DEG))]
        m = len(idx)
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(m, m))
        nc, lab = connected_components(g, directed=False)
        vox_struct[idx] = lab + base
        s_type.append(np.full(nc, t, dtype=np.int8))
        base += nc
    ns = base
    fr["vox_struct"] = vox_struct
    fr["s_type"] = np.concatenate(s_type) if s_type else np.zeros(0, dtype=np.int8)
    if ns == 0:
        for key in ("s_cid", "s_nvox", "s_npts", "s_dir", "s_axis1"):
            fr[key] = np.zeros((0, 3)) if key in ("s_dir", "s_axis1") else np.zeros(0, dtype=np.int64)
        fr["s_cent"] = np.zeros((0, 3))
        return
    v = np.flatnonzero(vox_struct >= 0)
    sv = vox_struct[v]
    s_nvox = np.bincount(sv, minlength=ns)
    s_npts = np.bincount(sv, weights=fr["counts"][v], minlength=ns).astype(np.int64)
    s_cid = np.full(ns, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(s_cid, sv, vkeys[v])
    s_cent = np.stack([np.bincount(sv, weights=cent[v, j], minlength=ns) for j in range(3)], 1)
    s_cent /= s_nvox[:, None]
    # structure direction: mean axis (edges) or mean normal (walls)
    st = fr["s_type"][sv]
    d = np.where((st == WALL)[:, None], fr["normal"][v], fr["axis"][v])
    s_dir = np.stack([np.bincount(sv, weights=d[:, j], minlength=ns) for j in range(3)], 1)
    nrm = np.linalg.norm(s_dir, axis=1, keepdims=True)
    s_dir = np.where(nrm > 1e-12, s_dir / np.maximum(nrm, 1e-12), 0.0)
    s_dir[fr["s_type"] == CURV] = 0.0
    # structure main axis (PCA of its centroids), used for the extent
    c = cent[v] - s_cent[sv]
    cov = np.zeros((ns, 3, 3))
    for i in range(3):
        for j in range(i, 3):
            cov[:, i, j] = cov[:, j, i] = np.bincount(sv, weights=c[:, i] * c[:, j], minlength=ns)
    _, vec = np.linalg.eigh(cov)
    s_axis1 = vec[:, :, 2]
    s_axis1 = np.where(s_axis1[:, 2:3] < 0, -s_axis1, s_axis1)
    fr.update({"s_cid": s_cid, "s_nvox": s_nvox, "s_npts": s_npts, "s_cent": s_cent,
               "s_dir": s_dir, "s_axis1": s_axis1})


def _angle_deg(u, v):
    """Angle between unoriented directions (deg)."""
    d = np.abs(np.clip((u * v).sum(-1), -1.0, 1.0))
    return np.degrees(np.arccos(d))


def bersaglio_icp(fr: Dict, xyz_a: np.ndarray, differito: bool = False):
    """Alignment target for the next scan: all points in canonical order with the
    type and normal of their voxel (None if empty). differito=True (O4) returns
    (target with "tree" None, future of the KD-tree built in a thread); the tree is
    put in place by completa_bersaglio."""
    o = fr["order"]
    vp = fr["vox_of_point"][o]
    pts = xyz_a[o]
    if differito:
        if not len(pts):
            return None, None
        fut = _pool().submit(cKDTree, pts, balanced_tree=False)
        return {"pts": pts, "types": fr["types"][vp], "normal": fr["normal"][vp], "tree": None}, fut
    return {"pts": pts, "types": fr["types"][vp], "normal": fr["normal"][vp],
            "tree": cKDTree(pts, balanced_tree=False)} if len(pts) else None


def copia_ancora(A: Dict) -> Dict:
    """Independent copy of an anchor (growth changes keys and dil)."""
    B = dict(A)
    for key in ("keys", "dil", "cent", "axis1", "dir", "edge_dir"):
        B[key] = np.array(A[key], copy=True)
    B["band"] = tuple(float(v) for v in A["band"])
    B["lost"] = 0
    B["assoc"] = []
    return B


class StructureTracker:
    """Anchors and tracking for one stream (one generator). reset() at every new
    genome, i.e. every segment."""

    def __init__(self, track: str = "scan", rule: str = "R1",
                 lidar_pose=LIDAR_POSE, voxel: float = VOXEL_M, escludi_pavimento: bool = False,
                 ripiego: bool = False, insieme: bool = False):
        if track not in ("scan", "odom"):
            raise ValueError(f"inseguitore sconosciuto: {track}")
        if rule not in ("R1", "R2", "caso"):
            raise ValueError(f"regola sconosciuta: {rule}")
        if escludi_pavimento and rule != "caso":
            raise ValueError("escludi_pavimento richiede la regola caso")
        if ripiego and rule != "caso":
            raise ValueError("ripiego richiede la regola caso")
        if insieme and rule != "caso":
            raise ValueError("insieme richiede la regola caso")
        self.ripiego = ripiego
        self.insieme = bool(insieme)
        self.track = track
        self.rule = rule
        self.escludi_pavimento = escludi_pavimento
        self.lidar_xy = (float(lidar_pose[0]), float(lidar_pose[1]))
        self.base_in_sensor = -np.asarray(lidar_pose, dtype=np.float64)
        self.voxel = voxel
        self.reset()

    def completa_bersaglio(self):
        """O4: put the target KD-tree in place (waits for the thread). Afterwards the
        tracker state equals the one without O4."""
        fut = self.__dict__.pop("_albero", None)
        if fut is not None:
            self.prev["tree"] = fut.result()

    def reset(self):
        self.__dict__.pop("_albero", None)
        self.k = -1
        self.poses = []          # sensor pose in the anchor frame, per scan
        self.stamps = []
        self.prev = None         # previous scan target (anchor frame)
        self.anchors = {}        # type -> list of anchors (dict), in rule order
        self.per_op = False      # rule "caso" active: anchors per operator ("g5", ...)
        self.scelta = {}         # rule "caso": candidates per operator
        self.margin = {}
        self.center0 = None
        self.max_range0 = None
        self.odom0 = None        # sensor pose in odom at the first scan (check only)
        self.fisse = False       # zones installed by the attacker: no selection in the stream

    # -- attacker zones ----------------------------------------------------

    def installa(self, zone):
        """Install zones chosen by the attacker on the segment snapshot (object with
        anchors, scelta, center0, max_range0, bersaglio). The anchor frame is the
        snapshot's sensor frame, and the snapshot scan is the previous scan for
        aligning the first scan of the stream. No selection: anchors are copied and
        then tracked."""
        self.reset()
        self.fisse = True
        self.poses = [IDENTITY]
        self.stamps = [None]
        self.prev = zone.bersaglio
        self.per_op = True
        self.anchors = {op: [copia_ancora(A) for A in lst] for op, lst in zone.anchors.items()}
        self.scelta = dict(zone.scelta)
        self.margin = {op: float("nan") for op in self.anchors}
        self.center0 = None if zone.center0 is None else np.array(zone.center0, dtype=np.float64)
        self.max_range0 = zone.max_range0

    def installa_vuoto(self):
        """Invalid zones: no anchors and no selection for the whole segment (localised
        operators leave the scan untouched)."""
        self.reset()
        self.fisse = True
        self.per_op = True

    # -- poses -------------------------------------------------------------

    def _sensor_in_odom(self, base_pose):
        bx, by, byaw = base_pose
        return se2_compose((bx, by, byaw), (self.lidar_xy[0], self.lidar_xy[1], 0.0))

    def _predict(self, stamp):
        if not self.poses:
            return IDENTITY
        if len(self.poses) == 1:
            return self.poses[-1]
        a, b = self.poses[-2], self.poses[-1]
        inc = se2_compose(se2_inv(a), b)
        r = 1.0
        if stamp is not None and self.stamps[-1] is not None and self.stamps[-2] is not None:
            d0 = self.stamps[-1] - self.stamps[-2]
            if d0 > 1e-6:
                r = float(np.clip((stamp - self.stamps[-1]) / d0, 0.0, 4.0))
        return se2_compose(b, (inc[0] * r, inc[1] * r, inc[2] * r))

    def _odom_pose(self, ctx):
        p = None if ctx is None else ctx.get("pose")
        if p is None or self.odom0 is None:
            return None
        return se2_between(self.odom0, self._sensor_in_odom(p))

    # -- scan-to-scan alignment ----------------------------------------------

    def _icp(self, xyz, pose0):
        """SE(2) sensor pose in the anchor frame for scan xyz, starting from pose0.
        One point in ICP_SUB of the current scan is matched to the previous scan:
        point-to-plane on walls, planar point-to-point on other structure types. The
        type is that of the nearest previous point; pairs on floor or untyped voxels
        are dropped. Source sorted by coordinates."""
        prev = self.prev
        if prev is None:
            return pose0, 0, float("nan"), 0
        src = xyz[sub_mask(xyz)]
        src = src[np.lexsort((src[:, 2], src[:, 1], src[:, 0]))]
        if len(src) < ICP_MIN_PAIRS:
            return pose0, 0, float("nan"), 0
        self.completa_bersaglio()
        tree = prev["tree"]
        pose = pose0
        npair, rms, it = 0, float("nan"), 0
        gates = ICP_GATES + (ICP_GATES[-1],) * ICP_EXTRA
        wq = WORKERS if "O7" in OTT else 1
        for it, gate in enumerate(gates, 1):
            p = se2_apply(pose, src)
            d, j = tree.query(p, k=1, distance_upper_bound=gate, workers=wq)
            ok = np.isfinite(d)
            jj = j[ok]
            tt = prev["types"][jj]
            keep = np.isin(tt, STRUCT_TYPES)
            if keep.sum() < ICP_MIN_PAIRS:
                break
            pp, q, tt, jj = p[ok][keep], prev["pts"][jj[keep]], tt[keep], jj[keep]
            wall = tt == WALL
            H = np.zeros((3, 3)); g = np.zeros(3)
            res = []
            if wall.any():
                nrm = prev["normal"][jj[wall]][:, :2]
                nn = np.linalg.norm(nrm, axis=1, keepdims=True)
                good = nn[:, 0] > 0.5
                nrm = nrm[good] / nn[good]
                pw, qw = pp[wall][good], q[wall][good]
                r = ((pw[:, :2] - qw[:, :2]) * nrm).sum(1)
                J = np.c_[nrm[:, 0] * -pw[:, 1] + nrm[:, 1] * pw[:, 0], nrm[:, 0], nrm[:, 1]]
                H += J.T @ J; g += J.T @ r
                res.append(r)
            pt = ~wall
            if pt.any():
                pq, qq = pp[pt], q[pt]
                rx, ry = pq[:, 0] - qq[:, 0], pq[:, 1] - qq[:, 1]
                Jx = np.c_[-pq[:, 1], np.ones(len(pq)), np.zeros(len(pq))]
                Jy = np.c_[pq[:, 0], np.zeros(len(pq)), np.ones(len(pq))]
                H += Jx.T @ Jx + Jy.T @ Jy
                g += Jx.T @ rx + Jy.T @ ry
                res.append(np.hypot(rx, ry))
            H += np.eye(3) * (1e-9 * max(np.trace(H), 1.0))
            dth, dx, dy = np.linalg.solve(H, -g)
            pose = se2_compose((dx, dy, dth), pose)
            r_all = np.concatenate(res) if res else np.zeros(0)
            npair = int(keep.sum())
            rms = float(np.sqrt(np.mean(r_all ** 2))) if len(r_all) else float("nan")
            if it >= len(ICP_GATES) and max(abs(dx), abs(dy), abs(dth)) < ICP_STEP_M:
                break
        if npair < ICP_MIN_PAIRS:
            return pose0, npair, rms, it
        return pose, npair, rms, it

    # -- selection and association -----------------------------------------

    def _eligible(self, fr, pose):
        c_s = se2_apply(se2_inv(pose), fr["s_cent"]) if len(fr["s_cent"]) else fr["s_cent"]
        linf = np.abs(c_s - self.base_in_sensor[None, :]).max(axis=1) if len(c_s) else np.zeros(0)
        ok = (linf >= RANGE_MIN_M) & (linf <= RANGE_MAX_M)
        ok &= (fr["s_nvox"] >= MIN_VOX) & (fr["s_nvox"] <= MAX_VOX)
        ok &= (fr["s_type"] != WALL) | (fr["s_nvox"] >= MIN_VOX_WALL)
        dist = np.hypot(c_s[:, 0], c_s[:, 1]) if len(c_s) else np.zeros(0)
        ok &= dist <= DIST_MAX_M
        if len(c_s):
            ok &= ~self._occluso(fr, pose, ok)
        az = np.arctan2(c_s[:, 1], c_s[:, 0]) if len(c_s) else np.zeros(0)
        return ok, dist, az

    def _occluso(self, fr, pose, ok):
        """Vertical edges on the far side of an occlusion (LOAM criterion): just outside
        their azimuth interval (within atan(2 voxel / r)), at the same height, there is a
        non-floor voxel closer than r - OCCL_GAP_M. Used only for selection."""
        occ = np.zeros(len(fr["s_type"]), dtype=bool)
        cand = np.flatnonzero(ok & (fr["s_type"] == EDGE_V))
        if len(cand) == 0:
            return occ
        vc = se2_apply(se2_inv(pose), fr["cent"])
        vr = np.hypot(vc[:, 0], vc[:, 1])
        va = np.arctan2(vc[:, 1], vc[:, 0])
        nf = fr["types"] != FLOOR
        for s in cand:
            v = np.flatnonzero(fr["vox_struct"] == s)
            if len(v) == 0:
                continue
            r = float(np.median(vr[v]))
            a0 = float(np.median(va[v]))
            da = np.angle(np.exp(1j * (va[v] - a0)))
            lo, hi = float(da.min()), float(da.max())
            dlt = float(np.arctan(2.0 * self.voxel / max(r, 1e-6)))
            z0, z1 = float(vc[v, 2].min()), float(vc[v, 2].max())
            dall = np.angle(np.exp(1j * (va - a0)))
            lato = ((dall >= lo - dlt) & (dall < lo)) | ((dall > hi) & (dall <= hi + dlt))
            m = nf & lato & (vc[:, 2] >= z0) & (vc[:, 2] <= z1) & (vr < r - OCCL_GAP_M)
            occ[s] = bool(m.any())
        return occ

    def _rank(self, fr, t, ok, dist, az, az_target):
        cand = np.flatnonzero(ok & (fr["s_type"] == t))
        if len(cand) == 0:
            return [], float("nan")
        r1 = cand[np.lexsort((fr["s_cid"][cand], dist[cand], -fr["s_nvox"][cand]))]
        margin = (float(fr["s_nvox"][r1[0]] - fr["s_nvox"][r1[1]]) if len(r1) > 1 else float("nan"))
        if self.rule in ("R1", "caso") or az_target is None:
            return list(r1[:N_STRUCT]), margin
        # R2: azimuth closest to genes 7-8; within R2_TIE_DEG R1 decides
        left = list(r1)
        out = []
        dang = np.degrees(np.abs(_wrap(az - az_target)))
        while left and len(out) < N_STRUCT:
            best = min(dang[s] for s in left)
            pick = next(s for s in left if dang[s] <= best + R2_TIE_DEG)
            out.append(pick)
            left.remove(pick)
        if len(out) > 1:
            margin = float(dang[out[1]] - dang[out[0]])
        return out, margin

    def _sul_pavimento(self, fr):
        """Structures with at least PAV_QUOTA of their voxels within PAV_BANDA_M of the
        floor plane (median height of floor voxels, anchor frame; base height under the
        sensor when there are no floor voxels)."""
        ns = len(fr["s_type"])
        if ns == 0:
            return np.zeros(0, dtype=bool), float("nan")
        fl = fr["types"] == FLOOR
        zf = float(np.median(fr["cent"][fl, 2])) if fl.any() else float(self.base_in_sensor[2])
        v = np.flatnonzero(fr["vox_struct"] >= 0)
        vicino = np.abs(fr["cent"][v, 2] - zf) <= PAV_BANDA_M
        q = np.bincount(fr["vox_struct"][v], weights=vicino, minlength=ns) / np.maximum(fr["s_nvox"], 1)
        return q >= PAV_QUOTA, zf

    def _rank_caso(self, fr, t, gene, ok, seme, metro, pav=None, pav_tutte=None):
        """Rule "caso": one structure drawn uniformly among type-t candidates sorted by
        id, with a local generator (numpy global state untouched). pav: excluded
        (floor) structures. pav_tutte (fallback): with no type-t candidate, draw among
        all eligible off-floor candidates of any type."""
        cand = np.flatnonzero(ok & (fr["s_type"] == t))
        info = {"n_cand": int(len(cand)), "ripiego": False}
        if pav is not None:
            cand = cand[~pav[cand]]
            info["n_cand_fuori_pav"] = int(len(cand))
        if len(cand) == 0 and pav_tutte is not None:
            cand = np.flatnonzero(ok & np.isin(fr["s_type"], STRUCT_TYPES) & ~pav_tutte)
            info["ripiego"] = True
            info["n_cand_ripiego"] = int(len(cand))
        if len(cand) == 0:
            return [], info
        cand = cand[np.argsort(fr["s_cid"][cand], kind="stable")]
        rng = np.random.default_rng(np.random.SeedSequence([int(seme), int(metro), int(gene)]))
        return [int(cand[int(rng.integers(len(cand)))])], info

    def _ordine_insieme(self, fr, t, gene, ok, seme, metro, pav=None, pav_tutte=None):
        """"insieme" option: type-t zones (minus pav) in random order, then, with the
        fallback, the other eligible off-floor zones of any type. Candidates sorted by
        id, local generator from (seed, segment, gene) as in rule "caso"."""
        info = {"ripiego": False}
        cand = np.flatnonzero(ok & (fr["s_type"] == t))
        info["n_cand"] = int(len(cand))
        if pav is not None:
            cand = cand[~pav[cand]]
            info["n_cand_fuori_pav"] = int(len(cand))
        rng = np.random.default_rng(np.random.SeedSequence([int(seme), int(metro), int(gene)]))
        cand = cand[np.argsort(fr["s_cid"][cand], kind="stable")]
        ordine = [int(x) for x in cand[rng.permutation(len(cand))]]
        if pav_tutte is not None:
            altre = np.flatnonzero(ok & np.isin(fr["s_type"], STRUCT_TYPES) & ~pav_tutte)
            altre = altre[~np.isin(altre, cand)]
            altre = altre[np.argsort(fr["s_cid"][altre], kind="stable")]
            ordine += [int(x) for x in altre[rng.permutation(len(altre))]]
            info["n_cand_ripiego"] = int(len(altre))
        return ordine, info

    def _scegli_insieme(self, fr, ordine, gene, obiettivo, misura, keys_a, cent_a, dirs_a, noise_dir, pose):
        """Anchors of the set: zones in order, each measured on the voxels it hits at
        the first scan (same association and distance filter as normal anchors).
        misura: "punti", "tolti" (one point in two per voxel) or "voxel". Zones are
        added until the sum reaches the target; for gene 10 a zone that would exceed
        it is skipped, since that zone is moved as a whole."""
        riempi = gene != 10
        out, tot = [], 0
        for s in ordine:
            if obiettivo <= 0 or (riempi and tot >= obiettivo):
                break
            A = self._make_anchor(fr, s, keys_a, cent_a, dirs_a, noise_dir)
            while True:
                nk = len(A["keys"])
                v, _ = self._associate(fr, A, keys_a, cent_a, dirs_a)
                if len(A["keys"]) == nk:
                    break
            if len(v):
                dv = np.hypot(cent_a[v, 0] - pose[0], cent_a[v, 1] - pose[1])
                v = v[dv <= HIT_MAX_M]
            c = fr["counts"][v]
            m = int(len(v)) if misura == "voxel" else int(c.sum() if misura == "punti" else (c // 2).sum())
            if m == 0 or (not riempi and tot + m > obiettivo):
                continue
            out.append(A)
            tot += m
        info = {"obiettivo": int(obiettivo), "misura": misura, "n_zone": len(out), "misura_tot": int(tot),
                "tipi": [TYPE_NAMES[A["type"]] for A in out]}
        return out, info

    def _make_anchor(self, fr, s, keys_a, cent_a, dirs_a, noise_dir):
        v = np.flatnonzero(fr["vox_struct"] == s)
        keys = np.unique(keys_a[v])
        c = fr["s_cent"][s].copy()
        ax1 = fr["s_axis1"][s].copy()
        proj = (cent_a[v] - c) @ ax1
        lo, hi = float(proj.min()), float(proj.max())
        span = max(hi - lo, self.voxel)
        mid = 0.5 * (lo + hi)
        band = (mid - 0.5 * GROWTH * span, mid + 0.5 * GROWTH * span)
        sdir = dirs_a[s].copy()
        # edge shift: horizontal direction orthogonal to the axis, closest to the
        # genes 0-2 direction (radial if that is zero)
        nd = np.asarray(noise_dir, dtype=np.float64)
        h = nd - (nd @ sdir) * sdir if np.any(sdir) else nd.copy()
        h = np.array([h[0], h[1], 0.0])
        if np.linalg.norm(h) < 1e-6:
            h = np.array([c[0], c[1], 0.0])
        h /= max(np.linalg.norm(h), 1e-12)
        return {"cid0": int(fr["s_cid"][s]), "s0": int(s), "type": int(fr["s_type"][s]), "keys": keys,
                "dil": dilate(keys), "cent": c, "axis1": ax1, "band": band, "dir": sdir,
                "edge_dir": h, "nvox0": int(fr["s_nvox"][s]), "lost": 0, "assoc": []}

    def _fuori_tipo(self, fr, A, keys_a, cent_a):
        """No structure of the anchor type: non-floor voxels of any type inside the dilated
        anchor and within the band along its axis (types change with the viewpoint). The
        anchor does not grow."""
        vv = np.flatnonzero(fr["types"] != FLOOR)
        if len(vv) == 0:
            return np.zeros(0, dtype=np.int64)
        inA = _in_ordinati(keys_a[vv], A["dil"]) if "O5" in OTT else np.isin(keys_a[vv], A["dil"])
        proj = (cent_a[vv] - A["cent"]) @ A["axis1"]
        inband = (proj >= A["band"][0]) & (proj <= A["band"][1])
        return vv[inA & inband]

    def _associate(self, fr, A, keys_a, cent_a, dirs_a, grow=True):
        """Voxels hit by anchor A in the current scan and associated structures."""
        t = A["type"]
        vt = np.flatnonzero(fr["types"] == t)
        if len(vt) == 0:
            return self._fuori_tipo(fr, A, keys_a, cent_a), []
        sv = fr["vox_struct"][vt]
        inA = _in_ordinati(keys_a[vt], A["dil"]) if "O5" in OTT else np.isin(keys_a[vt], A["dil"])
        ns = len(fr["s_type"])
        cnt_in = np.bincount(sv, weights=inA, minlength=ns)
        ov = np.where(fr["s_nvox"] > 0, cnt_in / np.maximum(fr["s_nvox"], 1), 0.0)
        assoc = np.flatnonzero((ov >= OVERLAP_RHO) & (fr["s_type"] == t))
        if len(assoc):
            proj = (cent_a[vt] - A["cent"]) @ A["axis1"]
            inband = (proj >= A["band"][0]) & (proj <= A["band"][1])
            hit = vt[np.isin(sv, assoc) & inA & inband]
        else:
            cand = np.flatnonzero(fr["s_type"] == t)
            d = np.linalg.norm(fr["s_cent"][cand] - A["cent"], axis=1)
            okc = d <= GATE_M
            if t != CURV and len(cand):
                okc &= _angle_deg(dirs_a[cand], A["dir"]) <= GATE_DEG
            if not okc.any():
                return self._fuori_tipo(fr, A, keys_a, cent_a), []
            dd = np.where(okc, d, np.inf)
            best = cand[int(np.lexsort((fr["s_cid"][cand], dd))[0])]
            assoc = np.array([best])
            hit = vt[sv == best]
        if grow and len(hit):
            new = np.union1d(A["keys"], keys_a[hit])
            if len(new) != len(A["keys"]):
                A["keys"] = new
                A["dil"] = dilate(new)
        return hit, [int(fr["s_cid"][s]) for s in assoc]

    # -- per-scan step ---------------------------------------------------------

    def update(self, xyz: np.ndarray, types_needed, az_target=None, noise_dir=(1.0, 0.0, 0.0),
               ctx: Optional[Dict] = None, seme=None, metro=None, operatori=None,
               obiettivi=None) -> Dict:
        """Analyse clean scan xyz (sensor frame) and find the voxels hit by each anchor.
        Returns the analysis plus pose, hits {type: [voxel indices per anchor]}, info.
        Rule "caso" with seme, metro and operatori ({"g5": (5, type), ...}) at the first
        scan: anchors and hits per operator, key "gN". "insieme" option: obiettivi
        {"g5": (count, measure), ...} at the first scan gives a set of zones per
        listed operator."""
        ctx = ctx or {}
        stamp = ctx.get("stamp")
        self.k += 1
        first = self.k == 0 and not self.fisse
        info = {"k": self.k, "track": self.track, "rule": self.rule}
        if first:
            pose_pred = IDENTITY
            if ctx.get("pose") is not None:
                self.odom0 = self._sensor_in_odom(ctx["pose"])
        elif self.track == "odom":
            p = self._odom_pose(ctx)
            info["pose_missing"] = p is None
            pose_pred = p if p is not None else self._predict(stamp)
        else:
            pose_pred = self._predict(stamp)
        if not first and self.track == "scan":
            pose_pred, npair, rms, nit = self._icp(xyz, pose_pred)
            info.update({"icp_pairs": npair, "icp_rms_m": rms, "icp_it": nit})
        # analysis in the anchor frame at the corrected pose
        pose = pose_pred
        xyz_a = se2_apply(pose, xyz)
        fr = analyse(xyz_a, sensor_xy=(pose[0], pose[1]), voxel=self.voxel)
        cent_a = fr["cent"]
        keys_a = voxel_keys(cent_a, self.voxel)
        s_cent_a = fr["s_cent"]
        dirs_a = fr["s_dir"]
        if first:
            self.center0 = xyz.mean(axis=0)
            self.max_range0 = float(np.linalg.norm(xyz, axis=1).max()) + 1e-6
            ok, dist, az = self._eligible(fr, pose)
            self.per_op = (self.rule == "caso" and seme is not None and metro is not None
                           and operatori is not None)
            if self.per_op:
                self.scelta = {}
                pav = None
                if self.escludi_pavimento or self.ripiego:
                    pav, zf = self._sul_pavimento(fr)
                    self.scelta["quota_pavimento_m"] = zf
                for op, (gene, t) in sorted(operatori.items()):
                    if self.insieme and obiettivi is not None and op in obiettivi:
                        ordine, sc = self._ordine_insieme(
                            fr, t, gene, ok, seme, metro,
                            pav if (op == "g10" and self.escludi_pavimento) else None,
                            pav if self.ripiego else None)
                        self.anchors[op], si = self._scegli_insieme(
                            fr, ordine, gene, obiettivi[op][0], obiettivi[op][1],
                            keys_a, cent_a, dirs_a, noise_dir, pose)
                        sc.update(si)
                        sc["ripiego"] = any(A["type"] != t for A in self.anchors[op])
                        self.margin[op] = float("nan")
                        self.scelta[op] = sc
                        continue
                    else:
                        picks, sc = self._rank_caso(fr, t, gene, ok, seme, metro,
                                                    pav if (op == "g10" and self.escludi_pavimento) else None,
                                                    pav if self.ripiego else None)
                    self.anchors[op] = [self._make_anchor(fr, s, keys_a, cent_a, dirs_a, noise_dir)
                                        for s in picks]
                    self.margin[op] = float("nan")
                    self.scelta[op] = sc
            for t in (() if self.per_op else types_needed):
                picks, margin = self._rank(fr, t, ok, dist, az, az_target)
                self.anchors[t] = [self._make_anchor(fr, s, keys_a, cent_a, dirs_a, noise_dir)
                                   for s in picks]
                self.margin[t] = margin
        hits = {}
        ctrl = None
        pose_o = self._odom_pose(ctx) if (self.track == "scan" and not first) else None
        if pose_o is not None:
            dlt = se2_compose(pose_o, se2_inv(pose))
            cent_o = se2_apply(dlt, cent_a)
            keys_o = voxel_keys(cent_o, self.voxel)
            dirs_o = se2_rot(dlt[2], dirs_a) if len(dirs_a) else dirs_a
            fr_o = dict(fr)
            fr_o["s_cent"] = se2_apply(dlt, s_cent_a) if len(s_cent_a) else s_cent_a
            ctrl = {"n": 0, "same": 0}
        for t, lst in self.anchors.items():
            hits[t] = []
            for A in lst:
                if first:
                    # same association as later scans, iterated with growth to a
                    # fixed point: with an identical scan and zero motion the hit
                    # voxels stay those of k = 0
                    while True:
                        nk = len(A["keys"])
                        v, ids = self._associate(fr, A, keys_a, cent_a, dirs_a)
                        if len(A["keys"]) == nk:
                            break
                else:
                    if ctrl is not None:
                        _, ids_o = self._associate(fr_o, A, keys_o, cent_o, dirs_o, grow=False)
                    v, ids = self._associate(fr, A, keys_a, cent_a, dirs_a)
                    if ctrl is not None:
                        ctrl["n"] += 1
                        ctrl["same"] += int(sorted(ids) == sorted(ids_o))
                if len(v):
                    # no voxels beyond HIT_MAX_M from the sensor (anchors grow along walls)
                    dv = np.hypot(cent_a[v, 0] - pose[0], cent_a[v, 1] - pose[1])
                    v = v[dv <= HIT_MAX_M]
                A["lost"] = 0 if len(v) else A["lost"] + 1
                A["assoc"] = ids
                hits[t].append(v)
        # target for the next alignment: all points in canonical order, with the
        # type and normal of their voxel
        if self.track == "scan":
            if "O4" in OTT:
                self.prev, fut = bersaglio_icp(fr, xyz_a, differito=True)
                if fut is not None:
                    self._albero = fut
            else:
                self.prev = bersaglio_icp(fr, xyz_a)
        self.poses.append(pose)
        self.stamps.append(stamp)
        info["pose"] = (float(pose[0]), float(pose[1]), float(np.degrees(pose[2])))
        nome = (lambda t: t) if self.per_op else (lambda t: TYPE_NAMES[t])
        info["anchors"] = {nome(t): [A["cid0"] for A in lst] for t, lst in self.anchors.items()}
        info["assoc"] = {nome(t): [A["assoc"] for A in lst] for t, lst in self.anchors.items()}
        info["hit_vox"] = {nome(t): [int(len(v)) for v in hl] for t, hl in hits.items()}
        info["margin"] = {nome(t): m for t, m in self.margin.items()}
        if self.per_op:
            info["scelta"] = self.scelta
            info["tipo_ancora"] = {t: [TYPE_NAMES[A["type"]] for A in lst] for t, lst in self.anchors.items()}
        if ctrl is not None:
            info["ctrl_odom_same"] = ctrl["same"]
            info["ctrl_odom_n"] = ctrl["n"]
        fr["pose"] = pose
        fr["xyz_pred"] = xyz_a
        fr["keys_a"] = keys_a
        fr["cent_a"] = cent_a
        fr["hits"] = hits
        fr["info"] = info
        return fr

    def center_and_range(self, pose):
        """Distortion centre and normalisation range, fixed at the first scan of the
        segment, expressed in the current sensor frame."""
        if self.center0 is None:
            return None, None
        c = se2_apply(se2_inv(pose), self.center0[None, :])[0]
        return c, self.max_range0
