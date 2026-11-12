"""
Attack zones chosen by the attacker (--zone-attaccante option of node and
orchestrator).

At the start of each 1 m segment, with the robot stopped, the attacker takes a
snapshot: its last clean scan and the pose its own MOLA estimated for that scan
(base in odom). For each genome the zones are computed only from the snapshot and
the seed (genome, segment, operator), with the coherent "insieme" logic
(StructureTracker with rule "caso" and "insieme", same operators and caps, targets
of the independent operator mode on the snapshot scan). The anchor frame is the
snapshot sensor frame, fixed in odom through the snapshot pose.

Intensity matches the independent operator mode: for each localised operator the
expected number of points touched per scan equals that mode with the same genome,
taken only inside the zones. g5: expected removed fraction (target on the snapshot
/ snapshot points) applied to the scan points; ghosts: same base (points minus
expected removed); g10: balls with centres and radius as in
_apply_cluster_perturbation (n_clusters centres drawn from the snapshot with the
seed, radius = 10th percentile of distances from the centroid), i.e. the gene 10
zones are the snapshot balls; g13: same target (min(500, 20 % of the points)).

The streams (twin, victim) do not choose: they install the zones from file and
track them with the scan tracker, whose first scan is aligned to the snapshot.

Files (never overwritten):
  w{k}_fotografia.npz   xyzi float32 as received, meta (JSON)
  w{k}_<genoma_sha>.npz meta (JSON) and per-anchor arrays
each with a <file>.sha256 holding the file sha256 and the content hash. The
content hash (canonical sha256 of meta and arrays) does not depend on write time:
two identical computations give the same hash.
"""

import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.perturbations import structures as st

# Generator settings, same as the node's make_generator with --operator-mode insieme.
KW_INSIEME = {"operator_mode": "coherent", "struct_track": "scan", "struct_rule": "caso",
              "struct_g10_pavimento": True, "struct_ripiego": True,
              "struct_g13_punteggio": True, "struct_insieme": True}
LIMITI = ("max_point_shift", "noise_std", "max_dropout_rate", "max_ghost_points_ratio")
STAMP_TOL_S = 1e-6
# Selection margins on the snapshot: zones cover more than the expected points
# because the robot moves during the segment and g5 also removes points in the g13
# zones; per scan at most the expected points are touched.
RISERVA_G10 = 1.5      # snapshot points inside the balls / expected gene 10 points
MAX_PALLE = 4          # at most MAX_PALLE * n_clusters balls
RISERVA_G13 = 3        # g13 zone selection target / expected points


class ZoneNonValide(RuntimeError):
    """Unusable zones or snapshot; stato = short reason."""

    def __init__(self, stato, dettaglio=""):
        super().__init__(f"{stato}{': ' + dettaglio if dettaglio else ''}")
        self.stato = stato


# ---------------------------------------------------------------- genoma

def genoma_f32(genoma) -> np.ndarray:
    """Genome as received by the node (Float32MultiArray), as float64."""
    return np.asarray(genoma, dtype=np.float64).astype(np.float32).astype(np.float64)


def genoma_sha(genoma) -> str:
    return hashlib.sha256(np.asarray(genoma, dtype=np.float64).astype(np.float32).tobytes()).hexdigest()[:16]


def seme_genoma(genoma, test_seed: int = 0) -> int:
    """Node seed for the genome (genome_seed + --test-seed)."""
    h = hashlib.sha1(genoma_f32(genoma).tobytes()).digest()
    return (int.from_bytes(h[:4], "little") + int(test_seed)) % (2 ** 31 - 1)


def _plain(v):
    """Plain JSON values (no numpy types)."""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return [_plain(x) for x in v.tolist()]
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    return v


def firma_parametri(params: Dict) -> str:
    """Fingerprint of the parameters decoded from the genome (exact values)."""
    d = {k: [repr(float(x)) for x in np.ravel(np.asarray(v, dtype=np.float64))]
         for k, v in sorted(params.items())}
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- file

def hash_contenuto(meta: Dict, arrays: Dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(meta, sort_keys=True, separators=(",", ":")).encode())
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(name.encode())
        h.update(a.dtype.str.encode())
        h.update(repr(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _scrivi(path, meta: Dict, arrays: Dict[str, np.ndarray]):
    """Write the npz (new file, never overwritten) and its .sha256.
    Returns (file sha256, content hash)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h = hash_contenuto(meta, arrays)
    buf = io.BytesIO()
    np.savez(buf, meta=np.array(json.dumps(meta, sort_keys=True)), **arrays)
    data = buf.getvalue()
    sha = hashlib.sha256(data).hexdigest()
    with open(path, "xb") as f:
        f.write(data)
    with open(str(path) + ".sha256", "x") as f:
        f.write(f"{sha}  {path.name}\nhash_contenuto {h}\n")
    return sha, h


def _leggi(path, sha256: Optional[str] = None, hash_atteso: Optional[str] = None):
    path = Path(path)
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        raise ZoneNonValide("file_assente", str(path))
    except OSError as e:
        raise ZoneNonValide("illeggibile", f"{path}: {e}")
    sha = hashlib.sha256(data).hexdigest()
    if sha256 is not None and sha != sha256:
        raise ZoneNonValide("sha256_diverso", f"{path}: {sha[:12]} invece di {str(sha256)[:12]}")
    try:
        z = np.load(io.BytesIO(data), allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        arrays = {k: z[k] for k in z.files if k != "meta"}
    except Exception as e:
        raise ZoneNonValide("illeggibile", f"{path}: {e}")
    h = hash_contenuto(meta, arrays)
    if hash_atteso is not None and h != hash_atteso:
        raise ZoneNonValide("hash_diverso", f"{path}: {h[:12]} invece di {str(hash_atteso)[:12]}")
    return meta, arrays, sha, h


# ---------------------------------------------------------------- snapshot

class Fotografia:
    def __init__(self, xyzi, meta, sha256, hash_, path=None):
        self.xyzi = np.asarray(xyzi, dtype=np.float32)
        self.meta = meta
        self.sha256 = sha256
        self.hash = hash_
        self.path = None if path is None else str(path)

    @property
    def cloud(self) -> np.ndarray:
        """Scan as the node sees it (float64 from the message float32)."""
        return self.xyzi.astype(np.float64)


def salva_fotografia(path, xyzi, meta: Dict):
    """xyzi (N, 4): the clean scan as received (exact float32)."""
    a = np.ascontiguousarray(np.asarray(xyzi)[:, :4], dtype=np.float32)
    m = _plain(dict(meta))
    m["n_punti"] = int(len(a))
    m["tipo"] = "fotografia"
    return _scrivi(path, m, {"xyzi": a})


def carica_fotografia(path, sha256: Optional[str] = None, hash_atteso: Optional[str] = None) -> Fotografia:
    meta, arrays, sha, h = _leggi(path, sha256, hash_atteso)
    if meta.get("tipo") != "fotografia" or "xyzi" not in arrays:
        raise ZoneNonValide("illeggibile", f"{path}: non e' una fotografia")
    return Fotografia(arrays["xyzi"], meta, sha, h, path)


def bersaglio_fotografia(foto: Fotografia):
    """Previous-scan target for the first scan of each stream: the snapshot scan
    analysed at the identity pose."""
    xyz = foto.cloud[:, :3]
    pose = st.IDENTITY
    xyz_a = st.se2_apply(pose, xyz)
    fr = st.analyse(xyz_a, sensor_xy=(pose[0], pose[1]), voxel=st.VOXEL_M)
    return st.bersaglio_icp(fr, xyz_a)


def sensore_in_odom(pose_base, lidar_pose=st.LIDAR_POSE):
    return st.se2_compose(tuple(float(v) for v in pose_base), (float(lidar_pose[0]), float(lidar_pose[1]), 0.0))


# ---------------------------------------------------------------- zones

class Zone:
    """Zones of one genome in one segment, ready for StructureTracker.installa."""

    def __init__(self, meta, arrays, sha256, hash_, path=None, bersaglio=None):
        self.meta = meta
        self.sha256 = sha256
        self.hash = hash_
        self.path = None if path is None else str(path)
        self.bersaglio = bersaglio
        self.genoma_sha = meta["genoma_sha"]
        self.seme = int(meta["seme"])
        self.metro = int(meta["metro"])
        self.firma = meta["firma_parametri"]
        # bound multipliers used to compute them (None = off)
        self.moltiplicatori = meta.get("moltiplicatori")
        self.obiettivi = {op: (int(v[0]), str(v[1])) for op, v in (meta.get("obiettivi") or {}).items()}
        self.scelta = meta.get("scelta") or {}
        c0 = meta.get("center0")
        self.center0 = None if c0 is None else np.asarray(c0, dtype=np.float64)
        self.max_range0 = meta.get("max_range0")
        self.quota_g5 = meta.get("quota_g5")
        self.attesi_g10 = meta.get("attesi_g10")
        self.palle_g10 = (np.asarray(arrays["palle_g10_chiavi"], dtype=np.int64)
                          if "palle_g10_chiavi" in arrays else None)
        self.palle_centri = (np.asarray(arrays["palle_g10_centri"], dtype=np.float64)
                             if "palle_g10_centri" in arrays else None)
        pm = meta.get("palle_g10") or {}
        self.palle_raggio = pm.get("raggio_m")
        self.quota_g10 = meta.get("quota_g10")
        self.anchors = {}
        for a in meta["ancore"]:
            nome = a["nome"]
            v = np.asarray(arrays[nome + "_vettori"], dtype=np.float64)
            keys = np.asarray(arrays[nome + "_chiavi"], dtype=np.int64)
            A = {"cid0": int(a["cid0"]), "s0": int(a["s0"]), "type": int(a["type"]), "keys": keys,
                 "dil": st.dilate(keys), "cent": v[0:3].copy(), "axis1": v[3:6].copy(),
                 "band": (float(v[6]), float(v[7])), "dir": v[8:11].copy(),
                 "edge_dir": v[11:14].copy(), "nvox0": int(a["nvox0"]), "lost": 0, "assoc": []}
            self.anchors.setdefault(a["op"], []).append(A)

    def n_zone(self):
        return {op: len(lst) for op, lst in self.anchors.items()}


def generatore(limiti: Dict):
    """Coherent "insieme" generator with the node bounds, plus "moltiplicatori"
    ({family: K}) when present in the bounds."""
    from src.perturbations.perturbation_generator import PerturbationGenerator
    mol = {"moltiplicatori": dict(limiti["moltiplicatori"])} if limiti.get("moltiplicatori") else {}
    return PerturbationGenerator(**{k: float(limiti[k]) for k in LIMITI}, **KW_INSIEME, **mol)


def calcola_zone(foto: Fotografia, genoma, metro: int, limiti: Dict, test_seed: int = 0):
    """Zones of the genome from the snapshot alone: (meta, arrays, tracker,
    generator). The snapshot scan acts as the first scan of the segment."""
    g = genoma_f32(genoma)
    gen = generatore(limiti)
    gen._riserva_scelta = {"g13": RISERVA_G13}
    params = gen.encode_perturbation(g)
    seme = seme_genoma(g, test_seed)
    gen._apply_coherent(foto.cloud, params, ctx={"stamp": None}, seme_genoma=seme, metro=int(metro))
    tr = gen._structs
    ob = gen._obiettivi or {}
    xyz_f = foto.cloud[:, :3]
    quota_g5 = float(ob["g5"][0]) / len(xyz_f) if "g5" in ob and len(xyz_f) else None
    palle, centri, palle_meta = ((palle_g10(xyz_f, seme, metro, gen, ob["g10"][0] if "g10" in ob else 0))
                                 if "g10" in tr.anchors else (None, None, None))
    pose_b = foto.meta.get("posa_base_odom")
    s_odom = sensore_in_odom(pose_b) if pose_b is not None else None
    ancore, arrays = [], {}
    for op in sorted(tr.anchors):
        if op == "g10" and palle is not None:
            continue          # gene 10 uses the balls
        for i, A in enumerate(tr.anchors[op]):
            nome = f"a_{op}_{i:03d}"
            ancore.append({"nome": nome, "op": op, "i": i, "cid0": int(A["cid0"]), "s0": int(A["s0"]),
                           "type": int(A["type"]), "tipo": st.TYPE_NAMES[int(A["type"])],
                           "nvox0": int(A["nvox0"]), "n_chiavi": int(len(A["keys"]))})
            arrays[nome + "_chiavi"] = np.asarray(A["keys"], dtype=np.int64)
            arrays[nome + "_vettori"] = np.r_[A["cent"], A["axis1"], np.asarray(A["band"], dtype=np.float64),
                                              A["dir"], A["edge_dir"]].astype(np.float64)
            if s_odom is not None:
                arrays[nome + "_centro_odom"] = st.se2_apply(s_odom, np.asarray(A["cent"])[None, :])[0]
    if palle is not None:
        arrays["palle_g10_chiavi"] = palle
        arrays["palle_g10_centri"] = centri
    inf = gen.last_struct_info or {}
    meta = _plain({
        "tipo": "zone",
        "genoma_sha": genoma_sha(g),
        "genoma_f32": g.tolist(),
        "seme": seme,
        "metro": int(metro),
        "firma_parametri": firma_parametri(params),
        "limiti": {k: float(limiti[k]) for k in LIMITI},
        "fotografia_file": Path(foto.path).name if foto.path else None,
        "fotografia_sha256": foto.sha256,
        "fotografia_hash": foto.hash,
        "posa_base_odom": pose_b,
        "posa_sensore_odom": s_odom,
        "terna": "sensore della fotografia (odom tramite posa_sensore_odom)",
        "obiettivi": {op: list(v) for op, v in (gen._obiettivi or {}).items()},
        "scelta": tr.scelta,
        "center0": None if tr.center0 is None else list(tr.center0),
        "max_range0": tr.max_range0,
        "ancore": ancore,
        "n_hit_fotografia": inf.get("n_hit"),
        "quota_g5": quota_g5,
        "attesi_g10": ob["g10"][0] if "g10" in ob else None,
        "quota_g10": float(ob["g10"][0]) / len(xyz_f) if "g10" in ob and len(xyz_f) else None,
        "palle_g10": palle_meta,
    })
    if gen.moltiplicatori_attivi():
        # bound multipliers on: stored in the zone content (and hash); when off the
        # key is absent and the hash is unchanged
        meta["moltiplicatori"] = _plain(gen.moltiplicatori_attivi())
    return meta, arrays, tr, gen


def palle_g10(xyz, seme, metro, gen, attesi):
    """Gene 10 zones: balls as in _apply_cluster_perturbation, on the snapshot.
    Radius: 10th percentile of distances from the centroid. Centres: points in
    random order (local generator from seed, segment, 10, 2), the first n_clusters
    plus more until the balls hold RISERVA_G10 * attesi points (at most
    MAX_PALLE * n_clusters). Returns the voxel keys (anchor frame) of the points in
    the balls, the centres in priority order and a description."""
    from scipy.spatial import cKDTree
    n = len(xyz)
    n_c = min(gen.n_clusters, n // 100)
    if n_c <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 3)), {"n_centri": 0}
    rng = np.random.default_rng(np.random.SeedSequence([int(seme), int(metro), 10, 2]))
    perm = rng.permutation(n)
    r = float(np.percentile(np.linalg.norm(xyz - xyz.mean(axis=0), axis=1), 10))
    tree = cKDTree(xyz)
    dentro = np.zeros(n, dtype=bool)
    usati = 0
    for j in range(min(MAX_PALLE * n_c, n)):
        if j >= n_c and dentro.sum() >= RISERVA_G10 * attesi:
            break
        dentro[tree.query_ball_point(xyz[perm[j]], r)] = True
        usati = j + 1
    centri = xyz[perm[:usati]].astype(np.float64)
    chiavi = np.unique(st.voxel_keys(xyz[dentro]))
    return chiavi, centri, {"n_centri": int(usati), "n_clusters": int(n_c), "raggio_m": r,
                            "punti_fotografia": int(dentro.sum()), "attesi": int(attesi), "voxel": int(len(chiavi))}


def zone_genoma(dir_zone, foto: Fotografia, genoma, metro: int, limiti: Dict, test_seed: int = 0):
    """Zones of the genome in the segment: computed and saved to w{k}_<sha>.npz, or,
    if the file exists (same genome, same segment), read back and checked against
    the computation. Returns the Zone (with path, sha256, hash)."""
    meta, arrays, _, _ = calcola_zone(foto, genoma, metro, limiti, test_seed)
    path = Path(dir_zone) / f"w{int(metro)}_{meta['genoma_sha']}.npz"
    h = hash_contenuto(meta, arrays)
    if path.exists():
        m2, a2, sha, h2 = _leggi(path)
        if h2 != h:
            raise ZoneNonValide("hash_diverso", f"{path} esiste con un altro contenuto")
        return Zone(m2, a2, sha, h2, path)
    sha, h = _scrivi(path, meta, arrays)
    return Zone(meta, arrays, sha, h, path)


def carica_zone(path, sha256: Optional[str] = None, hash_atteso: Optional[str] = None,
                cache: Optional[Dict] = None) -> Zone:
    """Load and check zones (file sha256, content hash, snapshot), with the
    alignment target built from the snapshot (cached by snapshot sha256)."""
    meta, arrays, sha, h = _leggi(path, sha256, hash_atteso)
    if meta.get("tipo") != "zone":
        raise ZoneNonValide("illeggibile", f"{path}: non e' un file di zone")
    fsha = meta.get("fotografia_sha256")
    cache = {} if cache is None else cache
    if fsha not in cache:
        fp = Path(path).parent / str(meta.get("fotografia_file"))
        try:
            foto = carica_fotografia(fp, fsha, meta.get("fotografia_hash"))
        except ZoneNonValide as e:
            raise ZoneNonValide("fotografia_" + e.stato, str(e))
        cache.clear()     # one snapshot per segment
        cache[fsha] = bersaglio_fotografia(foto)
    return Zone(meta, arrays, sha, h, path, bersaglio=cache[fsha])
