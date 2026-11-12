"""
Perturbation generator for LiDAR point clouds.

A genome of 17 values in [-1, 1] is decoded into 13 perturbation parameters
(encode_perturbation), which drive nine operators applied in sequence to every
scan (apply_perturbation): per-point noise with directional bias, cluster
displacement, density-weighted dropout, ghost point injection (random or near
geometric features), global geometric distortion, edge/corner attack, temporal
drift, and scanline shift along the beam direction.

Physical bounds are centimeter-scale (default: 5 cm per-point shift, 8 cm on
edge points, 5 cm drift per frame, at most 3% dropout and 2% ghost points).
Perceptibility is measured with the bidirectional Chamfer distance combined
with a structural penalty for the change in point count
(compute_perturbation_magnitude), expressed in cm. The structural term has two
forms (structural): "net" (default) r = |M - N| / N, and "gross"
r = (removed + added) / N, with the counts of the last apply_perturbation.

References:
- FLAT: Flux-Aware Imperceptible Adversarial Attacks (ECCV 2024)
- SLACK: Attacking LiDAR-based SLAM (arXiv 2024)
- Adversarial attack on ICP-based localization (arXiv 2403.05666)
- ASP: Attribution-based Scanline Perturbation (IEEE 2024)
- Survey on adversarial robustness of LiDAR-based ML (2024)
"""

import copy
import hashlib
import math
import weakref
from typing import Dict, Optional

import numpy as np
from scipy.spatial import cKDTree


# Plausibility bounds, independent of the genome: an attacker that replaces the
# cloud is not realistic, so the ghost offset from its base point and the
# removable fraction are capped.
MAX_GHOST_OFFSET_M = 0.10   # max distance of a ghost point from its base point (m)
MAX_DROPOUT_RATE = 0.05     # max fraction of removable points
# Reference speed used to express the temporal drift cap as a delay: an
# accumulated drift of |d| m equals a scan late by |d| / DRIFT_REF_SPEED_MPS s
# (follower cruise speed). The cap acts on the accumulator norm; the direction
# stays the genome's. Without a cap the steady state is
# 0.05 * strength / (1 - 0.98) = 2.5 m.
DRIFT_REF_SPEED_MPS = 0.8
# Geometric distortion caps. Yaw and scaling have no physical plausibility
# bound (perceptibility is measured by the Chamfer objective). Smaller caps
# (3 deg, 3 %) stay inside the band that ICP corrects, leaving the operator
# inert (~1 deg / 1.8 cm induced); these values can actually misalign the
# registration.
GEOM_MAX_YAW_RAD = 0.105     # ~6 deg, beyond the band GICP corrects
GEOM_MAX_SCALE = 0.06        # 6 %
# Wavenumber of the systematic scanline component, over the beam elevation
# angle (not the array index): points on the same scanline get the same radial
# shift. The XT-32 covers ~31 deg with 32 channels.
SCANLINE_ELEV_WAVES = 16.0
# Forms of the structural perceptibility term: "net" counts only the change in
# point count |M - N| (removing and adding the same number costs zero), "gross"
# counts removed plus added points.
STRUCTURAL_MODES = ("net", "gross")
# Operator modes. "random": random draws seeded per cloud (the node passes
# hash(genome) + frame). "coherent": no random draws; the operators that choose
# where to strike (genes 5, 6, 10, 13, 16) hit whole structures, chosen on the
# first cloud of the meter and tracked on the following ones
# (src/perturbations/structures.py).
OPERATOR_MODES = ("random", "coherent")
# Coherent mode with struct_g13_punteggio: at most this many points moved by
# gene 13, chosen by edge score (same cap as random mode).
G13_MAX_PUNTI = 500
# Genome ramp (node option --rampa-genoma, off by default): on the R - 1 mixed
# clouds of the meter (k = 0..R-2) the previous genome P and the new one N are
# applied together, each with its own operators, zones and seed, weighted
# w_P = 1 - alpha_k and w_N = alpha_k, alpha_k = min(1, (k + 1) / R); from
# k = R - 1 (alpha = 1) only N. k is the tracker index (clouds processed since
# the zones were installed, 0 = first). The anchors of P live in the same
# tracker with this prefix.
RAMPA_PREFISSO = "p:"
# Genome in motion (imposta_gradino, off by default): step ramp. From k = 0 the
# previous genome P alone (with its zones on the meter snapshot, anchors with
# RAMPA_PREFISSO in the same tracker as N); N enters on the first cloud with
# k >= kmin and tracker displacement |p_k - p_rif| >= S (cm), or at k = kmax
# (fallback, state "k_max"). p_rif is the tracker pose on the first cloud of the
# continuous stretch of the stream (k = 0, or the cloud after
# riprendi_riferimento_moto). Per-cloud phases: "P", "ingresso" (first cloud
# with N), "N"; "metro0" without P (meter 0).
GRADINO_FASI = ("P", "ingresso", "N", "metro0")
# Limit multiplier for the damage-vs-strength study (off by default: every
# family at 1). Each family scales its own limits; directions (genes 0-2, 7-9)
# and the modulators g4, g11 are not scaled:
#   deriva       g14: per-cloud step (max_temporal_drift) and cap (max_drift_m)
#   rumore       g3: per-point intensity and clip (max_point_shift)
#   geometria    g12: range bias 5 cm, yaw GEOM_MAX_YAW_RAD, scale GEOM_MAX_SCALE,
#                constant bias 2 cm; g15: scanline amplitudes (2 cm, 3 cm in random mode)
#   strutturale  g6/g16: ghost fraction (max_ghost_points_ratio; offset unchanged);
#                g10: cluster shift (cluster_shift_std); g13: edge shift
#                (max_edge_shift) and the G13_MAX_PUNTI cap
#   dropout      g5: decoding cap (MAX_DROPOUT_RATE) and minimum kept fraction
#                (90 % -> 1 - 0.10 K), bounded by MAX_DROPOUT_FISICO
FAMIGLIE_LIMITI = ("deriva", "rumore", "geometria", "strutturale", "dropout")
# Physical stop of the dropout under the multiplier: never more than this
# fraction of the scan (at x1 the cap is 10 %, so it never triggers).
MAX_DROPOUT_FISICO = 0.5


def moltiplicatori_da_opzioni(k, famiglie=None):
    """{family: K} from the options --moltiplicatore-limiti K and --famiglie-limiti
    (string "a,b,c" or sequence; None = all). K finite real >= 1; families from
    FAMIGLIE_LIMITI, no repetitions. With K = 1 the dict is empty (off)."""
    try:
        k = float(k)
    except (TypeError, ValueError):
        raise ValueError(f"moltiplicatore dei limiti non numerico: {k!r}")
    if not math.isfinite(k) or k < 1.0:
        raise ValueError(f"moltiplicatore dei limiti: serve un numero >= 1 (dato {k!r})")
    if famiglie is None:
        fam = list(FAMIGLIE_LIMITI)
    else:
        fam = [f.strip() for f in (famiglie.split(",") if isinstance(famiglie, str) else famiglie)]
        if not fam or any(not f for f in fam):
            raise ValueError(f"famiglie dei limiti vuote o con un elemento vuoto: {famiglie!r}")
        ignote = [f for f in fam if f not in FAMIGLIE_LIMITI]
        if ignote:
            raise ValueError(f"famiglie dei limiti sconosciute: {ignote} (ammesse: {', '.join(FAMIGLIE_LIMITI)})")
        if len(set(fam)) != len(fam):
            raise ValueError(f"famiglie dei limiti ripetute: {famiglie!r}")
    return {} if k == 1.0 else {f: k for f in FAMIGLIE_LIMITI if f in fam}


def rampa_alfa(k: int, R: int) -> float:
    """Weight of the new genome on cloud k of the ramp."""
    return min(1.0, (int(k) + 1) / int(R))


def rampa_conta(m: int, R: int, T: int) -> int:
    """Points removed or added with weight m / R on a target of T points:
    floor(m T / R + 1/2) in integer arithmetic, (2 m T + R) // (2 R), i.e.
    round half up, identical in every process. m = R gives T, m = 0 gives 0;
    T <= 0 gives 0."""
    T = int(T)
    if T <= 0:
        return 0
    return (2 * int(m) * T + int(R)) // (2 * int(R))


def impronta_stato(obj) -> str:
    """sha256 of a stream state (PerturbationGenerator.stato_flusso): dicts by
    key, arrays by dtype, shape and bytes, numbers by exact repr; zones by their
    content hash, KD-trees skipped (derived from the points stored alongside)."""
    h = hashlib.sha256()
    _impronta(obj, h)
    return h.hexdigest()


def _impronta(o, h):
    if o is None or isinstance(o, (bool, int, float, str, np.bool_, np.integer, np.floating)):
        h.update(f"{type(o).__name__}:{o!r};".encode())
    elif isinstance(o, np.ndarray):
        a = np.ascontiguousarray(o)
        h.update(f"nd:{a.dtype.str}:{a.shape};".encode())
        h.update(a.tobytes())
    elif isinstance(o, dict):
        h.update(b"{")
        for k in sorted(o, key=repr):
            h.update(f"{k!r}=".encode())
            _impronta(o[k], h)
        h.update(b"}")
    elif isinstance(o, (list, tuple)):
        h.update(b"[")
        for x in o:
            _impronta(x, h)
        h.update(b"]")
    elif isinstance(o, cKDTree):
        h.update(b"kdtree;")
    elif type(o).__name__ == "Zone":
        h.update(f"zone:{o.hash};".encode())
    elif hasattr(o, "__dict__"):
        h.update(f"<{type(o).__name__}>".encode())
        _impronta(vars(o), h)
    else:
        h.update(repr(o).encode())


def _WQ():
    """Query workers for detect_edges_and_corners: structures.WORKERS with the
    O9 optimization, otherwise -1 (all cores)."""
    from src.perturbations import structures as _s
    return _s.WORKERS if "O9" in _s.OTT else -1


def _clip_offsets(offsets):
    """Clip the norm of each offset to MAX_GHOST_OFFSET_M."""
    import numpy as _np
    norm = _np.linalg.norm(offsets, axis=1, keepdims=True)
    scale = _np.minimum(1.0, MAX_GHOST_OFFSET_M / _np.maximum(norm, 1e-9))
    return offsets * scale


class PerturbationGenerator:
    """
    Adversarial perturbation generator for LiDAR point clouds.

    Applies per-point perturbations within centimeter-scale bounds and, when
    enabled, concentrates them on high-curvature regions used by SLAM feature
    extraction.
    """

    def __init__(
        self,
        # Per-point perturbation bounds (in meters)
        max_point_shift: float = 0.05,  # 5 cm max per-point displacement
        # Noise parameters
        noise_std: float = 0.02,  # 2 cm Gaussian noise std
        # Feature targeting
        target_high_curvature: bool = True,
        curvature_percentile: float = 90.0,  # Target top 10% curvature points
        # Point manipulation (kept small for MOLA stability)
        max_dropout_rate: float = 0.03,  # Max 3% point removal
        max_ghost_points_ratio: float = 0.02,  # Max 2% ghost points added
        # Cluster perturbation
        cluster_shift_std: float = 0.03,  # 3 cm cluster displacement std
        n_clusters: int = 5,  # Number of perturbation clusters
        # Edge attack and temporal drift
        max_edge_shift: float = 0.08,  # 8 cm max shift for edge points
        max_temporal_drift: float = 0.05,  # 5 cm max accumulated drift per frame
        max_drift_lag_s: float = float("inf"),  # cap on accumulated drift (s)
        structural: str = "net",  # structural term: "net" or "gross"
        fast_ops: bool = False,  # vectorized voxel grouping and edge PCA
        operator_mode: str = "random",  # "random" or "coherent"
        struct_track: str = "scan",     # coherent: tracker on scan (no pose) or odom
        struct_rule: str = "R1",        # coherent: choice by size (R1), azimuth (R2) or caso
        struct_dropout: str = "a",      # coherent: a = thinning, b = whole voxels
        struct_cluster_type: str = "curvatura",  # coherent: structure type of gene 10
        lidar_pose=None,                # coherent: sensor pose on the base (x, y, z)
        struct_g10_pavimento: bool = False,  # coherent, rule caso: g10 skips floor structures
        struct_ripiego: bool = False,   # rule caso: no candidate of the type -> any non-floor type
        struct_g13_punteggio: bool = False,  # coherent: g13 on the zone points with top edge score
        struct_insieme: bool = False,   # coherent, rule caso: fixed zone set per operator
        zone_attaccante: bool = False,  # insieme: zones from zone_fornitore, no in-stream choice
        moltiplicatori: Optional[Dict[str, float]] = None,  # {family: K >= 1}; None = all at 1
    ):
        """
        Initialize the perturbation generator.

        Args:
            max_point_shift: Maximum displacement per point in meters (default: 5 cm)
            noise_std: Standard deviation of Gaussian noise in meters (default: 2 cm)
            target_high_curvature: Whether to target high-curvature regions
            curvature_percentile: Percentile threshold for high-curvature points
            max_dropout_rate: Maximum fraction of points to remove
            max_ghost_points_ratio: Maximum ratio of ghost points to add
            cluster_shift_std: Std of cluster-based displacement
            n_clusters: Number of perturbation clusters
            max_edge_shift: Maximum shift for detected edge points
            max_temporal_drift: Maximum accumulated drift per frame
            structural: Form of the structural term (STRUCTURAL_MODES)
            fast_ops: Voxel grouping with a 1D key and blockwise edge PCA; same
                result bit for bit, less time holding the GIL
        """
        if structural not in STRUCTURAL_MODES:
            raise ValueError(f"structural sconosciuto: {structural!r} (net o gross)")
        self.structural = structural
        # Points removed and added by the last apply_perturbation, tied to its
        # output: (weak reference to the returned array, counts).
        self._last_counts = None
        self.max_point_shift = max_point_shift
        self.noise_std = noise_std
        self.target_high_curvature = target_high_curvature
        self.curvature_percentile = curvature_percentile
        # Never beyond the plausibility bound, whatever the requested value.
        self.max_dropout_rate = min(max_dropout_rate, MAX_DROPOUT_RATE)
        self.max_ghost_points_ratio = max_ghost_points_ratio
        self.cluster_shift_std = cluster_shift_std
        self.n_clusters = n_clusters
        self.max_edge_shift = max_edge_shift
        self.max_temporal_drift = max_temporal_drift
        # Per-family limit multiplier (FAMIGLIE_LIMITI). The public attributes
        # above keep the given limits (node status gen_limits, attacker zones);
        # the operators use the effective limits _lim_*, equal to the given ones
        # at x1 (product by 1.0, exact).
        self.moltiplicatori = {f: 1.0 for f in FAMIGLIE_LIMITI}
        for f, v in (moltiplicatori or {}).items():
            if f not in FAMIGLIE_LIMITI:
                raise ValueError(f"famiglia dei limiti sconosciuta: {f!r}")
            v = float(v)
            if not math.isfinite(v) or v < 1.0:
                raise ValueError(f"moltiplicatore della famiglia {f}: serve un numero >= 1 (dato {v!r})")
            self.moltiplicatori[f] = v
        kd, kr, kg, ks, kx = (self.moltiplicatori[f] for f in FAMIGLIE_LIMITI)
        self._lim_drift_step = max_temporal_drift * kd
        self._lim_point_shift = max_point_shift * kr
        self._lim_geo_range = 0.05 * kg
        self._lim_geo_yaw = GEOM_MAX_YAW_RAD * kg
        self._lim_geo_scala = GEOM_MAX_SCALE * kg
        self._lim_geo_bias = 0.02 * kg
        self._lim_scan_sist = 0.02 * kg
        self._lim_scan_rand = 0.03 * kg
        self._lim_ghost_ratio = max_ghost_points_ratio * ks
        self._lim_cluster_std = cluster_shift_std * ks
        self._lim_edge_shift = max_edge_shift * ks
        self._lim_g13_punti = G13_MAX_PUNTI if ks == 1.0 else int(G13_MAX_PUNTI * ks)
        self._lim_dropout_rate = min(self.max_dropout_rate * kx, MAX_DROPOUT_FISICO)
        # minimum kept fraction: 90 % at x1, 1 - 0.10 K above, never below the physical stop
        self._min_keep = 0.90 if kx == 1.0 else max(1.0 - MAX_DROPOUT_FISICO, 1.0 - 0.10 * kx)
        # Cap on the accumulated drift norm, m (inf = none), already scaled by
        # the deriva family.
        self.max_drift_m = max_drift_lag_s * DRIFT_REF_SPEED_MPS * kd
        # Temporal state for the drift operator (persists across frames)
        self._accumulated_drift = np.zeros(3)
        self._frame_counter = 0
        # Kept drift (node keep mode): with a genome without drift the
        # accumulated drift keeps decaying and being applied instead of
        # vanishing at once.
        self.keep_drift = False
        self.fast_ops = bool(fast_ops)
        if operator_mode not in OPERATOR_MODES:
            raise ValueError(f"operator_mode sconosciuto: {operator_mode}")
        self.operator_mode = operator_mode
        self._structs = None
        self.last_struct_info = None
        self.last_hits = None
        self.struct_insieme = bool(struct_insieme)
        self._obiettivi = None
        if self.struct_insieme and operator_mode != "coherent":
            raise ValueError("struct_insieme richiede operator_mode coherent")
        # Attacker zones: on the first cloud after the genome the tracker gets
        # the zones from zone_fornitore(seed, meter, params), which returns
        # (zones or None, state); with None no zone and no choice.
        self.zone_attaccante = bool(zone_attaccante)
        if self.zone_attaccante and not self.struct_insieme:
            raise ValueError("zone_attaccante richiede struct_insieme")
        self.zone_fornitore = None
        self.zone_stato = None
        self.zone_hash = None
        # Intensity matched to random mode (only with valid installed zones):
        # expected removed quota and gene 10 balls, from the snapshot.
        self._zone_int = None
        self._riserva_scelta = None   # only while computing the attacker zones
        # Genome ramp (imposta_rampa, attacker zones only): spec for the meters
        # of the current genome, ramp in progress, status info. Without
        # imposta_rampa these stay None and coherent mode is unchanged.
        self.rampa_spec = None
        self.rampa_info = None
        self._rampa = None
        self._rampa_info_metro = None
        self._zone_n = None           # installed zones of the current genome (ramp checks)
        # Genome in motion (imposta_gradino, attacker zones only): spec and
        # state of the meter step. None = off, coherent mode and stream state
        # unchanged.
        self._gradino = None
        if operator_mode == "coherent":
            from src.perturbations import structures as _st
            if struct_dropout not in ("a", "b"):
                raise ValueError(f"struct_dropout sconosciuto: {struct_dropout}")
            if struct_cluster_type not in ("curvatura", "parete"):
                raise ValueError(f"struct_cluster_type sconosciuto: {struct_cluster_type}")
            self._st = _st
            self.struct_dropout = struct_dropout
            self.struct_g13_punteggio = bool(struct_g13_punteggio)
            self.struct_cluster_type = _st.TYPE_BY_NAME[struct_cluster_type]
            self._structs = _st.StructureTracker(
                track=struct_track, rule=struct_rule,
                lidar_pose=_st.LIDAR_POSE if lidar_pose is None else lidar_pose,
                escludi_pavimento=struct_g10_pavimento, ripiego=struct_ripiego,
                insieme=self.struct_insieme)

    def moltiplicatori_attivi(self):
        """Families with multiplier other than 1, {family: K}; None if off
        (run record, node status, zone metadata)."""
        att = {f: v for f, v in self.moltiplicatori.items() if v != 1.0}
        return att or None

    def ultimi_conteggi(self):
        """Point counts of the last apply_perturbation, measured on the clouds:
        n_in (input), n_removed (input minus points left after dropout), n_added
        (ghosts) and frac_removed = n_removed / n_in. None before the first call."""
        if self._last_counts is None:
            return None
        c = dict(self._last_counts[1])
        c["frac_removed"] = c["n_removed"] / c["n_in"] if c["n_in"] > 0 else 0.0
        return c

    def get_genome_size(self) -> int:
        """
        Return the size of the genome encoding.

        Genome structure (17 parameters):
        - [0-2]: Directional bias for per-point noise (normalized direction)
        - [3]: Noise intensity scale [0, 1]
        - [4]: Curvature targeting strength [0, 1]
        - [5]: Point dropout rate [0, 1]
        - [6]: Ghost points ratio [0, 1]
        - [7-9]: Cluster perturbation direction
        - [10]: Cluster perturbation strength [0, 1]
        - [11]: Spatial correlation of perturbations [0, 1]
        - [12]: Geometric distortion strength [0, 1] (ICP attack)
        - [13]: Edge attack strength [0, 1] (edges/corners, SLACK)
        - [14]: Temporal drift strength [0, 1] (accumulating drift, ICP attack)
        - [15]: Scanline perturbation [0, 1] (ASP)
        - [16]: Strategic ghost placement [0, 1] (ghosts near features)
        """
        return 17

    def encode_perturbation(self, genome: np.ndarray) -> Dict[str, any]:
        """
        Decode a genome into perturbation parameters.

        Args:
            genome: Normalized parameters in range [-1, 1]

        Returns:
            Dictionary with perturbation parameters
        """
        # Rates are mapped to [0, 1]; directions stay in [-1, 1]
        genome = np.clip(genome, -1, 1)

        # Directional bias for noise (unit vector)
        noise_direction = genome[0:3]
        noise_direction_norm = np.linalg.norm(noise_direction)
        if noise_direction_norm > 0:
            noise_direction = noise_direction / noise_direction_norm

        # Noise intensity [0, 1] -> [0, max_point_shift]; zero disables the noise
        noise_intensity = (genome[3] + 1) / 2 * self._lim_point_shift

        # Curvature targeting strength [0, 1]
        curvature_strength = (genome[4] + 1) / 2

        # Dropout rate [0, max_dropout_rate]; zero disables the dropout
        dropout_rate = (genome[5] + 1) / 2 * self._lim_dropout_rate

        # Ghost points ratio [0, max_ghost_points_ratio]
        ghost_ratio = (genome[6] + 1) / 2 * self._lim_ghost_ratio

        # Cluster perturbation direction (unit vector)
        cluster_direction = genome[7:10]
        cluster_dir_norm = np.linalg.norm(cluster_direction)
        if cluster_dir_norm > 0:
            cluster_direction = cluster_direction / cluster_dir_norm

        # Cluster strength [0, 1]
        cluster_strength = (genome[10] + 1) / 2

        # Spatial correlation [0, 1]: how correlated nearby point perturbations are
        spatial_correlation = (genome[11] + 1) / 2

        # Geometric distortion [0, 1]: systematic distortions targeting ICP convergence
        geometric_distortion = (genome[12] + 1) / 2  # Maps [-1,1] -> [0, 1]

        # Parameters of the SLACK / ICP attack / ASP inspired operators.
        # The length checks keep older 13-gene genomes decodable.
        # Edge attack strength [0, 1]: targets edges/corners used by ICP
        edge_attack_strength = (genome[13] + 1) / 2 if len(genome) > 13 else 0.0

        # Temporal drift [0, 1]: bias accumulating across frames
        temporal_drift_strength = (genome[14] + 1) / 2 if len(genome) > 14 else 0.0

        # Scanline perturbation [0, 1]: shift along the laser beam (ASP)
        scanline_strength = (genome[15] + 1) / 2 if len(genome) > 15 else 0.0

        # Strategic ghost placement [0, 1]: ghosts placed near geometric features
        strategic_ghost = (genome[16] + 1) / 2 if len(genome) > 16 else 0.0

        params = {
            "noise_direction": noise_direction,
            "noise_intensity": noise_intensity,
            "curvature_strength": curvature_strength,
            "dropout_rate": dropout_rate,
            "ghost_ratio": ghost_ratio,
            "cluster_direction": cluster_direction,
            "cluster_strength": cluster_strength,
            "spatial_correlation": spatial_correlation,
            "geometric_distortion": geometric_distortion,
            "edge_attack_strength": edge_attack_strength,
            "temporal_drift_strength": temporal_drift_strength,
            "scanline_strength": scanline_strength,
            "strategic_ghost": strategic_ghost,
        }
        # Limit multiplier: the activation thresholds look at the x1 value (the
        # multiplier changes strength, never which operators turn on). The
        # <name>_x1 keys exist only for scaled families, so with the multiplier
        # off the parameters, and the zone signature, are unchanged.
        if self.moltiplicatori["rumore"] != 1.0:
            params["noise_intensity_x1"] = (genome[3] + 1) / 2 * self.max_point_shift
        if self.moltiplicatori["dropout"] != 1.0:
            params["dropout_rate_x1"] = (genome[5] + 1) / 2 * self.max_dropout_rate
        if self.moltiplicatori["strutturale"] != 1.0:
            params["ghost_ratio_x1"] = (genome[6] + 1) / 2 * self.max_ghost_points_ratio
        return params

    @staticmethod
    def _x1(p, chiave):
        """x1 value of a decoded parameter, for activation thresholds: <key>_x1
        when its family multiplier is on, otherwise the parameter itself."""
        return p.get(chiave + "_x1", p[chiave])

    def operatori_accesi(self, p):
        """Operators active for parameters p, with the operators' own thresholds
        (x1 values for noise, dropout, ghosts): sorted tuple of genes (g4 and g11
        only modulate g3; g6 or g16 depending on strategic placement)."""
        on = []
        if self._x1(p, "noise_intensity") > 0.001:
            on.append("g3")
            if self.target_high_curvature and p["curvature_strength"] > 0.1:
                on.append("g4")
            if p["spatial_correlation"] >= 0.3:
                on.append("g11")
        if self._x1(p, "dropout_rate") > 0.01:
            on.append("g5")
        if self._x1(p, "ghost_ratio") > 0.01:
            on.append("g16" if p.get("strategic_ghost", 0) > 0.5 else "g6")
        if p["cluster_strength"] > 0.1:
            on.append("g10")
        if p.get("geometric_distortion", 0.0) >= 0.01:
            on.append("g12")
        if p.get("edge_attack_strength", 0) > 0.1:
            on.append("g13")
        if p.get("temporal_drift_strength", 0) > 0.1:
            on.append("g14")
        if p.get("scanline_strength", 0) > 0.1:
            on.append("g15")
        return tuple(sorted(on, key=lambda g: int(g[1:])))

    def compute_curvature(self, points: np.ndarray, k: int = 10) -> np.ndarray:
        """
        Compute an approximate local curvature for each point.

        The curvature (smallest eigenvalue over the sum of eigenvalues of the
        local covariance) is computed on a random sample of at most 1000 points
        and propagated to every point from its nearest sampled neighbour.

        Args:
            points: Point cloud (N, 3+) XYZ coordinates
            k: Number of nearest neighbors

        Returns:
            Curvature values for each point (N,)
        """
        n_points = len(points)
        if n_points < k + 1:
            return np.zeros(n_points)

        # Small sample for speed (1000 points max)
        sample_size = min(n_points, 1000)
        sample_indices = np.random.choice(n_points, sample_size, replace=False)
        sample_points = points[sample_indices, :3]

        # KD-tree on the sampled points
        tree = cKDTree(sample_points)

        k_use = min(k, sample_size - 1)

        # Batch query for all sample points (workers=-1 uses all cores)
        _, all_neighbors = tree.query(sample_points, k=k_use + 1, workers=-1)

        # Vectorised over the sample: same covariance/eigenvalue computation as a
        # per-point loop, batched. Covariance uses ddof=1 (divide by k_use).
        sample_curvatures = np.zeros(sample_size)
        if k_use + 1 > 3:
            nb = sample_points[all_neighbors]                     # (S, k+1, 3)
            centered = nb - nb.mean(axis=1, keepdims=True)
            m = centered.shape[1]
            cov = np.einsum("sij,sik->sjk", centered, centered) / (m - 1)
            eigenvalues = np.linalg.eigvalsh(cov)                  # (S,3) ascending
            total = eigenvalues.sum(axis=1)
            ok = total > 0
            # eigenvalues[:, 0] is the smallest eigenvalue
            sample_curvatures[ok] = eigenvalues[ok, 0] / total[ok]

        # Assign curvature to all points from the nearest sampled point
        _, nearest = tree.query(points[:, :3], k=1, workers=-1)
        curvatures = sample_curvatures[nearest]

        return curvatures

    def detect_edges_and_corners(self, points: np.ndarray, k: int = 15) -> np.ndarray:
        """
        Detect edge and corner points using eigenvalue analysis (SLACK).

        Classification by eigenvalue ratios:
        planar l1 ~ l2 >> l3 (surfaces), edge l1 >> l2 ~ l3 (lines),
        corner l1 ~ l2 ~ l3 (3D features).

        Args:
            points: Point cloud (N, 3+)
            k: Number of neighbors for local analysis

        Returns:
            Edge scores for each point (N,), higher = more edge-like
        """
        n_points = len(points)
        if n_points < k + 1:
            return np.zeros(n_points)

        # Sample for speed
        sample_size = min(n_points, 2000)
        sample_indices = np.random.choice(n_points, sample_size, replace=False)
        sample_points = points[sample_indices, :3]

        tree = cKDTree(sample_points)
        edge_scores = np.zeros(sample_size)

        _, all_neighbors = tree.query(sample_points, k=k + 1, workers=_WQ())

        # Vectorised over the sample: same covariance/eigenvalue computation as a
        # per-point loop, batched.
        #   linearity  = (l1 - l2) / l1        with l1 >= l2 >= l3
        #   sphericity = l3 / l1
        #   score      = linearity + 0.5 * sphericity
        # eigvalsh returns ascending eigenvalues, hence the reversal to [l1, l2, l3].
        if k + 1 > 3:
            nb = sample_points[all_neighbors]                      # (S, k+1, 3)
            centered = nb - nb.mean(axis=1, keepdims=True)
            m = centered.shape[1]
            cov = np.einsum("sij,sik->sjk", centered, centered) / (m - 1)
            ev = np.linalg.eigvalsh(cov)[:, ::-1]                  # descending
            l1 = ev[:, 0] + 1e-10
            linearity = (ev[:, 0] - ev[:, 1]) / l1
            sphericity = ev[:, 2] / l1
            edge_scores = linearity + sphericity * 0.5

        # Assign to all points
        _, nearest = tree.query(points[:, :3], k=1, workers=_WQ())
        all_edge_scores = edge_scores[nearest]

        return all_edge_scores

    def _compute_perturbation_weights(self, perturbed, n_points, params):
        """Compute curvature-based per-point weights (1.0 on targeted points, 0.3 elsewhere)."""
        if not self.target_high_curvature or params["curvature_strength"] <= 0.1:
            return np.ones(n_points)

        curvatures = self.compute_curvature(perturbed[:, :3])
        if curvatures.max() > curvatures.min():
            curvature_weights = (curvatures - curvatures.min()) / (
                curvatures.max() - curvatures.min()
            )
        else:
            return np.ones(n_points)

        threshold = np.percentile(
            curvature_weights, 100 - self.curvature_percentile * params["curvature_strength"]
        )
        return np.where(curvature_weights >= threshold, 1.0, 0.3)

    def _apply_noise(self, perturbed, n_points, perturbation_weights, params):
        """
        Apply per-point Gaussian noise with a directional bias (FLAT-style per-point shift).

        noise_intensity sets the magnitude in [0, max_point_shift]; noise_direction
        adds a bias of 30% of the intensity. Each displacement is clipped to
        max_point_shift (default 5 cm).
        """
        if self._x1(params, "noise_intensity") <= 0.001:
            return perturbed

        # Noise scaled by noise_intensity rather than the fixed noise_std, so that
        # genome[3] = -1 gives no noise and genome[3] = 1 gives the full max_point_shift
        noise = np.random.randn(n_points, 3) * params["noise_intensity"]

        if params["spatial_correlation"] > 0.1:
            noise = self._apply_spatial_correlation(
                perturbed[:, :3], noise, params["spatial_correlation"]
            )

        # Directional bias (30% of intensity in the specified direction)
        directional_component = params["noise_direction"] * params["noise_intensity"] * 0.3
        noise += directional_component
        noise *= perturbation_weights[:, np.newaxis]

        # Clip to max_point_shift
        noise_norms = np.linalg.norm(noise, axis=1, keepdims=True)
        noise = np.where(
            noise_norms > self._lim_point_shift,
            noise / noise_norms * self._lim_point_shift,
            noise,
        )
        perturbed[:, :3] += noise
        return perturbed

    def _apply_dropout(self, perturbed, n_points, perturbation_weights, params):
        """
        Remove points with a density-weighted probability.

        dropout_rate sets the base removal rate (at most max_dropout_rate, 3%);
        denser regions get up to 10% extra removal, since SLAM matching relies on
        dense geometric structure. At least 90% of the points are always kept.
        """
        if self._x1(params, "dropout_rate") <= 0.01:
            return perturbed

        # Local density from k-nearest neighbours
        k = min(20, n_points - 1)
        if k > 0:
            tree = cKDTree(perturbed[:, :3])
            # Most expensive step of the generator: a full-cloud neighbour query
            # used only to estimate local density. workers=-1 parallelises it.
            distances, _ = tree.query(perturbed[:, :3], k=k + 1, workers=-1)
            # Local density = inverse of mean distance to k neighbours
            local_density = 1.0 / (distances[:, 1:].mean(axis=1) + 1e-6)
            # Normalise to [0, 1]
            density_weights = (local_density - local_density.min()) / (
                local_density.max() - local_density.min() + 1e-6
            )
        else:
            density_weights = np.ones(n_points)

        # Higher density -> lower keep probability, while staying close to the
        # requested dropout rate (at most 10% extra removal for the densest points)
        keep_prob_base = 1 - params["dropout_rate"]
        keep_prob_per_point = keep_prob_base * (1.0 - 0.1 * density_weights)
        keep_mask = np.random.random(n_points) < keep_prob_per_point

        # Keep at least 90% of the points to bound perceptibility
        # (1 - 0.10 K with the dropout multiplier, never below the physical stop)
        min_keep_ratio = max(self._min_keep, 1 - params["dropout_rate"] * 2)
        if keep_mask.sum() < n_points * min_keep_ratio:
            keep_mask = np.random.random(n_points) < min_keep_ratio

        return perturbed[keep_mask]

    def _add_ghost_points(self, perturbed, params):
        """
        Append ghost points to the cloud.

        ghost_ratio sets how many (at most max_ghost_points_ratio, 2%);
        strategic_ghost > 0.5 places them near geometric features (SLACK),
        otherwise they are placed around random points.
        """
        if self._x1(params, "ghost_ratio") <= 0.01 or len(perturbed) == 0:
            return perturbed

        n_ghost = int(len(perturbed) * params["ghost_ratio"])
        if n_ghost > 0:
            strategic = params.get("strategic_ghost", 0)
            if strategic > 0.5:
                ghost_points = self._generate_strategic_ghost_points(perturbed, n_ghost, params)
            else:
                ghost_points = self._generate_ghost_points(perturbed, n_ghost)
            perturbed = np.vstack([perturbed, ghost_points])
        return perturbed

    def _generate_strategic_ghost_points(
        self, point_cloud: np.ndarray, n_ghost: int, params: Dict[str, any]
    ) -> np.ndarray:
        """
        Generate ghost points near edges and corners (SLACK: placement matters more
        than quantity).

        Bases are drawn from the top 30% edge scores and offset by Gaussian noise
        with 2.5 cm std, so that the ghosts create ambiguous ICP correspondences.
        """
        edge_scores = self.detect_edges_and_corners(point_cloud)

        # High-feature points as bases for the ghosts
        threshold = np.percentile(edge_scores, 70)
        feature_mask = edge_scores >= threshold
        feature_indices = np.where(feature_mask)[0]

        if len(feature_indices) < n_ghost:
            feature_indices = np.arange(len(point_cloud))

        base_indices = np.random.choice(feature_indices, n_ghost, replace=True)
        ghost_points = point_cloud[base_indices].copy()

        # Small offsets: close enough to real features to create ambiguous matches
        offsets = _clip_offsets(np.random.randn(n_ghost, 3) * 0.025)  # 2.5 cm std
        ghost_points[:, :3] += offsets

        # Slight intensity change
        ghost_points[:, 3] += np.random.randn(n_ghost) * 10
        ghost_points[:, 3] = np.clip(ghost_points[:, 3], 0, 255)

        return ghost_points

    def apply_perturbation(
        self, point_cloud: np.ndarray, params: Dict[str, any], seed: Optional[int] = None,
        ctx: Optional[Dict] = None, seme_genoma: Optional[int] = None, metro: Optional[int] = None,
    ) -> np.ndarray:
        """
        Apply the full perturbation pipeline to a point cloud.

        Operators run in sequence: curvature weights, per-point noise, cluster
        perturbation, dropout, ghost points, geometric distortion, edge attack,
        temporal drift, scanline perturbation.

        Args:
            point_cloud: Input point cloud (N, 4) with [x, y, z, intensity]
            params: Perturbation parameters from encode_perturbation()
            seed: Random seed for reproducibility
            ctx: Coherent mode: cloud stamp (s) and base pose in odom
                (x, y, yaw) estimated by the attacker, if available
            seme_genoma, metro: Coherent mode, rule "caso": seed of the zone
                choice (from the genome) and meter index; None = R1

        Returns:
            Perturbed point cloud (M, 4) where M may differ from N
        """
        # In coherent mode the seed is ignored and numpy's global state is
        # left untouched.
        if self.operator_mode == "coherent":
            return self._apply_coherent(point_cloud, params, ctx, seme_genoma, metro)

        if seed is not None:
            np.random.seed(seed)

        perturbed = point_cloud.copy()
        n_points = len(perturbed)

        perturbation_weights = self._compute_perturbation_weights(perturbed, n_points, params)
        perturbed = self._apply_noise(perturbed, n_points, perturbation_weights, params)

        if params["cluster_strength"] > 0.1:
            perturbed = self._apply_cluster_perturbation(
                perturbed, params["cluster_direction"], params["cluster_strength"]
            )

        perturbed = self._apply_dropout(perturbed, n_points, perturbation_weights, params)
        n_kept = len(perturbed)
        perturbed = self._add_ghost_points(perturbed, params)
        # Only dropout and ghosts change the point count (the following
        # operators work in place): counts for the "gross" term.
        counts = {"n_in": n_points, "n_removed": n_points - n_kept,
                  "n_added": len(perturbed) - n_kept}

        # Global geometric distortion (ICP attack)
        perturbed = self._apply_geometric_distortion(perturbed, params)

        # Edge attack: larger shifts on edge/corner points (SLACK)
        if params.get("edge_attack_strength", 0) > 0.1:
            perturbed = self._apply_edge_attack(perturbed, params)

        # Temporal drift: bias accumulating across frames (ICP attack)
        if params.get("temporal_drift_strength", 0) > 0.1 or self._drift_carried():
            perturbed = self._apply_temporal_drift(perturbed, params)

        # Scanline perturbation: shift along the laser beam (ASP)
        if params.get("scanline_strength", 0) > 0.1:
            perturbed = self._apply_scanline_perturbation(perturbed, params)

        self._last_counts = (weakref.ref(perturbed), counts)
        return perturbed

    def point_counts(self, original: np.ndarray, perturbed: np.ndarray) -> Optional[Dict[str, int]]:
        """Points removed and added by the last apply_perturbation, only if
        `perturbed` is exactly the array it returned and `original` has the
        point count of its input; otherwise None (cloud produced by another
        path, e.g. the Gaussian arm or test probes)."""
        if self._last_counts is None:
            return None
        ref, counts = self._last_counts
        if ref() is not perturbed or counts["n_in"] != len(original):
            return None
        if counts["n_in"] - counts["n_removed"] + counts["n_added"] != len(perturbed):
            return None
        return dict(counts)

    def point_ratios(self, original: np.ndarray, perturbed: np.ndarray):
        """(r_net, r_gross, counts). r_net = |M - N| / N; r_gross
        = (removed + added) / N, None if the counts are not available."""
        n_orig = len(original)
        r_net = abs(len(perturbed) - n_orig) / max(n_orig, 1)
        counts = self.point_counts(original, perturbed)
        r_gross = (None if counts is None
                   else (counts["n_removed"] + counts["n_added"]) / max(n_orig, 1))
        return r_net, r_gross, counts

    def _structural_ratio(self, original, perturbed, r_net):
        """r used in the structural term. With "gross" and no counts it falls back
        to |M - N| / N, which equals the gross value when the cloud only lost or
        only gained points."""
        if self.structural != "gross":
            return r_net
        _, r_gross, _ = self.point_ratios(original, perturbed)
        return r_net if r_gross is None else r_gross

    def _apply_edge_attack(self, point_cloud: np.ndarray, params: Dict[str, any]) -> np.ndarray:
        """
        Shift edge and corner points perpendicular to their principal direction.

        Inspired by SLACK ("location of injection matters more than quantity"):
        edge_attack_strength scales the shift, up to max_edge_shift (8 cm), on the
        top 20% edge scores (at most 500 points). The shift direction is the
        smallest-eigenvalue axis of the local PCA, which disturbs ICP correspondences.
        """
        perturbed = point_cloud.copy()
        n_points = len(perturbed)
        if n_points < 100:
            return perturbed

        strength = params.get("edge_attack_strength", 0)
        if strength < 0.1:
            return perturbed

        edge_scores = self.detect_edges_and_corners(perturbed)

        # Top 20% edge/corner points
        threshold = np.percentile(edge_scores, 80)
        edge_mask = edge_scores >= threshold

        if edge_mask.sum() < 10:
            return perturbed

        edge_indices = np.where(edge_mask)[0]
        tree = cKDTree(perturbed[:, :3])

        sel = edge_indices[: min(self._lim_g13_punti, len(edge_indices))]  # Limit for speed

        # Neighbour indices are queried in one batch on the unmodified cloud:
        # each selected index is visited once, so the result equals per-point queries.
        if len(sel) == 0:
            return perturbed
        _, all_nb_idx = tree.query(perturbed[sel, :3], k=10, workers=-1)

        edge_max = edge_scores.max() + 1e-6
        if getattr(self, "fast_ops", False):
            return self._edge_shift_blocks(perturbed, sel, all_nb_idx, edge_scores,
                                           edge_max, strength, params["noise_direction"])

        # The PCA loop is kept sequential on purpose: neighbours may already have
        # been shifted in a previous iteration.
        for j, idx in enumerate(sel):
            neighbors = perturbed[all_nb_idx[j], :3]

            # PCA of the local neighbourhood
            centered = neighbors - neighbors.mean(axis=0)
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # Shift along the smallest-eigenvalue axis (perpendicular to the edge)
            perp_dir = eigenvectors[:, 0]

            # Shift amount scaled by strength and normalised edge score
            shift_amount = strength * self._lim_edge_shift * (edge_scores[idx] / edge_max)
            # Consistent sign: the perpendicular shift is oriented so that its
            # projection on the attack direction always has the same sign, so
            # the edges build a net distortion along noise_direction instead of
            # cancelling out (with a random sign the mean shift is ~0).
            sgn = np.sign(float(perp_dir @ params["noise_direction"]))
            if sgn == 0.0:
                sgn = 1.0
            shift = perp_dir * shift_amount * sgn

            perturbed[idx, :3] += shift

        return perturbed

    @staticmethod
    def _cov_blocks(nb):
        """Covariances of a block of neighbourhoods (B, k, 3), with the same steps
        as np.cov(centered.T) on a single neighbourhood: double centering, X X^T
        per matrix (syrk, like dot) and scaling by 1 / (k - 1)."""
        centered = nb - nb.mean(axis=1, keepdims=True)
        # np.cov casts to float64 after the first centering (done in the input
        # dtype): same step here, so float32 clouds also match the loop.
        # No copy for float64.
        centered = centered.astype(np.float64, copy=False)
        # (B, 3, k) view with the same memory layout as the copy np.cov makes
        # of centered.T (order K): same summation order.
        x = centered.transpose(0, 2, 1)
        x -= x.mean(axis=2)[:, :, None]
        c = np.matmul(x, x.transpose(0, 2, 1))
        c *= np.true_divide(1, x.shape[2] - 1)
        return c

    def _edge_shift_blocks(self, perturbed, sel, all_nb_idx, edge_scores, edge_max,
                           strength, direction):
        """Same result as the sequential loop of _apply_edge_attack, in blocks.

        Point j reads its neighbours as they are at its turn: shifted if they are
        edges with a lower index in sel, unshifted otherwise. If sel[i] is among
        the neighbours of sel[j] (or vice versa), the lower one is computed first.
        Each level of the precedence chain is a block with no internal links:
        PCA, eigenvectors and shifts are computed together and applied after.
        """
        s = len(sel)
        pos = np.full(len(perturbed), -1, dtype=np.int64)
        pos[sel] = np.arange(s)
        nb_pos = pos[all_nb_idx]                                   # (S, k), -1 if not an edge
        jj, kk = np.nonzero(nb_pos >= 0)
        ii = nb_pos[jj, kk]
        keep = ii != jj
        a = np.minimum(ii[keep], jj[keep])
        b = np.maximum(ii[keep], jj[keep])
        order = np.lexsort((a, b))
        a, b = a[order], b[order]
        starts = np.searchsorted(b, np.arange(s + 1))
        # Levels in pure Python: a few hundred steps without numpy calls.
        a_l, st_l = a.tolist(), starts.tolist()
        lev = [0] * s
        for j in range(s):
            if st_l[j + 1] > st_l[j]:
                lev[j] = max(lev[i] for i in a_l[st_l[j]:st_l[j + 1]]) + 1
        level = np.asarray(lev, dtype=np.int64)

        coef = strength * self._lim_edge_shift
        dsum = float(np.abs(np.asarray(direction, dtype=np.float64)).sum())
        for lv in range(int(level.max()) + 1 if s else 0):
            js = np.nonzero(level == lv)[0]
            idx = sel[js]
            cov = self._cov_blocks(perturbed[all_nb_idx[js], :3])
            _, eigenvectors = np.linalg.eigh(cov)
            perp = eigenvectors[:, :, 0]                           # (B, 3)
            # same dtype and operation order as the per-point loop (float32 clouds too)
            shift_amount = coef * np.array([edge_scores[q] / edge_max for q in idx])
            proj = perp @ direction
            sgn = np.sign(proj)
            # The sign of a dot product very close to zero may depend on the
            # summation order: those are recomputed as in the loop.
            dubbi = np.nonzero(np.abs(proj) <= 1e-9 * dsum)[0]
            for q in dubbi:
                sgn[q] = np.sign(float(perp[q] @ direction))
            sgn[sgn == 0.0] = 1.0
            shift = perp * shift_amount[:, None] * sgn[:, None]
            perturbed[idx, :3] += shift
        return perturbed

    def _apply_temporal_drift(self, point_cloud: np.ndarray, params: Dict[str, any]) -> np.ndarray:
        """
        Translate the whole cloud by a bias that accumulates across frames.

        Global transformation, unlike the per-point operators. Inspired by the ICP
        adversarial attack: temporal_drift_strength scales the per-frame step along
        noise_direction, up to max_temporal_drift (5 cm); the accumulated drift
        decays by 0.98 per frame so it stays bounded. Targets loop closure.
        """
        perturbed = point_cloud.copy()

        strength = params.get("temporal_drift_strength", 0)
        if strength < 0.1 and not self._drift_carried():
            return perturbed

        self._frame_counter += 1

        # Per-frame drift along the noise direction (zero step when only the
        # kept drift of keep mode is applied)
        drift_direction = params["noise_direction"]
        frame_drift = (drift_direction * strength * self._lim_drift_step
                       if strength >= 0.1 else np.zeros(3))

        # Accumulate with decay so the drift stays bounded
        decay = 0.98
        self._accumulated_drift = self._accumulated_drift * decay + frame_drift
        # Saturate the norm; the direction stays the genome's.
        norm = float(np.linalg.norm(self._accumulated_drift))
        if norm > self.max_drift_m > 0:
            self._accumulated_drift *= self.max_drift_m / norm

        # Apply the accumulated drift to all points
        perturbed[:, :3] += self._accumulated_drift

        return perturbed

    def _apply_scanline_perturbation(
        self, point_cloud: np.ndarray, params: Dict[str, any]
    ) -> np.ndarray:
        """
        Shift points along their range direction (toward/away from the sensor).

        Inspired by ASP: a shift along the laser beam resembles particles between
        sensor and object, which is physically plausible. scanline_strength scales
        a mix of Gaussian noise (3 cm std) and a sinusoidal pattern (2 cm amplitude).
        """
        perturbed = point_cloud.copy()
        n_points = len(perturbed)

        strength = params.get("scanline_strength", 0)
        if strength < 0.1 or n_points < 10:
            return perturbed

        # Range direction of each point (sensor at the origin)
        points_xyz = perturbed[:, :3]
        ranges = np.linalg.norm(points_xyz, axis=1, keepdims=True)
        range_directions = points_xyz / (ranges + 1e-6)

        # Random plus systematic component along the beam. The systematic part is
        # a function of the beam elevation (atan2(z, horizontal range)), not of
        # the array index: points on the same scanline get the same radial
        # shift, like a band of particles at that height. The array index has
        # no geometric meaning (ASP perturbs physical scanlines, not the point
        # order in the message).
        elevation = np.arctan2(points_xyz[:, 2],
                               np.hypot(points_xyz[:, 0], points_xyz[:, 1]) + 1e-9)
        random_component = np.random.randn(n_points) * self._lim_scan_rand  # 3 cm std at x1
        systematic_component = np.sin(elevation * SCANLINE_ELEV_WAVES) * self._lim_scan_sist

        scanline_shift = (random_component + systematic_component) * strength
        scanline_shift = scanline_shift[:, np.newaxis] * range_directions

        perturbed[:, :3] += scanline_shift

        return perturbed

    def reset_structures(self):
        """Coherent mode: forget anchors and pose (new meter). The node calls it
        from other threads: the actual reset happens at the start of the next
        cloud, never in the middle of one."""
        if self._structs is not None:
            self._struct_reset_pending = True

    # ------------------------------------------------------------------
    # Coherent mode: whole structures chosen on the first cloud of the meter
    # ------------------------------------------------------------------

    def _insieme_attivo(self):
        """True if the insieme option applies to the current meter (rule caso, seed and meter)."""
        return self.struct_insieme and self._structs is not None and self._structs.per_op

    def _n_zone(self):
        """Zones used per operator: all with the insieme option, N_STRUCT otherwise."""
        return None if self._insieme_attivo() else self._st.N_STRUCT

    def _obiettivi_serieA(self, xyz, p, ops, seme, metro):
        """Points that random mode touches in cloud xyz with the same parameters,
        per operator: {"gN": (count, zone measure)}. Expected values, not draws:
        g5 = sum of the removal probabilities of _apply_dropout (with the minimum
        kept fraction); g6/g16 = int((n - g5) * ghost_ratio); g10 = median over
        16 draws (local generator from seed, meter, 10, 1) of the points within
        the _apply_cluster_perturbation radius of the n_clusters centres;
        g13 = min(500, 20 % of the points), as in _apply_edge_attack.
        numpy's global state is not changed."""
        n = len(xyz)
        ob = {}
        n_rm = 0
        if "g5" in ops and n > 1:
            k = min(20, n - 1)
            d, _ = cKDTree(xyz).query(xyz, k=k + 1, workers=-1)
            dens = 1.0 / (d[:, 1:].mean(axis=1) + 1e-6)
            dw = (dens - dens.min()) / (dens.max() - dens.min() + 1e-6)
            keep = (1 - p["dropout_rate"]) * (1.0 - 0.1 * dw)
            min_keep = max(self._min_keep, 1 - p["dropout_rate"] * 2)
            n_rm = int(round(float((1.0 - keep).sum())))
            if n - n_rm < n * min_keep:
                n_rm = int(round(n * (1 - min_keep)))
            ob["g5"] = (n_rm, "tolti")
        for g in ("g6", "g16"):
            if g in ops:
                ob[g] = (int((n - n_rm) * p["ghost_ratio"]), "voxel")
        if "g10" in ops:
            n_c = min(self.n_clusters, n // 100)
            if n_c > 0:
                r = np.percentile(np.linalg.norm(xyz - xyz.mean(axis=0), axis=1), 10)
                tree = cKDTree(xyz)
                rng = np.random.default_rng(np.random.SeedSequence([int(seme), int(metro), 10, 1]))
                cnt = []
                for _ in range(16):
                    c = rng.choice(n, n_c, replace=False)
                    hit = np.zeros(n, dtype=bool)
                    for lst in tree.query_ball_point(xyz[c], r, workers=-1):
                        hit[lst] = True
                    cnt.append(int(hit.sum()))
                ob["g10"] = (int(np.median(cnt)), "punti")
            else:
                ob["g10"] = (0, "punti")
        if "g13" in ops:
            ob["g13"] = (min(self._lim_g13_punti, int(np.ceil(0.2 * n))) if n >= 100 else 0, "punti")
        return ob

    def _chiave(self, op, t):
        """Key of anchors and hits: the operator with rule "caso", the type otherwise."""
        return op if self._structs.per_op else t

    def _apply_coherent(self, point_cloud, params, ctx=None, seme_genoma=None, metro=None):
        """O4 (structures.OTT): the target KD-tree built in a thread while the
        operators run is put back in place before returning. If the body raises,
        its error wins over the thread's."""
        try:
            out = self._apply_coherent_corpo(point_cloud, params, ctx, seme_genoma, metro)
        except BaseException:
            if self._structs is not None:
                try:
                    self._structs.completa_bersaglio()
                except Exception:
                    pass
            raise
        if self._structs is not None:
            self._structs.completa_bersaglio()
        return out

    def _apply_coherent_corpo(self, point_cloud, params, ctx=None, seme_genoma=None, metro=None):
        """Coherent-mode pipeline, same order as random mode. Structures and
        descriptors come from the clean input cloud; directions and amplitudes
        from the genome, fixed in the anchor frame (sensor frame on the first
        cloud of the meter)."""
        st = self._st
        perturbed = point_cloud.copy()
        n = len(perturbed)
        if n == 0:
            return perturbed
        p = params
        on10 = p["cluster_strength"] > 0.1
        on5 = self._x1(p, "dropout_rate") > 0.01
        on13 = p.get("edge_attack_strength", 0) > 0.1
        ghosts = self._x1(p, "ghost_ratio") > 0.01
        strategic = p.get("strategic_ghost", 0) > 0.5
        need = set()
        if on13 or on5 or (ghosts and strategic):
            need.add(st.EDGE_V)
        if ghosts and not strategic:
            need.add(st.WALL)
        if on10:
            need.add(self.struct_cluster_type)
        cd = np.asarray(p["cluster_direction"], dtype=np.float64)
        az_t = float(np.arctan2(cd[1], cd[0])) if np.hypot(cd[0], cd[1]) > 1e-9 else None
        tr = self._structs
        if getattr(self, "_struct_reset_pending", False):
            self._struct_reset_pending = False
            tr.reset()
        ops = {}
        if on5:
            ops["g5"] = (5, st.EDGE_V)
        if ghosts:
            ops["g16" if strategic else "g6"] = (16, st.EDGE_V) if strategic else (6, st.WALL)
        if on10:
            ops["g10"] = (10, self.struct_cluster_type)
        if on13:
            ops["g13"] = (13, st.EDGE_V)
        if self.zone_attaccante and tr.k == -1 and not tr.fisse:
            self._installa_zone(seme_genoma, metro, p)
            if self.rampa_spec is not None or self._rampa is not None:
                self._installa_rampa(metro)
            if self._gradino is not None:
                self._installa_gradino(metro)
        if self._rampa is not None and tr.k + 1 >= self._rampa["R"] - 1:
            # cloud k = R - 1 (alpha = 1): from here on only N, plain coherent path
            self._fine_rampa()
        ob = None
        if (self.struct_insieme and tr.k == -1 and tr.rule == "caso" and not tr.fisse
                and seme_genoma is not None and metro is not None):
            ob = self._obiettivi_serieA(np.asarray(perturbed[:, :3], dtype=np.float64), p, ops,
                                        seme_genoma, metro)
            self._obiettivi = ob
        ob_scelta = ob
        if ob is not None and self._riserva_scelta:
            # attacker zones: choice with a reserve, application with the target
            ob_scelta = {k: ((int(v[0] * self._riserva_scelta[k]), v[1]) if k in self._riserva_scelta else v)
                         for k, v in ob.items()}
        fr = tr.update(perturbed[:, :3], sorted(need), az_target=az_t,
                       noise_dir=p["noise_direction"], ctx=ctx,
                       seme=seme_genoma, metro=metro, operatori=ops, obiettivi=ob_scelta)
        if self.rampa_info is not None:
            self._rampa_passo(tr.k)
        if self._rampa is not None:
            return self._apply_coherent_rampa(point_cloud, perturbed, fr, p, seme_genoma, metro)
        if self._gradino is not None and self._gradino_passo(fr, None if ctx is None else ctx.get("stamp")):
            return self._apply_coherent_gradino(point_cloud, perturbed, fr, metro)
        th = fr["pose"][2]
        vop = fr["vox_of_point"]
        hits = {}

        # genes 3, 4 and 11: weights from the curvature of the point's voxel
        # (random-mode rule), averaged over neighbouring voxels, shift along
        # the gene 0-2 direction (1.3 times the intensity)
        w_pt = np.ones(n)
        if self.target_high_curvature and p["curvature_strength"] > 0.1:
            cv = fr["curv"]
            if cv.max() > cv.min():
                cw = ((cv - cv.min()) / (cv.max() - cv.min()))[vop]
                thr = np.percentile(cw, 100 - self.curvature_percentile * p["curvature_strength"])
                w_pt = np.where(cw >= thr, 1.0, 0.3)
        if self._x1(p, "noise_intensity") > 0.001:
            corr = p["spatial_correlation"]
            if corr >= 0.3 and n >= 100:
                cnt = fr["counts"].astype(np.float64)
                w_vox = np.bincount(vop, weights=w_pt, minlength=len(cnt)) / cnt
                w_pt = w_pt * (1 - corr) + w_vox[fr["nb"]].mean(axis=1)[vop] * corr
            shift = np.asarray(p["noise_direction"]) * p["noise_intensity"] * 1.3
            disp = w_pt[:, None] * shift[None, :]
            nr = np.linalg.norm(disp, axis=1, keepdims=True)
            disp = np.where(nr > self._lim_point_shift,
                            disp / np.maximum(nr, 1e-12) * self._lim_point_shift, disp)
            perturbed[:, :3] += disp

        # gene 10: uniform shift of the chosen structures
        zi = self._zone_int
        obz = {}
        if on10 and zi is not None and zi.get("palle_g10") is not None:
            # attacker zones: voxels of the balls chosen on the snapshot (centres
            # and radius as in random mode), tracked in the anchor frame
            t10 = obz["g10"] = int(round((zi.get("quota_g10") or 0.0) * n))
            m = self._punti_palle(fr, zi, t10, n)
            sh = st.se2_rot(-th, cd * p["cluster_strength"] * self._lim_cluster_std * 1.5)
            perturbed[m, :3] += sh
            hits["g10"] = np.flatnonzero(m)
        elif on10:
            hl = [v for v in fr["hits"].get(self._chiave("g10", self.struct_cluster_type), []) if len(v)]
            if self._insieme_attivo() and "g10" in (self._obiettivi or {}):
                # whole zones in order, skipping those that exceed the target
                lim, tot, sc = self._obiettivi["g10"][0], 0, []
                for v in hl:
                    c = int(fr["counts"][v].sum())
                    if tot + c <= lim:
                        sc.append(v)
                        tot += c
                hl = sc
            m = np.isin(vop, np.concatenate(hl)) if hl else np.zeros(n, dtype=bool)
            sh = st.se2_rot(-th, cd * p["cluster_strength"] * self._lim_cluster_std * 1.5)
            perturbed[m, :3] += sh
            hits["g10"] = np.flatnonzero(m)

        # gene 5: thinning (a) or whole voxels (b) in the chosen edges
        alive = np.arange(n)
        if on5:
            t5 = None
            if zi is not None and zi.get("quota_g5") is not None:
                # random-mode expected removals on the whole cloud, taken only in the zones
                t5 = obz["g5"] = int(round(zi["quota_g5"] * n))
            drop = self._coherent_struct_dropout(fr, n, p["dropout_rate"], t5)
            alive = np.flatnonzero(~drop)
            perturbed = perturbed[alive]
            hits["g5"] = np.flatnonzero(drop)
        pos = np.full(n, -1, dtype=np.int64)
        pos[alive] = np.arange(len(alive))
        n_real = len(perturbed)

        # genes 6 and 16: one ghost per voxel of the chosen structures
        if ghosts and n_real > 0:
            t = st.EDGE_V if strategic else st.WALL
            t = self._chiave("g16" if strategic else "g6", t)
            tg = None
            if zi is not None:
                # random-mode base in expectation: points left after the expected removals
                t5a = int(round(zi["quota_g5"] * n)) if (on5 and zi.get("quota_g5") is not None) else 0
                tg = obz["g16" if strategic else "g6"] = int((n - t5a) * p["ghost_ratio"])
            perturbed, bases = self._coherent_struct_ghosts(perturbed, fr, t, pos, p, strategic, th, tg)
            hits["g16" if strategic else "g6"] = bases

        # gene 12: centre and normalization range of the first cloud of the meter
        c0, r0 = tr.center_and_range(fr["pose"])
        perturbed = self._apply_geometric_distortion(perturbed, p, center=c0, max_range=r0)

        # gene 13: rigid shift of the chosen edges, orthogonal to their axis
        if on13 and n_real >= 100:
            n13 = int(np.ceil(5 * p["edge_attack_strength"]))
            moved = []
            k13 = self._chiave("g13", st.EDGE_V)
            ins13 = self._insieme_attivo() and "g13" in (self._obiettivi or {})
            if ins13:
                n13 = None
                resto = self._obiettivi["g13"][0]
            for A, v in zip(tr.anchors.get(k13, [])[:n13], fr["hits"].get(k13, [])[:n13]):
                if not len(v):
                    continue
                ci = np.flatnonzero(np.isin(vop, v))
                idx = pos[ci]
                ci, idx = ci[idx >= 0], idx[idx >= 0]
                if ins13:
                    # zone points within the target, by edge score
                    if resto <= 0:
                        break
                    if len(ci) > resto:
                        sel = self._g13_per_punteggio(point_cloud[ci, :3], seme_genoma, metro, tr.k,
                                                      quanti=resto)
                        ci, idx = ci[sel], idx[sel]
                    resto -= len(ci)
                elif self.struct_g13_punteggio and len(ci) > self._lim_g13_punti:
                    sel = self._g13_per_punteggio(point_cloud[ci, :3], seme_genoma, metro, tr.k,
                                                  quanti=self._lim_g13_punti)
                    ci, idx = ci[sel], idx[sel]
                d = st.se2_rot(-th, A["edge_dir"]) * (p["edge_attack_strength"] * self._lim_edge_shift)
                perturbed[idx, :3] += d
                moved.append(ci)
            hits["g13"] = np.concatenate(moved) if moved else np.zeros(0, dtype=np.int64)
            if zi is not None and ins13:
                obz["g13"] = int(self._obiettivi["g13"][0])

        if p.get("temporal_drift_strength", 0) > 0.1 or self._drift_carried():
            perturbed = self._apply_temporal_drift(perturbed, p)
        if p.get("scanline_strength", 0) > 0.1 and len(perturbed) >= 10:
            # gene 15: only the systematic component over the beam elevation
            xyz = perturbed[:, :3]
            rr = np.linalg.norm(xyz, axis=1, keepdims=True)
            elev = np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1]) + 1e-9)
            sh = np.sin(elev * SCANLINE_ELEV_WAVES) * self._lim_scan_sist * p["scanline_strength"]
            perturbed[:, :3] += sh[:, None] * (xyz / (rr + 1e-6))
        info = fr["info"]
        info["n_hit"] = {k: int(len(v)) for k, v in hits.items()}
        info["per_op"] = bool(tr.per_op)
        info["metro"] = None if metro is None else int(metro)
        if self.zone_attaccante:
            info["zone_stato"] = self.zone_stato
            info["zone_hash"] = self.zone_hash
            if zi is not None:
                # random-mode expected points per operator and gap (expected - touched) per cloud
                info["zone_attesi"] = obz
                info["zone_scarto"] = {k: v - int(len(hits.get(k, ()))) for k, v in obz.items()}
        self.last_struct_info = info
        self.last_hits = hits
        # Counts for the "gross" term: only g5 removes and only g6/g16 add.
        self._last_counts = (weakref.ref(perturbed),
                             {"n_in": n, "n_removed": n - n_real, "n_added": len(perturbed) - n_real})
        return perturbed

    def _installa_zone(self, seme, metro, params):
        """Attacker zones for the current genome, on the first cloud of the
        meter: installed in the tracker as they are, or no zone if the provider
        has no valid ones."""
        tr = self._structs
        try:
            z, stato = ((None, "fornitore_assente") if self.zone_fornitore is None
                        else self.zone_fornitore(seme, metro, params))
        except Exception as e:        # never a fallback choice
            z, stato = None, f"errore_fornitore: {e}"
        if z is not None and stato == "ok":
            tr.installa(z)
            self._obiettivi = dict(z.obiettivi)
            self.zone_hash = z.hash
            self._zone_int = {"quota_g5": z.quota_g5, "palle_g10": z.palle_g10, "quota_g10": z.quota_g10,
                              "centri_g10": z.palle_centri, "raggio_g10": z.palle_raggio}
        else:
            tr.installa_vuoto()
            self._obiettivi = None
            self.zone_hash = None
            self._zone_int = None
        self.zone_stato = stato
        self._zone_n = z if (z is not None and stato == "ok") else None

    # ------------------------------------------------------------------
    # Genome ramp: cross-fade between the previous and the new genome
    # ------------------------------------------------------------------

    def imposta_rampa(self, R, prec=None, stato=None, genoma_sha=None, seq=None):
        """Ramp for the meters of the incoming genome; call before
        reset_structures. prec = {"params", "seme", "zone", "sha"} of the
        previous genome P (zones already checked against genome, seed and
        parameters); None = no ramp, with stato = reason, and only N is applied.
        The spec holds for every zone installation (also after a replay) until
        the next call."""
        R = int(R)
        if prec is not None and (R < 2 or not self.zone_attaccante):
            raise ValueError("rampa: servono R >= 2 e le zone dell'attaccante")
        self.rampa_spec = None if prec is None else dict(prec, R=R)
        self.rampa_info = {"R": R, "genoma_sha": genoma_sha,
                           "genoma_prec_sha": None if prec is None else prec["sha"],
                           "stato": "armata" if prec is not None else str(stato or "prec_assente"),
                           "k": None, "alpha": None, "seq": seq}

    def _installa_rampa(self, metro):
        """First cloud of the meter, after the zones of N: anchors of P in the
        tracker with prefix RAMPA_PREFISSO (per-anchor pose and hit voxels do not
        depend on the other anchors, structures.py _associate and update). Only
        with valid zones of N and zones of P from the same snapshot (sha256,
        hash, center0, max_range0) and meter; otherwise no ramp and the reason in
        the state, never a fallback choice."""
        self._rampa = None
        sp = self.rampa_spec
        if sp is None:
            return
        zp = sp["zone"]
        stato, chiavi = self._ancore_prec(zp, metro)
        if chiavi is not None:
            self._rampa = {"R": int(sp["R"]), "params": sp["params"], "seme": int(sp["seme"]),
                           "sha": sp["sha"], "chiavi": chiavi, "obiettivi": dict(zp.obiettivi),
                           "zi": {"quota_g5": zp.quota_g5, "palle_g10": zp.palle_g10,
                                  "quota_g10": zp.quota_g10, "centri_g10": zp.palle_centri,
                                  "raggio_g10": zp.palle_raggio}}
        info = dict(self.rampa_info or {}, stato=stato, k=None, alpha=None, zone_prec_hash=zp.hash)
        self.rampa_info = info
        self._rampa_info_metro = info

    def _fine_rampa(self):
        """Before cloud k = R - 1: drop the anchors of P; from there only N."""
        tr = self._structs
        for key in self._rampa["chiavi"]:
            tr.anchors.pop(key, None)
            tr.margin.pop(key, None)
        self._rampa = None
        if self._rampa_info_metro is not None:
            self._rampa_info_metro["stato"] = "finita"

    def _rampa_passo(self, k):
        """Record cloud index and weight in the meter ramp state (status)."""
        ri = self._rampa_info_metro
        if ri is None or ri is not self.rampa_info:
            return
        ri["k"] = int(k)
        if ri.get("stato") in ("attiva", "finita"):
            ri["alpha"] = rampa_alfa(k, ri["R"])

    def _apply_coherent_rampa(self, point_cloud, perturbed, fr, p, seme_genoma, metro):
        """Ramp cloud k < R - 1: P with weight m_P / R = (R - k - 1) / R and N
        with weight m_N / R = (k + 1) / R (_coherent_pesato)."""
        rp, R, k = self._rampa, self._rampa["R"], self._structs.k
        termini = [{"pre": RAMPA_PREFISSO, "p": rp["params"], "seme": rp["seme"],
                    "ob": rp["obiettivi"], "zi": rp["zi"], "m": R - k - 1},
                   {"pre": "", "p": p, "seme": seme_genoma, "ob": self._obiettivi,
                    "zi": self._zone_int, "m": k + 1}]
        return self._coherent_pesato(point_cloud, perturbed, fr, termini, R, metro)

    def _ancore_prec(self, zp, metro):
        """Install the anchors of the previous genome's zones zp in the tracker,
        prefixed with RAMPA_PREFISSO, after the zones of N (ramp and step). Only
        with valid zones of N and zones of P from the same snapshot (sha256,
        hash, center0, max_range0) and meter. Returns (state, installed keys),
        keys None if nothing was installed."""
        tr = self._structs
        zn = self._zone_n
        if self.zone_stato != "ok" or zn is None or self._zone_int is None:
            return f"zone_nuovo_{self.zone_stato}", None
        if zp.metro != zn.metro or (metro is not None and zp.metro != int(metro)):
            return "metro_diverso", None
        if (zp.meta.get("fotografia_sha256") != zn.meta.get("fotografia_sha256")
                or zp.meta.get("fotografia_hash") != zn.meta.get("fotografia_hash")
                or zp.center0 is None or tr.center0 is None
                or not np.array_equal(zp.center0, tr.center0) or zp.max_range0 != tr.max_range0):
            return "fotografia_diversa", None
        chiavi = []
        for op, lst in sorted(zp.anchors.items()):
            key = RAMPA_PREFISSO + op
            tr.anchors[key] = [self._st.copia_ancora(A) for A in lst]
            tr.margin[key] = float("nan")
            chiavi.append(key)
        return "attiva", chiavi

    # ------------------------------------------------------------------
    # Genome in motion: step ramp triggered by the robot's motion
    # ------------------------------------------------------------------

    def imposta_gradino(self, S_cm, kmin, kmax, prec=None, stato=None, genoma_sha=None, seq=None):
        """Step for the meters of the incoming genome (call before
        reset_structures, like imposta_rampa). prec = {"params", "seme", "zone",
        "sha"} of the previous genome P (zones already checked); None = no P,
        with stato = reason ("metro0" on meter 0), and only N is applied. The
        spec holds for every zone installation until the next call."""
        S_cm, kmin, kmax = float(S_cm), int(kmin), int(kmax)
        if not (S_cm > 0 and 0 <= kmin <= kmax):
            raise ValueError(f"gradino: servono S > 0 e 0 <= kmin <= kmax (S {S_cm}, kmin {kmin}, kmax {kmax})")
        if prec is not None and not self.zone_attaccante:
            raise ValueError("gradino: servono le zone dell'attaccante")
        if self.rampa_spec is not None or self._rampa is not None:
            raise ValueError("gradino e rampa A3 insieme")
        self._gradino = {"S_cm": S_cm, "kmin": kmin, "kmax": kmax,
                         "spec": None if prec is None else dict(prec),
                         "genoma_sha": genoma_sha, "genoma_prec_sha": None if prec is None else prec["sha"],
                         "seq": seq, "stato": "armato" if prec is not None else str(stato or "prec_assente")}
        self._gradino_azzera()

    def _gradino_azzera(self):
        """Reset the per-meter step state: no P installed, no reference."""
        self._gradino.update(P=None, fase=None, k=None, rif=None, rif_k=None, riprendi=False,
                             spost_cm=None, spost_prec_cm=None, buco_s=None, prec_stamp=None,
                             k_ingresso=None, spost_ingresso_cm=None, spost_prima_cm=None, buco_stamp_s=None)

    def _installa_gradino(self, metro):
        """First cloud of the meter, after the zones of N: anchors of P
        (_ancore_prec); without P or with invalid zones only N for the whole
        meter, reason in the state, never a fallback choice."""
        g = self._gradino
        self._gradino_azzera()
        sp = g["spec"]
        if sp is None:
            if g["stato"] == "metro0":
                g["fase"] = "metro0"
            return
        zp = sp["zone"]
        g["zone_prec_hash"] = zp.hash
        stato, chiavi = self._ancore_prec(zp, metro)
        if chiavi is None:
            g["stato"] = stato
            return
        g["stato"], g["fase"] = "attivo", "P"
        g["P"] = {"params": sp["params"], "seme": int(sp["seme"]), "sha": sp["sha"], "chiavi": chiavi,
                  "obiettivi": dict(zp.obiettivi),
                  "zi": {"quota_g5": zp.quota_g5, "palle_g10": zp.palle_g10, "quota_g10": zp.quota_g10,
                         "centri_g10": zp.palle_centri, "raggio_g10": zp.palle_raggio}}

    def riprendi_riferimento_moto(self):
        """Take p_rif again on the first cloud processed after the call (after
        loading a saved state, or when the victim stream reopens)."""
        if self._gradino is not None:
            self._gradino["riprendi"] = True

    def _gradino_passo(self, fr, stamp):
        """After tr.update: cloud phase and trigger. True = cloud with P only;
        False = single-genome path with N only (on the entry cloud the anchors
        of P are already removed)."""
        g = self._gradino
        k = int(self._structs.k)
        g["k"] = k
        riprendi, g["riprendi"] = g["riprendi"], False
        if g["fase"] != "P":
            if g["fase"] == "ingresso":
                g["fase"] = "N"
            g["spost_cm"] = g["spost_prec_cm"] = g["buco_s"] = None
            return False
        x, y = float(fr["pose"][0]), float(fr["pose"][1])
        if riprendi or g["rif"] is None:
            g["rif"], g["rif_k"] = (x, y), k
            g["spost_prec_cm"] = g["buco_s"] = g["prec_stamp"] = None
        else:
            g["spost_prec_cm"] = g["spost_cm"]
            g["buco_s"] = (None if stamp is None or g["prec_stamp"] is None
                           else float(stamp) - float(g["prec_stamp"]))
        g["spost_cm"] = math.hypot(x - g["rif"][0], y - g["rif"][1]) * 100.0
        g["prec_stamp"] = stamp
        mosso = k >= g["kmin"] and g["spost_cm"] >= g["S_cm"]
        if not (mosso or k >= g["kmax"]):
            return True
        g.update(fase="ingresso", stato="attivo" if mosso else "k_max", k_ingresso=k,
                 spost_ingresso_cm=g["spost_cm"], spost_prima_cm=g["spost_prec_cm"], buco_stamp_s=g["buco_s"])
        self._fine_gradino(fr)
        return False

    def _fine_gradino(self, fr):
        """Entry cloud, after tr.update with both anchor sets: drop the anchors
        of P from the tracker, the hits and the cloud info; from here only N,
        on the single-genome path."""
        tr = self._structs
        inf = fr.get("info") or {}
        for key in self._gradino["P"]["chiavi"]:
            tr.anchors.pop(key, None)
            tr.margin.pop(key, None)
            fr["hits"].pop(key, None)
            for c in ("anchors", "assoc", "hit_vox", "margin", "tipo_ancora"):
                if isinstance(inf.get(c), dict):
                    inf[c].pop(key, None)
        self._gradino["P"] = None

    def _apply_coherent_gradino(self, point_cloud, perturbed, fr, metro):
        """Cloud in phase P: only P at full weight (m = R = 1); the anchors of N
        are tracked but not applied (_coherent_pesato skips zero-weight terms)."""
        P = self._gradino["P"]
        termini = [{"pre": RAMPA_PREFISSO, "p": P["params"], "seme": P["seme"],
                    "ob": P["obiettivi"], "zi": P["zi"], "m": 1}]
        return self._coherent_pesato(point_cloud, perturbed, fr, termini, 1, metro)

    def stato_gradino(self):
        """Step state of the last cloud (node status, run record), None if off."""
        g = self._gradino
        if g is None:
            return None
        return {c: g.get(c) for c in ("stato", "fase", "k", "k_ingresso", "spost_ingresso_cm", "spost_prima_cm",
                                      "buco_stamp_s", "rif_k", "spost_cm", "spost_prec_cm", "buco_s",
                                      "genoma_sha", "genoma_prec_sha", "seq", "zone_prec_hash")}

    def _coherent_pesato(self, point_cloud, perturbed, fr, termini, R, metro):
        """Coherent-mode operators with attacker zones for several genomes on
        the same cloud, after tr.update (same steps and order as _apply_coherent;
        at each step the genomes in the order of termini, P before N).

        termini: {"pre" (key prefix of anchors and hits), "p" (parameters),
        "seme", "ob" (targets), "zi" (zone intensity), "m"}; weight w = m / R,
        terms with m = 0 touch nothing. Each genome keeps the single-genome
        switches and thresholds. Noise, g10 and g13 apply the single-genome
        shift times w on the same points; g5 removes the first
        rampa_conta(m, R, T) points of the genome's removal list and ghosts add
        the first rampa_conta(m, R, T) of the genome (P first, then N), with
        bases among the points left after all removals; g12 uses _rampa_g12,
        the drift a single accumulator with step sum w f (_rampa_deriva), g15
        strength sum w s15. With a single term at m = R the output equals
        _apply_coherent."""
        st = self._st
        tr = self._structs
        n = len(perturbed)
        th = fr["pose"][2]
        vop = fr["vox_of_point"]
        hits, obz = {}, {}
        att = []
        for T in termini:
            if int(T["m"]) <= 0:
                continue
            q = T["p"]
            att.append(dict(T, w=int(T["m"]) / int(R), m=int(T["m"]),
                            on10=q["cluster_strength"] > 0.1, on5=self._x1(q, "dropout_rate") > 0.01,
                            on13=q.get("edge_attack_strength", 0) > 0.1, ghosts=self._x1(q, "ghost_ratio") > 0.01,
                            strategic=q.get("strategic_ghost", 0) > 0.5))

        # genes 3, 4 and 11: single-genome shift per genome, weighted
        for T in att:
            q = T["p"]
            w_pt = np.ones(n)
            if self.target_high_curvature and q["curvature_strength"] > 0.1:
                cv = fr["curv"]
                if cv.max() > cv.min():
                    cw = ((cv - cv.min()) / (cv.max() - cv.min()))[vop]
                    thr = np.percentile(cw, 100 - self.curvature_percentile * q["curvature_strength"])
                    w_pt = np.where(cw >= thr, 1.0, 0.3)
            if self._x1(q, "noise_intensity") > 0.001:
                corr = q["spatial_correlation"]
                if corr >= 0.3 and n >= 100:
                    cnt = fr["counts"].astype(np.float64)
                    w_vox = np.bincount(vop, weights=w_pt, minlength=len(cnt)) / cnt
                    w_pt = w_pt * (1 - corr) + w_vox[fr["nb"]].mean(axis=1)[vop] * corr
                shift = np.asarray(q["noise_direction"]) * q["noise_intensity"] * 1.3
                disp = w_pt[:, None] * shift[None, :]
                nr = np.linalg.norm(disp, axis=1, keepdims=True)
                disp = np.where(nr > self._lim_point_shift,
                                disp / np.maximum(nr, 1e-12) * self._lim_point_shift, disp)
                perturbed[:, :3] += T["w"] * disp

        # gene 10: same points as the single-genome path, weighted shift
        for T in att:
            if not T["on10"]:
                continue
            q, zi, pre = T["p"], T["zi"], T["pre"]
            cd = np.asarray(q["cluster_direction"], dtype=np.float64)
            sh = st.se2_rot(-th, cd * q["cluster_strength"] * self._lim_cluster_std * 1.5)
            if zi is not None and zi.get("palle_g10") is not None:
                t10 = obz[pre + "g10"] = int(round((zi.get("quota_g10") or 0.0) * n))
                m = self._punti_palle(fr, zi, t10, n)
            else:
                hl = [v for v in fr["hits"].get(pre + self._chiave("g10", self.struct_cluster_type), [])
                      if len(v)]
                ob = T["ob"]
                if self._insieme_attivo() and "g10" in (ob or {}):
                    lim, tot, sc = ob["g10"][0], 0, []
                    for v in hl:
                        c = int(fr["counts"][v].sum())
                        if tot + c <= lim:
                            sc.append(v)
                            tot += c
                    hl = sc
                m = np.isin(vop, np.concatenate(hl)) if hl else np.zeros(n, dtype=bool)
            perturbed[m, :3] += T["w"] * sh
            hits[pre + "g10"] = np.flatnonzero(m)

        # gene 5: weighted prefixes of the single-genome lists, union of removals
        alive = np.arange(n)
        drop = None
        for T in att:
            if not T["on5"]:
                continue
            q, zi, pre = T["p"], T["zi"], T["pre"]
            t5 = None
            if zi is not None and zi.get("quota_g5") is not None:
                t5 = int(round(zi["quota_g5"] * n))
                obz[pre + "g5"] = rampa_conta(T["m"], R, t5)
            d5 = self._coherent_struct_dropout(fr, n, q["dropout_rate"], t5,
                                               chiave=pre + self._chiave("g5", st.EDGE_V),
                                               frazione=(T["m"], R))
            hits[pre + "g5"] = np.flatnonzero(d5)
            drop = d5 if drop is None else (drop | d5)
        if drop is not None:
            alive = np.flatnonzero(~drop)
            perturbed = perturbed[alive]
        pos = np.full(n, -1, dtype=np.int64)
        pos[alive] = np.arange(len(alive))
        n_real = len(perturbed)

        # genes 6 and 16: weighted prefixes, bases among the remaining points
        for T in att:
            if not (T["ghosts"] and n_real > 0):
                continue
            q, zi, pre, strategic = T["p"], T["zi"], T["pre"], T["strategic"]
            key = pre + self._chiave("g16" if strategic else "g6", st.EDGE_V if strategic else st.WALL)
            tg = None
            if zi is not None:
                t5a = int(round(zi["quota_g5"] * n)) if (T["on5"] and zi.get("quota_g5") is not None) else 0
                tg = obz[key] = rampa_conta(T["m"], R, int((n - t5a) * q["ghost_ratio"]))
            perturbed, bases = self._coherent_struct_ghosts(perturbed, fr, key, pos, q, strategic, th, tg)
            hits[key] = bases

        # gene 12: centre and range of the snapshot, weighted strengths
        c0, r0 = tr.center_and_range(fr["pose"])
        perturbed = self._rampa_g12(perturbed, att, c0, r0)

        # gene 13: same points as the single-genome path (genome seed), weighted shift
        for T in att:
            q, pre = T["p"], T["pre"]
            if not (T["on13"] and n_real >= 100):
                continue
            n13 = int(np.ceil(5 * q["edge_attack_strength"]))
            moved = []
            k13 = pre + self._chiave("g13", st.EDGE_V)
            ob = T["ob"]
            ins13 = self._insieme_attivo() and "g13" in (ob or {})
            if ins13:
                n13 = None
                resto = ob["g13"][0]
            for A, v in zip(tr.anchors.get(k13, [])[:n13], fr["hits"].get(k13, [])[:n13]):
                if not len(v):
                    continue
                ci = np.flatnonzero(np.isin(vop, v))
                idx = pos[ci]
                ci, idx = ci[idx >= 0], idx[idx >= 0]
                if ins13:
                    if resto <= 0:
                        break
                    if len(ci) > resto:
                        sel = self._g13_per_punteggio(point_cloud[ci, :3], T["seme"], metro, tr.k,
                                                      quanti=resto)
                        ci, idx = ci[sel], idx[sel]
                    resto -= len(ci)
                elif self.struct_g13_punteggio and len(ci) > self._lim_g13_punti:
                    sel = self._g13_per_punteggio(point_cloud[ci, :3], T["seme"], metro, tr.k,
                                                  quanti=self._lim_g13_punti)
                    ci, idx = ci[sel], idx[sel]
                d = st.se2_rot(-th, A["edge_dir"]) * (q["edge_attack_strength"] * self._lim_edge_shift)
                perturbed[idx, :3] += T["w"] * d
                moved.append(ci)
            hits[k13] = np.concatenate(moved) if moved else np.zeros(0, dtype=np.int64)
            if T["zi"] is not None and ins13:
                obz[k13] = int(ob["g13"][0])

        if any(T["p"].get("temporal_drift_strength", 0) > 0.1 for T in att) or self._drift_carried():
            perturbed = self._rampa_deriva(perturbed, att)
        forze = [T["w"] * T["p"]["scanline_strength"] for T in att
                 if T["p"].get("scanline_strength", 0) > 0.1]
        if forze and len(perturbed) >= 10:
            # gene 15: systematic component with the weighted strength
            F = forze[0]
            for f in forze[1:]:
                F = F + f
            xyz = perturbed[:, :3]
            rr = np.linalg.norm(xyz, axis=1, keepdims=True)
            elev = np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1]) + 1e-9)
            sh = np.sin(elev * SCANLINE_ELEV_WAVES) * self._lim_scan_sist * F
            perturbed[:, :3] += sh[:, None] * (xyz / (rr + 1e-6))
        info = fr["info"]
        info["n_hit"] = {kk: int(len(v)) for kk, v in hits.items()}
        info["per_op"] = bool(tr.per_op)
        info["metro"] = None if metro is None else int(metro)
        if self.zone_attaccante:
            info["zone_stato"] = self.zone_stato
            info["zone_hash"] = self.zone_hash
            if self._zone_int is not None:
                # expected points (already weighted for g5 and ghosts) and gap per key
                info["zone_attesi"] = obz
                info["zone_scarto"] = {kk: v - int(len(hits.get(kk, ()))) for kk, v in obz.items()}
        self.last_struct_info = info
        self.last_hits = hits
        self._last_counts = (weakref.ref(perturbed),
                             {"n_in": n, "n_removed": n - n_real, "n_added": len(perturbed) - n_real})
        return perturbed

    def _rampa_g12(self, point_cloud, att, center, max_range):
        """_apply_geometric_distortion with effective strength s = w s12 per
        genome (no term if s12 < 0.01, the single-genome threshold): range and
        constant biases summed, angle (sum s) GEOM_MAX_YAW_RAD, scale matrix
        I + sum (s GEOM_MAX_SCALE) u u^T; same order of the four terms."""
        perturbed = point_cloud.copy()
        if len(perturbed) < 10:
            return perturbed
        act = [(T["w"] * T["p"]["geometric_distortion"], T["p"]) for T in att
               if T["p"].get("geometric_distortion", 0.0) >= 0.01]
        if not act:
            return perturbed
        points_xyz = perturbed[:, :3]
        if center is None:
            center = points_xyz.mean(axis=0)
        ranges = np.linalg.norm(points_xyz, axis=1, keepdims=True)
        if max_range is None:
            max_range = ranges.max() + 1e-6
        for s, q in act:
            range_bias = (ranges / max_range) * s * self._lim_geo_range
            perturbed[:, :3] += range_bias * q["noise_direction"].reshape(1, 3)
        s_tot = act[0][0]
        for s, _ in act[1:]:
            s_tot = s_tot + s
        angle = s_tot * self._lim_geo_yaw
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot_z = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
        perturbed[:, :3] = (perturbed[:, :3] - center) @ rot_z.T + center
        scale_matrix = np.eye(3)
        for s, q in act:
            scale_factor = 1.0 + s * self._lim_geo_scala
            scale_direction = np.abs(q["cluster_direction"])
            scale_direction = scale_direction / (scale_direction.sum() + 1e-6)
            scale_matrix = scale_matrix + np.outer(scale_direction, scale_direction) * (scale_factor - 1)
        perturbed[:, :3] = (perturbed[:, :3] - center) @ scale_matrix + center
        for s, q in act:
            perturbed[:, :3] += q["noise_direction"] * s * self._lim_geo_bias
        return perturbed

    def _rampa_deriva(self, point_cloud, att):
        """_apply_temporal_drift with a single accumulator (the stream's, which
        with keep carries over between genomes) and step sum w f,
        f = d s14 max_temporal_drift for genomes with s14 >= 0.1; same cap and
        decay as the single-genome path."""
        perturbed = point_cloud.copy()
        passi = [T["w"] * (np.asarray(T["p"]["noise_direction"]) * T["p"]["temporal_drift_strength"]
                           * self._lim_drift_step)
                 for T in att if T["p"].get("temporal_drift_strength", 0) >= 0.1]
        if not passi and not self._drift_carried():
            return perturbed
        self._frame_counter += 1
        frame_drift = passi[0] if passi else np.zeros(3)
        for f in passi[1:]:
            frame_drift = frame_drift + f
        decay = 0.98
        self._accumulated_drift = self._accumulated_drift * decay + frame_drift
        norm = float(np.linalg.norm(self._accumulated_drift))
        if norm > self.max_drift_m > 0:
            self._accumulated_drift *= self.max_drift_m / norm
        perturbed[:, :3] += self._accumulated_drift
        return perturbed

    # ------------------------------------------------------------------
    # Stream state (node option --stato-generatore)
    # ------------------------------------------------------------------

    # Attributes that determine the following clouds of the stream: drift,
    # tracker (k, pose, tracked anchors and zones, previous cloud), installed
    # zones and their intensity, genome ramp. The step (_gradino, genome in
    # motion) enters the state only when set, so without it state and
    # fingerprint are unchanged.
    STATO_FLUSSO = ("_accumulated_drift", "_frame_counter", "_structs", "_struct_reset_pending",
                    "_obiettivi", "zone_stato", "zone_hash", "_zone_int", "_zone_n",
                    "rampa_spec", "rampa_info", "_rampa", "_rampa_info_metro")
    STATO_GRADINO = "_gradino"

    @staticmethod
    def _memo_condivisi(d):
        """Objects shared by a state copy because nothing modifies them after
        creation: the zones (Zone) and the alignment target of the next cloud
        (tracker.prev, with its KD-tree)."""
        tr = d.get("_structs")
        memo = {}
        for o in (d.get("_zone_n"), (d.get("rampa_spec") or {}).get("zone"),
                  ((d.get("_gradino") or {}).get("spec") or {}).get("zone"),
                  None if tr is None else tr.prev):
            if o is not None:
                memo[id(o)] = o
        return memo

    def stato_flusso(self, copia=True):
        """Stream state (STATO_FLUSSO) to transfer to another generator.
        copia=True: deep copy, independent of this generator; copia=False:
        references, for the fingerprint only. Call between two clouds."""
        d = {a: getattr(self, a, None) for a in self.STATO_FLUSSO}
        d["_struct_reset_pending"] = bool(d["_struct_reset_pending"])
        if getattr(self, "_gradino", None) is not None:
            d[self.STATO_GRADINO] = self._gradino
        return copy.deepcopy(d, self._memo_condivisi(d)) if copia else d

    def imposta_stato_flusso(self, stato):
        """Replace this generator's state with one from stato_flusso (deep copy,
        so the source stays intact and can be reloaded): from the next cloud on,
        for the same input, the clouds match the source stream. Call between two
        clouds."""
        d = copy.deepcopy(stato, self._memo_condivisi(stato))
        for a in self.STATO_FLUSSO:
            setattr(self, a, d[a])
        self._gradino = d.get(self.STATO_GRADINO)

    def _punti_palle(self, fr, zi, quanti, n):
        """Gene 10 with attacker zones: mask of the points in the ball voxels, at
        most quanti. Priority: ball in draw order, then distance of the voxel
        centroid from the centre, then key; from the last voxel the first points
        in canonical order."""
        m = np.zeros(n, dtype=bool)
        vz = np.flatnonzero(np.isin(fr["keys_a"], zi["palle_g10"]))
        if not len(vz) or quanti <= 0:
            return m
        c = zi.get("centri_g10")
        if c is not None and len(c):
            D = np.linalg.norm(fr["cent_a"][vz][:, None, :] - c[None, :, :], axis=2)
            dentro = D <= (zi.get("raggio_g10") or 0.0) + self._st.VOXEL_M
            b = np.where(dentro.any(axis=1), dentro.argmax(axis=1), D.argmin(axis=1))
            o = np.lexsort((fr["keys_a"][vz], D[np.arange(len(vz)), b], b))
        else:
            o = np.argsort(fr["keys_a"][vz], kind="stable")
        vz = vz[o]
        cnt = fr["counts"][vz]
        cum = np.cumsum(cnt)
        k = int(np.searchsorted(cum, quanti, side="right"))
        order, starts = fr["order"], fr["starts"]
        m[np.isin(fr["vox_of_point"], vz[:k])] = True
        if k < len(vz):
            resto = quanti - (int(cum[k - 1]) if k > 0 else 0)
            u = vz[k]
            m[order[starts[u]:starts[u] + resto]] = True
        return m

    def _g13_per_punteggio(self, xyz, seme, metro, k, quanti=G13_MAX_PUNTI):
        """Sorted indices of the `quanti` zone points with the highest
        detect_edges_and_corners score, ties broken by coordinates. The internal
        sampling uses a local seed from (seed, meter, 13, k); numpy's global
        state is restored."""
        stato = np.random.get_state()
        try:
            ss = np.random.SeedSequence([int(seme or 0), int(metro or 0), 13, int(k)])
            np.random.seed(int(ss.generate_state(1)[0]))
            sc = self.detect_edges_and_corners(xyz)
        finally:
            np.random.set_state(stato)
        o = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0], -sc))
        return np.sort(o[:quanti])

    def _coherent_struct_dropout(self, fr, n, rate, quanti=None, chiave=None, frazione=None):
        """Removed points (mask over input indices) in the chosen edges, at most
        int(n * rate). a: in each voxel every other point in canonical order
        (coordinates), the first one stays; b: whole voxels, those with even
        index along the dominant axis of the anchor grid. Anchors in rule order,
        voxels in key order. Genome ramp: chiave = anchors of another genome in
        the tracker; frazione = (m, R) removes the first
        rampa_conta(m, R, limit) points of the same list."""
        st = self._st
        drop = np.zeros(n, dtype=bool)
        # quanti (attacker zones): random-mode expected removals; once the
        # alternate points are used up, the other points of the same voxels follow
        limit = int(n * rate) if quanti is None else int(quanti)
        # physical stop: never beyond MAX_DROPOUT_FISICO of the scan (does not
        # trigger at x1, where expected removals are at most 10 %)
        limit = min(limit, int(n * MAX_DROPOUT_FISICO))
        if frazione is not None:
            limit = rampa_conta(frazione[0], frazione[1], limit)
        if limit <= 0:
            return drop
        order, starts, counts = fr["order"], fr["starts"], fr["counts"]
        cand, resto = [], []
        k5 = self._chiave("g5", st.EDGE_V) if chiave is None else chiave
        nz = self._n_zone()
        for A, v in zip(self._structs.anchors.get(k5, [])[:nz],
                        fr["hits"].get(k5, [])[:nz]):
            if not len(v):
                continue
            v = v[np.argsort(fr["keys_a"][v], kind="stable")]
            if self.struct_dropout == "b":
                j = int(np.argmax(np.abs(A["axis1"])))
                kk = np.floor(fr["cent_a"][v, j] / self._structs.voxel).astype(np.int64)
                cand += [order[starts[u]:starts[u] + counts[u]] for u in v[kk % 2 == 0]]
            else:
                cand += [order[starts[u] + 1:starts[u] + counts[u]:2] for u in v]
                if quanti is not None:
                    resto += [order[starts[u]:starts[u] + counts[u]:2] for u in v]
        cand += resto
        if cand:
            c = np.concatenate(cand)
            _, first = np.unique(c, return_index=True)
            drop[c[np.sort(first)][:limit]] = True
        return drop

    def _coherent_struct_ghosts(self, perturbed, fr, t, pos, p, strategic, th, quanti=None):
        """Bases: for each hit voxel of the chosen structures, the remaining point
        closest to the voxel centroid (ties by coordinates). At most
        int(n * ghost_ratio) ghosts, voxels in key order. Fixed offset along the
        gene 0-2 direction in the anchor frame: 2.5 cm when strategic,
        alternating 3.5 and 8 cm otherwise."""
        st = self._st
        none = np.zeros(0, dtype=np.int64)
        n_ghost = int(len(perturbed) * p["ghost_ratio"]) if quanti is None else int(quanti)
        hl = [v for v in fr["hits"].get(t, [])[:self._n_zone()] if len(v)]
        if n_ghost <= 0 or not hl:
            return perturbed, none
        vop = fr["vox_of_point"]
        idx = np.flatnonzero(np.isin(vop, np.concatenate(hl)) & (pos >= 0))
        if not len(idx):
            return perturbed, none
        pa = fr["xyz_pred"][idx]
        vi = vop[idx]
        d2 = ((pa - fr["cent"][vi]) ** 2).sum(1)
        o = np.lexsort((pa[:, 2], pa[:, 1], pa[:, 0], d2, fr["keys_a"][vi]))
        kv = fr["keys_a"][vi][o]
        if quanti is None:
            bases = idx[o[np.r_[True, kv[1:] != kv[:-1]]]][:n_ghost]
        else:
            # attacker zones: first one point per voxel, then the second, ...
            # (the first round is the one above), up to the expected ghosts
            inizio = np.flatnonzero(np.r_[True, kv[1:] != kv[:-1]])
            rango = np.arange(len(kv)) - np.repeat(inizio, np.diff(np.r_[inizio, len(kv)]))
            o2 = np.lexsort((kv, rango))
            bases = idx[o[o2]][:n_ghost]
        d = st.se2_rot(-th, np.asarray(p["noise_direction"], dtype=np.float64))
        k = len(bases)
        if strategic:
            offsets = np.tile(d * 0.025, (k, 1))
        else:
            amp = np.where(np.arange(k) % 2 == 0, 0.035, 0.08)
            offsets = amp[:, None] * d[None, :]
        g = perturbed[pos[bases]].copy()
        g[:, :3] += _clip_offsets(offsets)
        if g.shape[1] > 3:
            g[:, 3] = np.clip(g[:, 3], 0, 255)
        return np.vstack([perturbed, g]), bases

    def reset_temporal_state(self):
        """Reset the accumulated drift and frame counter before a new sequence."""
        self._accumulated_drift = np.zeros(3)
        self._frame_counter = 0

    def set_temporal_state(self, drift):
        """Set the accumulated drift (keep mode: the twin starts from the
        victim's drift)."""
        self._accumulated_drift = np.asarray(drift, dtype=np.float64).copy()

    def _drift_carried(self) -> bool:
        return bool(getattr(self, "keep_drift", False)) and bool(np.any(self._accumulated_drift != 0.0))

    @property
    def accumulated_drift(self) -> np.ndarray:
        """Temporal drift accumulated so far (m, sensor frame)."""
        return self._accumulated_drift.copy()

    def _apply_geometric_distortion(
        self, point_cloud: np.ndarray, params: Dict[str, any], center=None, max_range=None
    ) -> np.ndarray:
        """
        Apply systematic distortions to the whole cloud (global transformation,
        unlike the per-point operators).

        Inspired by the ICP adversarial attack: ICP tolerates random noise but not
        systematic distortion. geometric_distortion scales four terms: range-dependent
        bias (up to 5 cm at max range), yaw rotation (up to GEOM_MAX_YAW_RAD, ~6 deg),
        non-uniform scaling (up to GEOM_MAX_SCALE, 6%) and a constant per-scan bias
        (up to 2 cm).
        """
        perturbed = point_cloud.copy()
        n_points = len(perturbed)
        if n_points < 10:
            return perturbed

        distortion_strength = params.get("geometric_distortion", 0.0)
        if distortion_strength < 0.01:
            return perturbed

        points_xyz = perturbed[:, :3]
        # In coherent mode centre and normalization range are those of the first
        # cloud of the meter, passed in by the caller.
        if center is None:
            center = points_xyz.mean(axis=0)

        # Range from the sensor (sensor at the origin)
        ranges = np.linalg.norm(points_xyz, axis=1, keepdims=True)
        if max_range is None:
            max_range = ranges.max() + 1e-6

        # 1. Range-dependent bias: farther points are pushed more, as in a
        #    calibration error. Up to 5 cm at max range with full distortion.
        range_bias = (ranges / max_range) * distortion_strength * self._lim_geo_range
        direction = params["noise_direction"].reshape(1, 3)
        perturbed[:, :3] += range_bias * direction

        # 2. Yaw rotation around the cloud centre, up to GEOM_MAX_YAW_RAD (~6 deg)
        angle = distortion_strength * self._lim_geo_yaw
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot_z = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
        perturbed[:, :3] = (perturbed[:, :3] - center) @ rot_z.T + center

        # 3. Non-uniform scaling along cluster_direction, up to GEOM_MAX_SCALE (6%)
        scale_factor = 1.0 + distortion_strength * self._lim_geo_scala
        scale_direction = np.abs(params["cluster_direction"])
        scale_direction = scale_direction / (scale_direction.sum() + 1e-6)
        scale_matrix = np.eye(3) + np.outer(scale_direction, scale_direction) * (scale_factor - 1)
        perturbed[:, :3] = (perturbed[:, :3] - center) @ scale_matrix + center

        # 4. Constant per-scan bias along noise_direction, up to 2 cm
        drift_bias = params["noise_direction"] * distortion_strength * self._lim_geo_bias
        perturbed[:, :3] += drift_bias

        return perturbed

    def _apply_spatial_correlation(
        self, points: np.ndarray, noise: np.ndarray, correlation: float, voxel: float = 0.5
    ) -> np.ndarray:
        """
        Make the noise spatially correlated so nearby points move similarly.

        Local correlation per voxel: points in the same voxel (side `voxel` m)
        share the mean of their noise, blended with the per-point noise by
        `correlation`. A global mean would only add a constant offset, already
        produced by the directional bias and the geometric distortion; here
        nearby points move together and distant ones do not. O(N). Skipped
        below 0.3.
        """
        if correlation < 0.3 or len(points) < 100:
            return noise

        keys = np.floor(points[:, :3] / voxel).astype(np.int64)
        # One integer key per voxel; inv maps each point to its group.
        inv = self._voxel_inverse(keys) if getattr(self, "fast_ops", False) else None
        if inv is None:
            _, inv = np.unique(keys, axis=0, return_inverse=True)
        ng = inv.max() + 1
        sums = np.zeros((ng, 3))
        np.add.at(sums, inv, noise)
        counts = np.bincount(inv, minlength=ng).reshape(-1, 1)
        voxel_mean = sums / np.maximum(counts, 1)
        local_shift = voxel_mean[inv]
        # Blend: at correlation = 1 the noise becomes the voxel mean.
        return noise * (1 - correlation) + local_shift * correlation

    @staticmethod
    def _voxel_inverse(keys):
        """Inverse of np.unique(keys, axis=0) computed from a 1D int64 key.

        The three components, shifted to start at zero, are combined into one
        integer with the same lexicographic order as the rows, so groups and
        numbering match axis=0. None if the key does not fit in int64 (the
        caller then falls back to axis=0).
        """
        lo = keys.min(axis=0)
        hi = keys.max(axis=0)
        span = [int(h) - int(l) + 1 for l, h in zip(lo, hi)]
        if span[0] * span[1] * span[2] >= 2 ** 62:
            return None
        k = keys - lo
        flat = (k[:, 0] * span[1] + k[:, 1]) * span[2] + k[:, 2]
        _, inv = np.unique(flat, return_inverse=True)
        return inv

    def _apply_cluster_perturbation(
        self, point_cloud: np.ndarray, direction: np.ndarray, strength: float
    ) -> np.ndarray:
        """
        Displace random clusters of points, as in a localised sensor error.

        cluster_strength scales a shift along cluster_direction (std cluster_shift_std,
        3 cm) with an exponential falloff from each of the n_clusters centres; the
        cluster radius is the 10th percentile of the distances to the cloud centre.
        """
        perturbed = point_cloud.copy()
        n_points = len(perturbed)

        if n_points < 100:
            return perturbed

        # Random cluster centres
        n_clusters = min(self.n_clusters, n_points // 100)
        cluster_centers_idx = np.random.choice(n_points, n_clusters, replace=False)

        # Cluster radius
        cloud_center = perturbed[:, :3].mean(axis=0)
        distances_to_center = np.linalg.norm(perturbed[:, :3] - cloud_center, axis=1)
        cluster_radius = np.percentile(distances_to_center, 10)

        for center_idx in cluster_centers_idx:
            center = perturbed[center_idx, :3]

            distances = np.linalg.norm(perturbed[:, :3] - center, axis=1)
            mask = distances < cluster_radius

            if mask.sum() > 0:
                # Cluster-specific displacement: directed part plus a random part
                cluster_shift = direction * strength * self._lim_cluster_std
                cluster_shift += np.random.randn(3) * self._lim_cluster_std * strength * 0.5

                # Exponential falloff from the cluster centre
                falloff = np.exp(-distances[mask] / cluster_radius)
                perturbed[mask, :3] += cluster_shift * falloff[:, np.newaxis]

        return perturbed

    def _generate_ghost_points(self, point_cloud: np.ndarray, n_ghost: int) -> np.ndarray:
        """
        Generate ghost points around random existing points.

        Half are near-duplicates (3.5 cm std) that create ambiguous matches, half
        are outliers (15 cm std) that add false geometric features.
        """
        base_indices = np.random.choice(len(point_cloud), n_ghost, replace=True)
        ghost_points = point_cloud[base_indices].copy()

        n_near = n_ghost // 2
        n_far = n_ghost - n_near

        # Near-duplicates: small offsets
        if n_near > 0:
            near_offsets = np.random.randn(n_near, 3) * 0.035  # 3.5 cm std
            ghost_points[:n_near, :3] += _clip_offsets(near_offsets)

        # Outliers: larger offsets, still within the plausibility bound
        if n_far > 0:
            far_offsets = np.random.randn(n_far, 3) * 0.08  # 8 cm std
            ghost_points[n_near:, :3] += _clip_offsets(far_offsets)

        # Plausible but different intensity
        ghost_points[:, 3] += np.random.randn(n_ghost) * 20
        ghost_points[:, 3] = np.clip(ghost_points[:, 3], 0, 255)

        return ghost_points

    @staticmethod
    def nn_distances(original: np.ndarray, perturbed: np.ndarray):
        """Nearest-neighbour distances in both directions (perturbed -> clean,
        clean -> perturbed): the Chamfer queries, passed as nn to the Chamfer and
        to the node's perceptibility metrics so the two KD-trees are built once.
        None if a cloud is empty."""
        if len(original) == 0 or len(perturbed) == 0:
            return None
        dist_forward, _ = cKDTree(original[:, :3]).query(perturbed[:, :3], k=1, workers=-1)
        dist_backward, _ = cKDTree(perturbed[:, :3]).query(original[:, :3], k=1, workers=-1)
        return dist_forward, dist_backward

    def compute_chamfer_distance(self, original: np.ndarray, perturbed: np.ndarray, nn=None) -> float:
        """
        Compute the bidirectional Chamfer distance between two point clouds.

        CD(A, B) = (1/|A|) * sum min ||a - b||^2 + (1/|B|) * sum min ||b - a||^2

        Lower values mean a less perceptible perturbation.

        Args:
            original: Original point cloud (N, 3+)
            perturbed: Perturbed point cloud (M, 3+)

        Returns:
            Chamfer distance (sum of mean squared nearest-neighbour distances, m^2)
        """
        if len(original) == 0 or len(perturbed) == 0:
            return float("inf")

        if nn is not None:
            # distances already computed by nn_distances on the same clouds
            dist_forward, dist_backward = nn
            return (dist_forward**2).mean() + (dist_backward**2).mean()

        tree_orig = cKDTree(original[:, :3])
        tree_pert = cKDTree(perturbed[:, :3])

        # Forward: for each perturbed point, nearest original point
        dist_forward, _ = tree_orig.query(perturbed[:, :3], k=1, workers=-1)

        # Backward: for each original point, nearest perturbed point
        dist_backward, _ = tree_pert.query(original[:, :3], k=1, workers=-1)

        chamfer = (dist_forward**2).mean() + (dist_backward**2).mean()

        return chamfer

    def compute_perturbation_magnitude(
        self, original: np.ndarray, perturbed: np.ndarray, params: Dict[str, any], nn=None
    ) -> float:
        """
        Compute the perturbation magnitude used as the NSGA-III perceptibility objective.

        Formula, with CD the Chamfer distance in m^2 and r = |M - N| / N the relative
        change in point count (dropout and ghost points); with structural =
        "gross" r = (removed + added) / N (point_ratios):
            chamfer_cm    = sqrt(CD * 10000)
            structural_cm = r * 20        (10% change in point count = 2 cm)
            magnitude     = sqrt(chamfer_cm^2 + structural_cm^2)

        Args:
            original: Original point cloud
            perturbed: Perturbed point cloud
            params: Perturbation parameters (unused, kept for interface compatibility)

        Returns:
            Combined perturbation magnitude in cm
        """
        chamfer_m2 = self.compute_chamfer_distance(original, perturbed, nn=nn)

        # m^2 -> cm
        chamfer_cm = np.sqrt(chamfer_m2 * 10000)

        # Structural term: change in point count relative to the original cloud
        n_orig = len(original)
        n_pert = len(perturbed)
        point_change_ratio = abs(n_pert - n_orig) / max(n_orig, 1)
        if self.structural == "gross":
            point_change_ratio = self._structural_ratio(original, perturbed, point_change_ratio)

        # Scaled so that a 10% change in point count weighs like 2 cm of Chamfer
        structural_penalty_cm = point_change_ratio * 20.0

        # Euclidean combination of displacement and structural terms
        total_perturbation_cm = np.sqrt(chamfer_cm**2 + structural_penalty_cm**2)

        return total_perturbation_cm

    def compute_perturbation_terms(
        self, original: np.ndarray, perturbed: np.ndarray, params: Dict[str, any], nn=None
    ) -> Dict[str, float]:
        """
        Separate terms of compute_perturbation_magnitude, with the same operations
        in the same order, so "magnitude_cm" equals that function's value.

        Keys: chamfer_m2 (CD), chamfer_cm (sqrt(CD*1e4), pure Chamfer),
        point_ratio (r = |M - N| / N), structural_cm (20 r), magnitude_cm,
        n_orig (N), n_pert (M), point_ratio_gross ((removed + added) / N),
        n_removed, n_added (these three None without counts), structural (form
        used). point_ratio stays the net value even with "gross"; structural_cm
        is 20 times the r of the form used, so
        magnitude_cm = sqrt(chamfer_cm^2 + structural_cm^2).
        """
        chamfer_m2 = self.compute_chamfer_distance(original, perturbed, nn=nn)
        chamfer_cm = np.sqrt(chamfer_m2 * 10000)
        n_orig = len(original)
        n_pert = len(perturbed)
        point_change_ratio = abs(n_pert - n_orig) / max(n_orig, 1)
        _, r_gross, counts = self.point_ratios(original, perturbed)
        r_used = point_change_ratio
        if self.structural == "gross" and r_gross is not None:
            r_used = r_gross
        structural_penalty_cm = r_used * 20.0
        total_perturbation_cm = np.sqrt(chamfer_cm**2 + structural_penalty_cm**2)
        return {"chamfer_m2": float(chamfer_m2), "chamfer_cm": float(chamfer_cm),
                "point_ratio": float(point_change_ratio),
                "structural_cm": float(structural_penalty_cm),
                "magnitude_cm": float(total_perturbation_cm),
                "n_orig": int(n_orig), "n_pert": int(n_pert),
                "point_ratio_gross": None if r_gross is None else float(r_gross),
                "n_removed": None if counts is None else int(counts["n_removed"]),
                "n_added": None if counts is None else int(counts["n_added"]),
                "structural": self.structural}

    def random_genome(self, size: Optional[int] = None) -> np.ndarray:
        """
        Generate random genome(s).

        Args:
            size: Number of genomes to generate (None for a single genome)

        Returns:
            Random genome(s) in range [-1, 1]
        """
        if size is None:
            return np.random.uniform(-1, 1, self.get_genome_size())
        return np.random.uniform(-1, 1, (size, self.get_genome_size()))
