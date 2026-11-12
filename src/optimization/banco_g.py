"""Bench test of the full-state twin (variant G) on a saved window (--g-banco).

The twin loads a .molastate saved by a run and performs nominal rollouts only
(attack off), with no victim, shadow, attacker or search. Each case is a saved
state with the poses of its window; each arm sets the twin's physical start
pose and the allowed MOLA variables, and may carry a genome applied from the
first cloud and the initial time drift (e.g. the clouds the victim received at
the end of the previous window). Only the ROS-free parts live here: spec
loading and validation, physical targets, rollout order, ICP printf lines.

Optional spec keys (without them spec, order and records are unchanged): extra
start modes (true saved victim pose, pose saved by the bench in the same
rollout, restore only, or a realigned twin that restarts from the end-of-settling
Isaac snapshot of another arm), offsets in the robot frame, roll/pitch handling
on set_state, the attacker zones saved by the run, per-arm repetitions,
victim-vs-twin comparisons, the genome ramp (--rampa-genoma) and the branch
taken by the twin MOLA as read from its ICP log lines.
"""

import json
import math
import re
import types
from pathlib import Path

import numpy as np

from src.optimization import twin_variants as tv
from src.perturbations import zone_attaccante as za

SPEC_VERSION = 1
INIT_AMMESSI = ("attacker_est", "ombra_est", "vera", "ripristino", "salvata_banco", "riallineato")
# Only the zero-twist negative control: other variables would change the MOLA
# configuration or the state load.
MOLA_ENV_AMMESSE = ("MOLA_FULL_STATE_PROVA_TWIST_ZERO",)
# Arm keys; genome and drift_init (m, sensor frame) are optional
CHIAVI_BRACCIO_V1 = ("nome", "init", "mola_env", "nota", "genome", "drift_init")
# Optional extension keys
OFFSET_CHIAVI = ("avanti_mm", "lato_mm", "yaw_deg")
CHIAVI_BRACCIO_NUOVE = OFFSET_CHIAVI + ("assetto", "zone", "rampa", "ripetizioni", "riallinea_da",
                                       "riallinea_posa")
# Physical pose of a realigned arm after restoring the end-of-settling snapshot:
# the case's attacker estimate (as in the main loop) or no set_state
RIALLINEA_POSE = ("attacker_est", "ripristino")
CHIAVI_BRACCIO = CHIAVI_BRACCIO_V1 + CHIAVI_BRACCIO_NUOVE
CHIAVI_CASO_NUOVE = ("metro", "drift_reset_on", "confronti")
# genome group 3 (all genes)
GENOMA_LEN = 17
# drift cap: 0.5 s x 0.8 m/s = 40 cm; above 1 m it is a unit error
DERIVA_MAX_M = 1.0
PRIME_ICP = 10
RX_ICP = re.compile(r"timestamp=\S*?(\d+:\d+:[\d.]+)\s.*?goodness=([\d.]+).*?isGood=([01])")

# Start modes with set_state after restore; those accepting robot-frame offsets
INIT_SET_STATE = ("attacker_est", "ombra_est", "vera", "salvata_banco")
INIT_OFFSET = ("attacker_est", "vera")
# Offsets at most 1 cm and 0.005 rad: above that it is a unit error
OFFSET_MAX_MM = 10.0
OFFSET_MAX_YAW_DEG = math.degrees(0.005)
ASSETTO_MODI = ("azzerato", "conservato", "esplicito", "esplicito_relativo", "esplicito_attaccante_relativo")
# Attacker roll/pitch variants: 3a = roll and pitch of the attacker MOLA as they
# are (threat-model compliant), 3b = composed with the true spawn attitude
# (diagnostic only, uses ground truth)
VARIANTE_ASSETTO = {"esplicito_attaccante_relativo": "3a, adottabile",
                    "esplicito_relativo": "3b diagnostica, non adottabile"}
# Roll and pitch of a robot standing on the floor are tenths of a degree; above
# 0.05 rad (2.9 deg) it is almost surely a unit error (the server rejects
# above 0.5 rad, isaac/assetto.py)
ASSETTO_MAX_RAD = 0.05
ZONA_CHIAVI = ("path", "sha256", "hash", "genoma_sha", "seme", "metro", "fotografia_path", "fotografia_sha256")
DRIFT_RESET_AMMESSI = ("genome", "keep", "drive")

# Branch outcome: dy = y(first cloud + 0.9 s) - y(first cloud) on published
# poses (ICP kind=0 line followed by pathStep with isGood=1); victim branch
# above +10 cm, twin branch below -10 cm. If a pose is missing, the nearest
# published one within 0.2 s.
RAMO_DT_S = 0.9
RAMO_SOGLIA_CM = 10.0
RAMO_TOL_S = 0.2
RAMO_PRIME = 20
# ROS warning on the same terminal as MOLA: it cuts lines in half, and they
# continue after the warning
RX_AVVISO_TF = re.compile(r"\[WARN\] \[[\d.]+\] \[\]: TF_OLD_DATA ignoring data from the past for frame base_link at time "
                          r"[\d.]+ according to authority Authority undetectable\n(?:Possible reasons are listed at \S+\n)?")
RX_SEGNALE = re.compile(r"\[INFO\] \[[\d.]+\] \[rclcpp\]: signal_handler\([^)\n]*\)\n")
RX_OSS = re.compile(r"\|Generator \(raw\)\] Processing observation .* stamp=([\d.]+)")
RX_NVOX = re.compile(r"\|FilterDecimateAdaptive \(map\)\] output '[^']*': nVoxels=(\d+)")
RX_ICP0 = re.compile(r"\|LidarOdometry\] ICP \(kind=0\): goodness=([\d.]+)% iters=(\d+) "
                     r"pose=\[(\S+) (\S+) (\S+) (\S+) (\S+) (\S+)\]"
                     r"(?:.*?sigmas:\{(\S+) (\S+) (\S+) \[m\] (\S+) (\S+) (\S+) \[deg\]\})?")
RX_PASSO = re.compile(r"pathStep=(\d+) timestamp=\S*?(\d+):(\d+):([\d.]+)")
RX_GOOD = re.compile(r" isGood=([01])")

# Acceptance criterion A: victim-twin thresholds (cm) on stop point (R1),
# lateral offset at equal true path length and damage
SOGLIE_CRITERIO_A = {"R1_cm": 3.0, "laterale_cm": 3.0, "danno_cm": 1.5}
# Full criterion A: adds the incremental E1 (3.5 cm), and damage counts only when
# the victim damage was measured (dm_real; otherwise not measurable)
SOGLIE_CRITERIO_A_COMPLETO = {"R1_cm": 3.0, "E1_inc_cm": 3.5, "laterale_cm": 3.0, "danno_cm": 1.5}
# In-repo copy of analisi_accettazione.py (md5 3609e146e2e5), the reference
# E1 method (misure_criterio_A.py, lines 117-128)
ANALISI_ACCETTAZIONE = Path(__file__).resolve().parents[2] / "scripts" / "analisi_accettazione_esito30ott.py"
_NON_DATO = object()

MOTIVO_RIALLINEATO = (
    "init riallineato (D-31c): il momento 'fine assestamento' e' la N-esima nuvola da fermo, come nel "
    "flusso principale: serve --settle-clouds > 0 (e il server di Isaac con save_nome)")
MOTIVO_RAMPA = ("rampa: parametri del formato riservato (prima della rampa A3); il formato del banco e' "
                "{genoma_prec, zone_prec, fonte}")
# Ramp in the spec: previous genome (17 genes) and its zones on the metre
# snapshot of the case (same keys as the arm zones)
RAMPA_CHIAVI = ("genoma_prec", "zone_prec", "fonte")


def _terna(v, nome, n=3):
    if not isinstance(v, (list, tuple)) or len(v) != n:
        raise ValueError(f"{nome}: attesi {n} valori")
    out = [float(x) for x in v]
    if not all(math.isfinite(x) for x in out):
        raise ValueError(f"{nome}: valori non finiti")
    return out


def _nome(v):
    return isinstance(v, str) and re.fullmatch(r"[A-Za-z0-9_]+", v) is not None


def _numero(v, nome):
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(float(v)):
        raise ValueError(f"{nome}: numero finito atteso")
    return float(v)


def _esa64(v):
    return re.fullmatch(r"[0-9a-f]{64}", str(v or "")) is not None


def _check_assetto(a, dove):
    if not isinstance(a, dict) or a.get("modo") not in ASSETTO_MODI:
        raise ValueError(f"{dove}: assetto = {{modo: {'|'.join(ASSETTO_MODI)}, ...}}")
    modo = a["modo"]
    angoli = {"esplicito": ("roll_rad", "pitch_rad"),
              "esplicito_attaccante_relativo": ("roll_rad", "pitch_rad"),
              "esplicito_relativo": ("roll_rad", "pitch_rad", "yaw_rad")}.get(modo, ())
    extra = sorted(set(a) - {"modo", "fonte", "nota"} - set(angoli))
    if extra:
        raise ValueError(f"{dove}: assetto {modo} con chiavi non gestite {extra}")
    out = {"modo": modo}
    for k in angoli:
        v = _numero(a.get(k), f"{dove}, assetto.{k}")
        if k != "yaw_rad" and abs(v) > ASSETTO_MAX_RAD:
            raise ValueError(f"{dove}: assetto.{k} {v} oltre {ASSETTO_MAX_RAD} rad (gradi al posto di radianti?)")
        out[k] = v
    if angoli:
        # the value comes from data: rejected without a source
        if not a.get("fonte"):
            raise ValueError(f"{dove}: assetto {modo} senza fonte (file e riga da cui viene)")
    for k in ("fonte", "nota"):
        if k in a:
            out[k] = a[k]
    return out


def _check_zone(z, genome, metro, dove):
    if not isinstance(z, dict):
        raise ValueError(f"{dove}: zone = {{{', '.join(ZONA_CHIAVI)}}}")
    extra = sorted(set(z) - set(ZONA_CHIAVI) - {"fonte"})
    mancano = [k for k in ZONA_CHIAVI if k not in z]
    if extra or mancano:
        raise ValueError(f"{dove}: zone con chiavi non gestite {extra} o mancanti {mancano}")
    for k in ("path", "fotografia_path"):
        if not isinstance(z[k], str) or not z[k]:
            raise ValueError(f"{dove}: zone.{k} assente")
    for k in ("sha256", "hash", "fotografia_sha256"):
        if not _esa64(z[k]):
            raise ValueError(f"{dove}: zone.{k} di 64 cifre esadecimali")
    if z["genoma_sha"] != za.genoma_sha(genome):
        raise ValueError(f"{dove}: zone.genoma_sha {z['genoma_sha']} non e' quello del genoma ({za.genoma_sha(genome)})")
    if not isinstance(z["seme"], int) or isinstance(z["seme"], bool) or z["seme"] != za.seme_genoma(genome):
        raise ValueError(f"{dove}: zone.seme {z['seme']} non e' quello del genoma ({za.seme_genoma(genome)})")
    if z["metro"] != metro:
        raise ValueError(f"{dove}: zone.metro {z['metro']} diverso dal metro del caso {metro}")


def rampa_nuova(rp):
    """True if the ramp uses the bench format (previous genome and its zones)."""
    return isinstance(rp, dict) and "genoma_prec" in rp


def _check_rampa(rp, dove, metro=None, genome=None):
    if not isinstance(rp, dict) or not rp:
        raise ValueError(f"{dove}: rampa = dizionario non vuoto di parametri")
    if rampa_nuova(rp):
        extra = sorted(set(rp) - set(RAMPA_CHIAVI))
        if extra or "zone_prec" not in rp:
            raise ValueError(f"{dove}: rampa con chiavi non gestite {extra} o senza zone_prec")
        if genome is None or metro is None:
            raise ValueError(f"{dove}: rampa senza genome nel braccio o senza metro nel caso")
        g = _terna(rp["genoma_prec"], f"{dove}, rampa.genoma_prec", GENOMA_LEN)
        if any(abs(x) > 1.0 for x in g):
            raise ValueError(f"{dove}: rampa.genoma_prec fuori da [-1, 1]")
        _check_zone(rp["zone_prec"], g, metro, f"{dove}, rampa.zone_prec")
        rp["genoma_prec"] = g
        return
    for k, v in rp.items():
        if not _nome(k):
            raise ValueError(f"{dove}: rampa, nome di parametro non valido {k!r}")
        if isinstance(v, bool) or (isinstance(v, str) and len(v) <= 64):
            continue
        _numero(v, f"{dove}, rampa.{k}")


def _stesso_attacco(a, b):
    """Same attack in both arms: genome, initial drift, zones and ramp."""
    return all(a.get(k) == b.get(k) for k in ("genome", "drift_init", "zone", "rampa"))


def check_spec(spec):
    """Validated and normalised spec; ValueError at the first problem."""
    if not isinstance(spec, dict) or spec.get("versione") != SPEC_VERSION:
        raise ValueError(f"specifica: versione attesa {SPEC_VERSION}")
    n = spec.get("ripetizioni")
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("specifica: 'ripetizioni' intero >= 1")
    casi = spec.get("casi")
    if not isinstance(casi, list) or not casi:
        raise ValueError("specifica: 'casi' vuoto")
    # optional pause before each rollout of repetitions [da_r, a_r): a different
    # pace within the same test
    pz = spec.get("pausa")
    if pz is not None:
        if not isinstance(pz, dict) or sorted(pz) != ["a_r", "da_r", "s"]:
            raise ValueError("specifica: 'pausa' = {da_r, a_r, s}")
        da, a, sec = pz["da_r"], pz["a_r"], pz["s"]
        if not all(isinstance(x, int) and not isinstance(x, bool) for x in (da, a)) or not 0 <= da <= a <= n:
            raise ValueError("specifica: pausa con 0 <= da_r <= a_r <= ripetizioni")
        if not isinstance(sec, (int, float)) or isinstance(sec, bool) or not 0 <= sec <= 600:
            raise ValueError("specifica: pausa.s fra 0 e 600 s")
        spec["pausa"] = {"da_r": da, "a_r": a, "s": float(sec)}
    nomi = set()
    for c in casi:
        nome = c.get("nome")
        if not _nome(nome) or nome in nomi:
            raise ValueError(f"caso '{nome}': nome mancante, non valido o ripetuto")
        nomi.add(nome)
        if not isinstance(c.get("state_path"), str):
            raise ValueError(f"caso {nome}: state_path assente")
        if not re.fullmatch(r"[0-9a-f]{64}", str(c.get("state_sha256", ""))):
            raise ValueError(f"caso {nome}: state_sha256 di 64 cifre esadecimali")
        for k in ("saved_odom", "saved_scene", "origin", "believed", "attacker_target_scene"):
            c[k] = _terna(c.get(k), f"caso {nome}, {k}")
        c["pose6"] = _terna(c.get("pose6"), f"caso {nome}, pose6", 6)
        metro = c.get("metro")
        if metro is not None and (not isinstance(metro, int) or isinstance(metro, bool) or metro < 0):
            raise ValueError(f"caso {nome}: metro intero >= 0")
        if c.get("drift_reset_on") is not None and c["drift_reset_on"] not in DRIFT_RESET_AMMESSI:
            raise ValueError(f"caso {nome}: drift_reset_on in {DRIFT_RESET_AMMESSI}")
        bracci = c.get("bracci")
        if not isinstance(bracci, list) or not bracci:
            raise ValueError(f"caso {nome}: 'bracci' vuoto")
        nb = {}
        for b in bracci:
            bn = b.get("nome")
            if not _nome(bn) or bn in nb:
                raise ValueError(f"caso {nome}: braccio '{bn}' senza nome valido o ripetuto")
            nb[bn] = b
            dove = f"caso {nome}, braccio {bn}"
            extra = sorted(set(b) - set(CHIAVI_BRACCIO))
            if extra:
                raise ValueError(f"{dove}: chiavi non gestite {extra}")
            if b.get("init") not in INIT_AMMESSI:
                raise ValueError(f"{dove}: init in {INIT_AMMESSI}")
            env = b.get("mola_env") or {}
            extra = sorted(set(env) - set(MOLA_ENV_AMMESSE))
            if extra:
                raise ValueError(f"caso {nome}, braccio {bn}: variabili non ammesse {extra}")
            b["mola_env"] = {str(k): str(v) for k, v in env.items()}
            if "genome" in b:
                g = _terna(b["genome"], f"caso {nome}, braccio {bn}, genome", GENOMA_LEN)
                if any(abs(x) > 1.0 for x in g):
                    raise ValueError(f"caso {nome}, braccio {bn}: genome fuori da [-1, 1]")
                b["genome"] = g
            if "drift_init" in b:
                if "genome" not in b:
                    raise ValueError(f"caso {nome}, braccio {bn}: drift_init senza genome "
                                     "(nel nominale la deriva non entra nella nuvola)")
                d = _terna(b["drift_init"], f"caso {nome}, braccio {bn}, drift_init")
                if math.sqrt(sum(x * x for x in d)) > DERIVA_MAX_M:
                    raise ValueError(f"caso {nome}, braccio {bn}: drift_init oltre {DERIVA_MAX_M} m")
                b["drift_init"] = d
            # --- extension keys
            off = [k for k in OFFSET_CHIAVI if k in b]
            if off and b["init"] not in INIT_OFFSET:
                raise ValueError(f"{dove}: scarti {off} solo con init in {INIT_OFFSET}")
            for k in off:
                v = _numero(b[k], f"{dove}, {k}")
                lim = OFFSET_MAX_YAW_DEG if k == "yaw_deg" else OFFSET_MAX_MM
                if abs(v) > lim:
                    raise ValueError(f"{dove}: {k} {v} oltre {round(lim, 4)}")
                b[k] = v
            if "riallinea_posa" in b:
                if b["init"] != "riallineato" or b["riallinea_posa"] not in RIALLINEA_POSE:
                    raise ValueError(f"{dove}: riallinea_posa in {RIALLINEA_POSE}, solo con init riallineato")
            if "assetto" in b:
                con_set_state = b["init"] in INIT_SET_STATE or (
                    b["init"] == "riallineato" and riallinea_posa(b) == "attacker_est")
                if not con_set_state:
                    raise ValueError(f"{dove}: assetto solo con init in {INIT_SET_STATE} (con set_state)")
                b["assetto"] = _check_assetto(b["assetto"], dove)
            if "zone" in b:
                if "genome" not in b:
                    raise ValueError(f"{dove}: zone senza genome")
                if metro is None:
                    raise ValueError(f"{dove}: zone senza metro nel caso")
                _check_zone(b["zone"], b["genome"], metro, dove)
            if "rampa" in b:
                _check_rampa(b["rampa"], dove, metro, b.get("genome"))
            if "ripetizioni" in b:
                rb = b["ripetizioni"]
                if not isinstance(rb, int) or isinstance(rb, bool) or not 1 <= rb <= n:
                    raise ValueError(f"{dove}: ripetizioni intero fra 1 e {n}")
            if (b["init"] == "riallineato") != ("riallinea_da" in b):
                raise ValueError(f"{dove}: init riallineato e riallinea_da vanno insieme")
            if b["init"] == "riallineato":
                v = nb.get(b["riallinea_da"])
                if v is None or v is b or v.get("init") != "ripristino":
                    raise ValueError(f"{dove}: riallinea_da deve nominare un braccio precedente con init ripristino")
                if not _stesso_attacco(v, b) or v.get("ripetizioni") != b.get("ripetizioni"):
                    raise ValueError(f"{dove}: genoma, deriva, zone, rampa e ripetizioni diversi da {b['riallinea_da']}")
        if any("zone" in b for b in bracci) and any("genome" in b and "zone" not in b for b in bracci):
            raise ValueError(f"caso {nome}: zone in alcuni bracci con genoma e non in altri")
        conf = c.get("confronti")
        if conf is not None:
            if not isinstance(conf, list) or not conf:
                raise ValueError(f"caso {nome}: confronti = elenco non vuoto di {{nome, vittima, gemello}}")
            nc = set()
            for cf in conf:
                if not isinstance(cf, dict) or sorted(cf) != ["gemello", "nome", "vittima"]:
                    raise ValueError(f"caso {nome}: confronto = {{nome, vittima, gemello}}")
                if not _nome(cf["nome"]) or cf["nome"] in nc:
                    raise ValueError(f"caso {nome}: confronto '{cf['nome']}' senza nome valido o ripetuto")
                nc.add(cf["nome"])
                v, w = nb.get(cf["vittima"]), nb.get(cf["gemello"])
                if v is None or w is None or v is w:
                    raise ValueError(f"caso {nome}, confronto {cf['nome']}: bracci assenti o uguali")
                if not _stesso_attacco(v, w) or v.get("ripetizioni") != w.get("ripetizioni"):
                    raise ValueError(f"caso {nome}, confronto {cf['nome']}: genoma, deriva, zone, rampa o "
                                     "ripetizioni diversi fra vittima e gemello")
    return spec


def riallinea_posa(braccio):
    """Pose mode of a realigned arm: attacker_est if not given."""
    return braccio.get("riallinea_posa", "attacker_est")


def usa_riallineato(spec):
    return any(b["init"] == "riallineato" for c in spec["casi"] for b in c["bracci"])


def bracci_riallineati(caso):
    """Names of the arms a realigned arm of the case restarts from."""
    return {b["riallinea_da"] for b in caso["bracci"] if b["init"] == "riallineato"}


def usa_genoma(spec):
    return any("genome" in b for c in spec["casi"] for b in c["bracci"])


def usa_deriva(spec):
    return any("drift_init" in b for c in spec["casi"] for b in c["bracci"])


def pausa_s(spec, r):
    """Wait (s) before a rollout of repetition r (0 without a pause)."""
    pz = spec.get("pausa")
    return pz["s"] if pz and pz["da_r"] <= r < pz["a_r"] else 0.0


def usa_braccio_e(spec):
    return any(b.get("nome") == "E" for c in spec["casi"] for b in c["bracci"])


def usa_estensioni(spec):
    """True if the spec uses any extension key or start mode: only then the
    bench records attitude, branch, damage and comparisons."""
    return any(set(c) & set(CHIAVI_CASO_NUOVE) for c in spec["casi"]) or any(
        set(b) & set(CHIAVI_BRACCIO_NUOVE) or b["init"] not in ("attacker_est", "ombra_est")
        for c in spec["casi"] for b in c["bracci"])


def usa_rampa(spec):
    """True if an arm has a ramp in the bench format."""
    return any(rampa_nuova(b.get("rampa")) for c in spec["casi"] for b in c["bracci"])


def usa_zone(spec):
    return any("zone" in b for c in spec["casi"] for b in c["bracci"])


def usa_assetto(spec, modi=ASSETTO_MODI[1:]):
    return any((b.get("assetto") or {}).get("modo") in modi for c in spec["casi"] for b in c["bracci"])


def usa_confronti(spec):
    return any(c.get("confronti") for c in spec["casi"])


def ripetizioni_braccio(spec, braccio):
    return braccio.get("ripetizioni", spec["ripetizioni"])


def load_spec(path):
    with open(path) as f:
        return check_spec(json.load(f))


def check_args(args):
    """Reason --g-banco cannot be used with these arguments, None if it can or
    if the option is off."""
    if not getattr(args, "g_banco", None):
        return None
    if getattr(args, "victim", "continuous") != "restart":
        return "--g-banco richiede --victim restart (nessuna vittima, ombra o attaccante)"
    if getattr(args, "twin_state", "clean") != "clean" or getattr(args, "twin_init", "true") != "true":
        return ("--g-banco richiede --twin-state clean e --twin-init true: stato e posa fisica "
                "del gemello vengono dalla specifica")
    if getattr(args, "uniform_sim_time", "off") == "off":
        return "--g-banco richiede --uniform-sim-time true o false, come le run G"
    if getattr(args, "shadow_check", False) or getattr(args, "repeat_genome", None) \
            or getattr(args, "repeat_spec", None) or getattr(args, "map_gate_test", False):
        return "--g-banco non va con --shadow-check, --repeat-genome, --repeat-spec o --map-gate-test"
    why = tv.g_overlay_problem(getattr(args, "g_overlay", None) or tv.G_OVERLAY_ROOT)
    if why:
        return f"--g-banco: {why}"
    try:
        spec = load_spec(args.g_banco)
    except (OSError, ValueError, TypeError, AttributeError) as e:
        return f"--g-banco {args.g_banco}: {e}"
    # genome arms: genome from the first cloud, drift fixed and never reset in the
    # rollout. With keep the initial drift enters the cloud even with gene 14
    # below threshold, as in the run's twin
    if usa_genoma(spec) and (getattr(args, "attack_from", "spawn") != "spawn"
                             or getattr(args, "drift_reset_on", "genome") not in ("genome", "keep")):
        return "--g-banco con bracci a genoma richiede --attack-from spawn e --drift-reset-on genome o keep"
    # arm E keeps the victim drift as in the run (keep); with genome, gene 14
    # below threshold would leave it out of the cloud
    if usa_braccio_e(spec) and getattr(args, "drift_reset_on", "genome") != "keep":
        return "--g-banco con il braccio E richiede --drift-reset-on keep"
    return problema_estensioni(spec, args)


def problema_estensioni(spec, args):
    """Check the extension keys against the arguments (None if fine)."""
    if any("rampa" in b and not rampa_nuova(b["rampa"]) for c in spec["casi"] for b in c["bracci"]):
        return f"--g-banco: {MOTIVO_RAMPA}"
    if usa_rampa(spec):
        R = getattr(args, "rampa_genoma", 1) or 1
        n = getattr(args, "settle_clouds", 0) or 0
        if R <= 1:
            return "--g-banco: rampa nella specifica: serve --rampa-genoma R > 1 (nodo e orchestratore)"
        if not (getattr(args, "zone_attaccante", False) and getattr(args, "metro_al_nodo", False)):
            return "--g-banco con rampa nella specifica richiede --zone-attaccante e --metro-al-nodo"
        if R > n - 8:
            return f"--g-banco: --rampa-genoma {R} oltre l'assestamento meno 8 nuvole (--settle-clouds {n})"
    if usa_riallineato(spec) and not (getattr(args, "settle_clouds", 0) or 0) > 0:
        return f"--g-banco: {MOTIVO_RIALLINEATO}"
    zone_arg = bool(getattr(args, "zone_attaccante", False))
    metro_arg = bool(getattr(args, "metro_al_nodo", False))
    if usa_zone(spec) and not (zone_arg and metro_arg):
        return ("--g-banco con zone nella specifica richiede --zone-attaccante e --metro-al-nodo "
                "(nodo con --operator-mode insieme --zone-attaccante)")
    if zone_arg and any("genome" in b and "zone" not in b for c in spec["casi"] for b in c["bracci"]):
        return ("--zone-attaccante con --g-banco: ogni braccio con genoma deve avere le zone salvate "
                "nella specifica (il banco non ha la fotografia del metro dal vivo)")
    if metro_arg and any(c.get("metro") is None and any("genome" in b for b in c["bracci"]) for c in spec["casi"]):
        return "--metro-al-nodo con --g-banco: ogni caso con bracci a genoma deve avere il metro"
    dro = getattr(args, "drift_reset_on", "genome")
    for c in spec["casi"]:
        if c.get("drift_reset_on") is not None and c["drift_reset_on"] != dro:
            return f"--g-banco: il caso {c['nome']} viene da una run con --drift-reset-on {c['drift_reset_on']}, non {dro}"
    if usa_assetto(spec, ("esplicito_relativo",)) and not getattr(args, "start_pose", None):
        return ("--g-banco con assetto esplicito_relativo richiede --start-pose (posa di partenza della run, "
                "dove si misura l'assetto allo spawn)")
    if usa_confronti(spec) and not (getattr(args, "damage_ref", None) == "baseline"
                                    and getattr(args, "baseline_path", None)):
        return "--g-banco con confronti richiede --damage-ref baseline e --baseline-path (laterale e danno)"
    return None


def problema_capacita(spec, caps):
    """Missing Isaac server commands required by the spec (None if all present)."""
    caps = set(caps or ())
    if usa_assetto(spec) and "set_state_assetto" not in caps:
        return ("assetto conservato o esplicito nella specifica: il server di Isaac non ha set_state_assetto "
                "(isaac/run_isaac_standalone.py con isaac/assetto.py)")
    if usa_assetto(spec, ("esplicito_relativo",)) and "assetto" not in caps:
        return "assetto esplicito_relativo: il server di Isaac non ha il comando assetto (misura allo spawn)"
    if usa_riallineato(spec) and "save_nome" not in caps:
        return "braccio riallineato (D-31c): il server di Isaac non ha save_nome (istantanea con nome)"
    return None


def schedule(spec):
    """Rollout order: repetitions outermost, cases and arms interleaved, so a
    drift over the session does not fall on a single arm. An arm with its own
    repetition count drops out of later repetitions."""
    return [(r, c, b) for r in range(spec["ripetizioni"]) for c in spec["casi"] for b in c["bracci"]
            if r < ripetizioni_braccio(spec, b)]


def tag(r, caso, braccio):
    return f"banco_{caso['nome']}_{braccio['nome']}_r{r}"


def compone_offset(pose, avanti_mm=0.0, lato_mm=0.0, yaw_deg=0.0):
    """Pose (x, y, yaw) shifted in the robot frame: forward along the heading,
    lateral to the left (mm), extra yaw (deg)."""
    x, y, yaw = (float(v) for v in pose)
    a, l_ = avanti_mm / 1000.0, lato_mm / 1000.0
    c, s = math.cos(yaw), math.sin(yaw)
    return [x + c * a - s * l_, y + s * a + c * l_, yaw + math.radians(yaw_deg)]


def offset_relativo(base, pose):
    """Inverse of compone_offset: (avanti_mm, lato_mm, yaw_deg) of pose in the
    robot frame at base."""
    dx, dy = float(pose[0]) - float(base[0]), float(pose[1]) - float(base[1])
    c, s = math.cos(float(base[2])), math.sin(float(base[2]))
    dyaw = math.atan2(math.sin(float(pose[2]) - float(base[2])), math.cos(float(pose[2]) - float(base[2])))
    return [1000.0 * (c * dx + s * dy), 1000.0 * (-s * dx + c * dy), math.degrees(dyaw)]


def init_target(caso, braccio, salvata=None):
    """Physical twin target (set_state after restore), velocities and joints zero
    as with --twin-init-vel zero. attacker_est: scene pose logged by the run;
    ombra_est: shadow-believed pose composed with the saved run state
    (twin_variants.twin_init_target); vera: saved true victim pose (case
    saved_scene); salvata_banco: pose saved by the bench in this rollout
    (salvata, scene frame); ripristino and riallineato: no set_state
    (target_scene None). Offsets shift the target in the robot frame."""
    init = braccio["init"]
    if init == "attacker_est" or (init == "riallineato" and riallinea_posa(braccio) == "attacker_est"):
        rec = {"target_odom": None, "target_scene": list(caso["attacker_target_scene"]),
               "v_set_scene": [0.0, 0.0], "w_set": 0.0, "zero_joints": True,
               "rule": ("banco: posa della run, velocita' zero" if init == "attacker_est" else
                        "banco, riallineato: restore dell'istantanea di fine assestamento del braccio "
                        f"{braccio.get('riallinea_da')}, poi posa dell'attaccante della run, velocita' zero")}
    elif init == "ombra_est":
        rec = tv.twin_init_target(caso["saved_odom"], caso["saved_scene"], caso["believed"], None)
        rec["rule"] = "banco: posa creduta dall'ombra, velocita' zero"
    elif init in ("vera", "salvata_banco"):
        if init == "salvata_banco" and salvata is None:
            raise ValueError("init salvata_banco senza la posa salvata dal banco")
        src = caso["saved_scene"] if init == "vera" else salvata
        rec = {"target_odom": None, "target_scene": [float(v) for v in src[:3]],
               "v_set_scene": [0.0, 0.0], "w_set": 0.0, "zero_joints": True,
               "rule": ("banco: posa vera salvata della vittima, velocita' zero" if init == "vera"
                        else "banco: posa salvata dal banco in questo rollout, velocita' zero")}
    else:
        rec = {"target_odom": None, "target_scene": None,
               "rule": "banco: solo restore (procedura della vittima), nessun set_state"}
    if init in ("attacker_est", "ombra_est"):
        rec["stimatore"] = "attaccante" if init == "attacker_est" else "ombra"
    elif init == "riallineato" and rec["target_scene"] is not None:
        rec["stimatore"] = "attaccante"
    else:
        rec["stimatore"] = None
    off = {k: braccio[k] for k in OFFSET_CHIAVI if k in braccio}
    if off:
        rec["target_scene_senza_scarti"] = list(rec["target_scene"])
        rec["target_scene"] = compone_offset(rec["target_scene"], off.get("avanti_mm", 0.0),
                                             off.get("lato_mm", 0.0), off.get("yaw_deg", 0.0))
        rec["scarti"] = off
    if "assetto" in braccio:
        rec["assetto"] = dict(braccio["assetto"])
    return rec


# ------------------------------------------------------------------ attitude

def rot_zyx(roll, pitch, yaw):
    """Rotation matrix R = Rz(yaw) Ry(pitch) Rx(roll) (same convention as
    isaac/assetto.py and MRPT CPose3D in the MOLA ICP lines)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def euler_zyx(R):
    """(roll, pitch, yaw) from R = Rz Ry Rx."""
    R = np.asarray(R, dtype=np.float64)
    pitch = math.asin(max(-1.0, min(1.0, -float(R[2, 0]))))
    return math.atan2(float(R[2, 1]), float(R[2, 2])), pitch, math.atan2(float(R[1, 0]), float(R[0, 0]))


def assetto_composto(spawn, relativo):
    """World attitude from the attacker MOLA's relative one: its map starts in
    the robot frame at spawn (initial roll and pitch zero), so
    R_world = R_spawn * R_rel. spawn and relativo are (roll, pitch, yaw) in rad;
    returns (roll, pitch, yaw) in the world."""
    return euler_zyx(rot_zyx(*spawn) @ rot_zyx(*relativo))


def parametri_set_state(assetto, spawn=None):
    """Extra set_state parameters for the arm attitude: {} (azzerato, default),
    keep_tilt (conservato), roll and pitch (esplicito; esplicito_attaccante_relativo
    uses the attacker MOLA values as they are; esplicito_relativo composes them
    with the spawn attitude)."""
    modo = (assetto or {}).get("modo", "azzerato")
    if modo == "azzerato":
        return {}
    if modo == "conservato":
        return {"keep_tilt": True}
    if modo in ("esplicito", "esplicito_attaccante_relativo"):
        return {"roll": float(assetto["roll_rad"]), "pitch": float(assetto["pitch_rad"])}
    if spawn is None:
        raise ValueError("assetto esplicito_relativo senza l'assetto misurato allo spawn")
    r, p, _ = assetto_composto(spawn, (assetto["roll_rad"], assetto["pitch_rad"], assetto["yaw_rad"]))
    return {"roll": r, "pitch": p}


def shadow_state(caso, local_path, sha256, nbytes, stamp_tick):
    """shadow_state record like ShadowRunner.save_state, from the local copy of
    the .molastate (the re-save is written next to the copy)."""
    return {"status": "ok", "state_path": str(local_path), "state_sha256": sha256,
            "state_bytes": int(nbytes), "state_stamp_tick": int(stamp_tick),
            "pose6": list(caso["pose6"]), "origin": list(caso["origin"]),
            "believed": list(caso["believed"]), "pose_stamp": None,
            "source_path": caso["state_path"], "env": None}


def icp_lines(text):
    """ICP printf lines (MOLA_LO_DEBUG_ICP_QUALITY=1) in order: stamp, goodness,
    isGood."""
    out = []
    for line in (text or "").splitlines():
        if "isGood=" not in line or "goodness=" not in line:
            continue
        m = RX_ICP.search(line)
        if m:
            out.append({"stamp": m.group(1), "q": float(m.group(2)), "good": m.group(3) == "1"})
    return out


def esito(righe, n=PRIME_ICP):
    """Rollout outcome: passes if at least one ICP among the first n is good."""
    prime = righe[:n]
    return {"righe": len(righe), "buone": sum(r["good"] for r in righe),
            "prime_n": len(prime), "buone_prime_n": sum(r["good"] for r in prime),
            "q1": prime[0]["q"] if prime else None,
            "q_prime_n": [r["q"] for r in prime],
            "passa": any(r["good"] for r in prime)}


# ------------------------------------------------------------- branch outcome

def testo_senza_avvisi(text):
    """MOLA log without whole TF_OLD_DATA warnings (warning line and, if present,
    the reasons line) and without the rclcpp shutdown-signal line, which can land
    in the middle of an ICP line: the split line is rebuilt."""
    return RX_SEGNALE.sub("", RX_AVVISO_TF.sub("", text or ""))


def nuvole_icp(text):
    """One entry per processed cloud (pathStep line), with stamp, q and pose of
    the preceding 'ICP (kind=0)' line, isGood, covariance sigmas (m, m, m, deg,
    deg, deg) and nVoxels of the map decimation. The stamp comes from the
    pathStep line or, if truncated, from 'Processing observation'."""
    out, st, nv, icp = [], None, None, None
    for ln in testo_senza_avvisi(text).splitlines():
        m = RX_OSS.search(ln)
        if m:
            st, nv, icp = round(float(m.group(1)), 1), None, None
            continue
        m = RX_NVOX.search(ln)
        if m:
            nv = int(m.group(1))
            continue
        m = RX_ICP0.search(ln)
        if m:
            icp = m.groups()
            continue
        if "pathStep=" not in ln:
            continue
        m, g = RX_PASSO.search(ln), RX_GOOD.search(ln)
        s = round(int(m.group(2)) * 3600 + int(m.group(3)) * 60 + float(m.group(4)), 1) if m else st
        v = {"stamp": s, "pathStep": int(m.group(1)) if m else None,
             "q": round(float(icp[0]) / 100.0, 4) if icp else None,
             "good": (g.group(1) == "1") if g else None, "nvoxels": nv,
             "x": None, "y": None, "z": None, "yaw_deg": None, "pitch_deg": None, "roll_deg": None, "sigma": None}
        if icp:
            v.update(x=float(icp[2]), y=float(icp[3]), z=float(icp[4]), yaw_deg=float(icp[5]),
                     pitch_deg=float(icp[6]), roll_deg=float(icp[7]),
                     sigma=[float(x) for x in icp[8:14]] if icp[8] is not None else None)
        out.append(v)
        st, nv, icp = None, None, None
    return out


def _pubblicata_vicina(pub, t, tol=RAMO_TOL_S):
    vicine = [r for r in pub if abs(r["stamp"] - t) <= tol + 1e-9]
    return min(vicine, key=lambda r: abs(r["stamp"] - t)) if vicine else None


def classe_ramo(dy_cm, soglia=RAMO_SOGLIA_CM):
    if dy_cm is None:
        return "non_classificabile"
    return "ramo_vittima" if dy_cm >= soglia else "ramo_gemello" if dy_cm <= -soglia else "nessun_salto"


def esito_ramo(text, stamp0=None, n=RAMO_PRIME):
    """Branch taken by MOLA after loading: dy = y(t0 + 0.9 s) - y(t0) on the
    published poses, t0 the first cloud (from stamp0 on, for a continuous log).
    Also the series of the first n clouds: y, q, isGood, sigma, nVoxels, ICP
    pitch and roll."""
    righe = [r for r in nuvole_icp(text)
             if r["stamp"] is not None and (stamp0 is None or r["stamp"] >= stamp0 - 1e-6)]
    out = {"regola": f"dy = y(t0 + {RAMO_DT_S} s) - y(t0) sulle pose pubblicate, soglia +-{RAMO_SOGLIA_CM} cm, "
                     f"posa piu' vicina entro {RAMO_TOL_S} s",
           "righe": len(righe), "t0": None, "stamp_y0": None, "stamp_y1": None, "dy_cm": None,
           "classe": "non_classificabile", "prime": []}
    if not righe:
        return out
    t0 = righe[0]["stamp"]
    pub = [r for r in righe if r["good"] and r["y"] is not None]
    a, b = _pubblicata_vicina(pub, t0), _pubblicata_vicina(pub, round(t0 + RAMO_DT_S, 1))
    dy = round(100.0 * (b["y"] - a["y"]), 3) if a and b else None
    out.update(t0=t0, stamp_y0=a["stamp"] if a else None, stamp_y1=b["stamp"] if b else None,
               dy_cm=dy, classe=classe_ramo(dy),
               prime=[{k: r[k] for k in ("stamp", "q", "good", "y", "pitch_deg", "roll_deg", "sigma", "nvoxels")}
                      for r in righe[:n]])
    return out


def nuvola_alla_stamp(text, stamp):
    """nuvole_icp entry at the given stamp (None if missing): the attitude a MOLA
    estimated at that stamp."""
    return next((r for r in nuvole_icp(text) if r["stamp"] is not None and abs(r["stamp"] - stamp) < 1e-6), None)


# ------------------------------------------------------------- zones and comparisons

def esito_zone(n_ok, n_err, hash_applicato, hash_atteso, frames):
    """Zone outcome of a rollout, classified as in the orchestrator's
    zone_controllo: clouds without valid zones or with a different hash are an
    error; no cloud with zones while clouds were sent is an error."""
    nessuna = n_ok + n_err == 0
    return ("nessuna_nuvola" if nessuna and frames == 0 else
            "nessuna_zona_applicata" if nessuna else
            "errore" if (n_err > 0 or hash_applicato != hash_atteso) else "ok")


def asse_baseline(poly):
    """Origin and unit direction of the baseline (first and last pose)."""
    poly = np.asarray(poly, dtype=np.float64)
    d = poly[-1] - poly[0]
    return poly[0], d / np.linalg.norm(d)


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def grandezze_traccia(rows, p0, u):
    """True travelled distance and lateral coordinate (left of the baseline
    positive) per trace row; s_fine = distance at the end of the trace (R1)."""
    s = [_num(r.get("travelled")) for r in rows or []]
    lat = [u[0] * (_num(r.get("true_y")) - p0[1]) - u[1] * (_num(r.get("true_x")) - p0[0]) for r in rows or []]
    return {"s_fine_m": s[-1] if s else math.nan, "travelled": s, "lat": [float(v) for v in lat]}


def _su_griglia(s, y, griglia):
    # last value per distance (distance non-decreasing), then linear interpolation
    ss, yy = [], []
    for a, b in zip(s, y):
        if math.isnan(a) or math.isnan(b):
            continue
        if ss and a <= ss[-1]:
            yy[-1] = b
            continue
        ss.append(a)
        yy.append(b)
    if len(ss) < 2:
        return np.full(len(griglia), np.nan)
    return np.interp(griglia, ss, yy)


def _modulo_analisi():
    """Load scripts/analisi_accettazione_esito30ott.py as a fresh module (exec of
    the file: no .pyc, no sys.path); errore_diff is replaced per call."""
    m = types.ModuleType("analisi_accettazione_esito30ott")
    m.__file__ = str(ANALISI_ACCETTAZIONE)
    exec(compile(ANALISI_ACCETTAZIONE.read_text(), str(ANALISI_ACCETTAZIONE), "exec"), m.__dict__)
    return m


def e1_incrementale(traccia_v, log_v, traccia_w, log_w, frame_log, vittima_v=True, vittima_w=False):
    """Incremental E1 of criterion A with the reference method
    (analisi_accettazione.misura_metro with an errore_diff that records the
    vector; E1_inc = |e(stamp) - e(first aligned common stamp)| at the aligned
    E1 stamp, i.e. the last common stamp <= min(stamps held by the two sides at
    the stop command)). traccia_*: rollout CSV; log_*: MOLA log; frame_log:
    node per-frame log; vittima_*: victim-side flow (True) or twin (False).
    Returns the measures; E1_inc_cm is None with a reason if not measurable."""
    out = {"E1_inc_cm": None, "E1_ass_cm": None, "R2_inc_cm": None, "E1_stamp_vittima_s": None,
           "n_comuni_allineata": 0, "psi_deg": None, "motivo": None,
           "metodo": "misure_criterio_A.py (emendamento 8) su analisi_accettazione_esito30ott.py"}
    mancano = [n for n, p_ in (("traccia della vittima", traccia_v), ("log della vittima", log_v),
                               ("traccia del gemello", traccia_w), ("log del gemello", log_w),
                               ("registro per frame", frame_log)) if not p_ or not Path(p_).is_file()]
    if mancano:
        out["motivo"] = "mancano: " + ", ".join(mancano)
        return out
    try:
        aa = _modulo_analisi()
        reg = []

        def errore_diff_reg(v, w, kv, kw, psi):
            pv, pw = np.array(v["pose"][kv][:2]), np.array(w["pose"][kw][:2])
            tv_, tw_ = np.array(v["T"][kv][:2]), np.array(w["T"][kw][:2])
            c, s = math.cos(psi), math.sin(psi)
            dt = tv_ - tw_
            vec = (pv - pw) - np.array([c * dt[0] - s * dt[1], s * dt[0] + c * dt[1]])
            reg.append((kv, vec))
            return float(np.hypot(*vec))
        aa.errore_diff = errore_diff_reg
        fl = aa.leggi_frame_log(str(frame_log))
        v = aa.prepara_lato(aa.leggi_traccia(str(traccia_v)), aa.leggi_log_mola(str(log_v)), fl, bool(vittima_v))
        w = aa.prepara_lato(aa.leggi_traccia(str(traccia_w)), aa.leggi_log_mola(str(log_w)), fl, bool(vittima_w))
        m = aa.misura_metro(v, w)
    except Exception as e:      # incomplete traces or logs: no measure, not a run error
        out["motivo"] = f"{type(e).__name__}: {e}"
        return out
    n_all = m.get("n_comuni_allineata") or 0
    out.update(E1_ass_cm=m.get("E1_allineata_cm"), n_comuni_allineata=n_all, psi_deg=m.get("psi_deg"),
               E1_stamp_vittima_s=m.get("E1_allineata_stamp_vittima"),
               spostamento_orologio_s=m.get("spostamento_orologio_allineata_s"))
    if not (n_all and len(reg) >= n_all):
        out["motivo"] = "nessuna stamp comune allineata"
        return out
    ser = reg[-n_all:]
    v0 = ser[0][1]
    inc = [(kv, float(np.hypot(*(vec - v0)))) for kv, vec in ser]
    out["R2_inc_cm"] = 100 * max(e for _, e in inc)
    out["prima_stamp_comune_s"] = ser[0][0] / 10
    st = out["E1_stamp_vittima_s"]
    out["E1_inc_cm"] = next((100 * e for kv, e in inc if kv == aa.k10(st)), None) if st is not None else None
    if out["E1_inc_cm"] is None:
        out["motivo"] = "nessuna stamp comune entro le stamp in mano al comando di arresto"
    return out


def confronto_coppia(gv, gw, ramo_v, ramo_w, danno_v, danno_w, soglie=None, e1=_NON_DATO, dm_real=_NON_DATO):
    """Victim vs twin in the same repetition, with the criterion A definitions:
    R1 = true distance at the end of the trace (victim - twin); laterale = max
    lateral offset at equal true distance (1 cm grid up to the smaller end);
    danno = |victim mean_cm - twin mean_cm|; branch from esito_ramo. segue = same
    classified branch and all measures within threshold. Full criterion when e1
    (dict from e1_incrementale or None) and dm_real (victim damage measured) are
    given: adds incremental E1 (3.5 cm) and damage is not measurable with
    dm_real false. Without them only the three basic measures
    (criterio_completo false)."""
    completo = e1 is not _NON_DATO and dm_real is not _NON_DATO
    if soglie is None:
        soglie = SOGLIE_CRITERIO_A_COMPLETO if completo else SOGLIE_CRITERIO_A
    out = {"ramo_vittima": (ramo_v or {}).get("classe"), "ramo_gemello": (ramo_w or {}).get("classe"),
           "dy_vittima_cm": (ramo_v or {}).get("dy_cm"), "dy_gemello_cm": (ramo_w or {}).get("dy_cm")}
    out["stesso_ramo"] = (out["ramo_vittima"] == out["ramo_gemello"]
                          and out["ramo_vittima"] not in (None, "non_classificabile"))
    sv, sw = gv["s_fine_m"], gw["s_fine_m"]
    out["R1_cm"] = round(100.0 * (sv - sw), 3) if not (math.isnan(sv) or math.isnan(sw)) else None
    lat = None
    if out["R1_cm"] is not None and min(sv, sw) > 0.02:
        gr = np.arange(0.0, min(sv, sw) + 1e-9, 0.01)
        d = _su_griglia(gv["travelled"], gv["lat"], gr) - _su_griglia(gw["travelled"], gw["lat"], gr)
        if np.any(np.isfinite(d)):
            lat = round(100.0 * float(np.nanmax(np.abs(d))), 3)
    out["laterale_cm"] = lat
    dv, dw = (danno_v or {}).get("mean_cm"), (danno_w or {}).get("mean_cm")
    out["danno_cm"] = round(abs(dv - dw), 3) if dv is not None and dw is not None else None
    if completo:
        out["criterio_completo"] = True
        out["dm_real"] = bool(dm_real)
        if not dm_real:
            # victim damage not measured -> comparison not measurable
            out["danno_cm"] = None
        out["E1_inc_cm"] = round(e1["E1_inc_cm"], 3) if (e1 or {}).get("E1_inc_cm") is not None else None
        out["e1"] = {k: v for k, v in (e1 or {}).items()} if e1 is not None else None
    elif "E1_inc_cm" in soglie:
        out["E1_inc_cm"] = None
    out["soglie"] = dict(soglie)
    out["non_misurati"] = [k for k in soglie if out[k] is None]
    out["oltre"] = [k for k in soglie if out[k] is not None and abs(out[k]) > soglie[k]]
    out["segue"] = bool(out["stesso_ramo"] and not out["oltre"] and not out["non_misurati"])
    return out
