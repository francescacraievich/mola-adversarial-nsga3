#!/usr/bin/env bash
# Repetitions of one campaign arm on one scenario.
#
# For each repetition the script starts Isaac Sim from the terminal
# (isaac/run_isaac_standalone.py, extra options in ISAAC_ARGS, e.g. --headless)
# and the perturbation_node with --min-latency-ms $MIN_LATENCY_MS (default 85, 140 with
# SERIE_B=1; the gaussian arm also gets --gaussian-sigma), runs the orchestrator and at
# the end stops orchestrator, node and Isaac: each process lives for a single run and
# every arm pays the same per-scan latency. The add_intensity_node must already be
# running.
#
# The gaussian arm runs once per value of SIGMAS (m, default "0.01 0.02 0.05 0.10"),
# one subfolder per value: gaussian_s<sigma>.
# Folders: none_g<group>, gaussian_s<sigma>, <arm>_g<group>_p<pop>x<gen> for random
# and nsga3 (different configurations are never mixed).
#
# A repetition is complete when the orchestrator exits with code 0: the script then
# writes the file "completata" in its folder, and repetitions with that file are
# skipped (the script can be relaunched after an interruption). A folder without
# "completata" (interrupted run: history.json is written after every window, so it may
# exist partially) is renamed rep_NN_incompleta_<date> and the repetition is redone.
# If the orchestrator stops on an Isaac stall (code 3) the repetition is retried once;
# the failed attempt is kept as rep_NN_stallo1 (rep_NN_stallo2 if the retry fails as
# well, then the script moves on). A repetition with rep_NN_stallo2 is not retried by
# later launches.
# With the chain check (SERIE_B=1, attack arms) a repetition closed by a chain fault has
# da_rifare.json and no "completata": the script moves on and exits with code 4 at the
# end; the repetition is redone with the same seed on the next launch (or by
# scripts/coda_serieB.sh).
#
# Transport (environment variables only): TRANSPORT=reliable injects the qos block
# (reliable, depth 50) into the LiDAR entry of every MOLA instance through
# MOLA_LIDAR_TOPIC_TYPE, checked offline with scripts/controlla_qos_yaml.py before the
# launch; LOCALHOST_ONLY=1 exports ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST (the
# add_intensity_node must already have it). With the chain running, the negotiated QoS
# is checked with scripts/controlla_qos_grafo.sh.
# Local discovery: DISCOVERY_LOCALE=1 exports, for Isaac, node, orchestrator, MOLA
# (inherited) and probes, ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST without DDS_PROFILE
# and SYSTEM_DEFAULT with DDS_PROFILE (local discovery is then in the profile, e.g.
# config/dds/catena_udp4M_scoperta_locale.xml: with LOCALHOST rmw_fastrtps would add
# its own transports to the profile ones), and unsets ROS_LOCALHOST_ONLY (with 1, rcl
# forces LOCALHOST). The adapter must have the same value (checked below).
# No suffix in the folder name; value and reason go to ambiente.txt. 0 = off.
# LOCALHOST_ONLY=1 together with DDS_PROFILE is an error.
#
# DDS profile: DDS_PROFILE=<xml> exports FASTRTPS_DEFAULT_PROFILES_FILE (absolute path)
# for Isaac, perturbation_node and orchestrator and passes --dds-profile to the
# orchestrator (all MOLA instances); the adapter, launched outside this script, must
# already have the same variable. Empty = no profile. CONTROLLO_ETA=<s> passes
# --controllo-eta to the orchestrator (age of the cloud when MOLA takes it, per
# rollout, from the node's per-frame log); empty = off.
# MOLA_BINARIO_G1=1 passes --mola-binario-g1: every MOLA instance runs the binary of
# the install_G1 overlay, with the same variables as the shadow (environment only).
# SONDA=1 starts scripts/sonda_udp.py and scripts/sonda_nuvole.py before the
# orchestrator and stops them afterwards; output in <repetition>/sonda (with the logs).
#
# Usage:
#   scripts/run_campaign.sh <scenario> <arm> <group> <repetitions> [pop] [gen]
#
#   scenario: straight | L | rect   (waypoints defined below)
#   arm:      none | gaussian | random | nsga3
#   group:    1 | 2 | 3
#
# Examples:
#   scripts/run_campaign.sh straight nsga3 1 10 4 2
#   SIGMAS="0.02 0.05" scripts/run_campaign.sh straight gaussian 1 3
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$HERE/.."
cd "$ROOT"

SCENARIO="${1:?scenario}"
ARM="${2:?arm}"
GROUP="${3:?gruppo}"
REPS="${4:?ripetizioni}"
POP="${5:-4}"
GEN="${6:-2}"
START_POSE="${START_POSE:--5.99,-1.0,0}"   # absolute start pose x,y,yaw_deg
MIN_LATENCY_MS="${MIN_LATENCY_MS:-}"       # per-scan minimum latency of the node (ms; default 85, 140 with SERIE_B=1)
SIGMAS="${SIGMAS:-0.01 0.02 0.05 0.10}"    # standard deviations of the gaussian arm (m)
CHAMFER_BUDGET="${CHAMFER_BUDGET:-}"       # perceptibility budget (cm) for random/nsga3; empty = none
RUN_SUFFIX="${RUN_SUFFIX:-}"               # suffix appended to the arm folder name
STOP_ON="${STOP_ON:-estimate}"             # rollout stop: estimate (MOLA estimate, default) or true (true distance)
WINDOW_STOP="${WINDOW_STOP:-}"             # window closing with STOP_ON=estimate: empty or legacy (default), or ramp
STOP_RAMP_K="${STOP_RAMP_K:-}"             # ramp gain (1/s); empty = orchestrator default (1.5)
STOP_RAMP_VMIN="${STOP_RAMP_VMIN:-}"       # ramp minimum speed before H (m/s); empty = orchestrator default (0.05)
VICTIM="${VICTIM:-continuous}"             # victim SLAM on the applied path: continuous (default) or restart
SELECT="${SELECT:-maxdamage}"              # selection rule: maxdamage, ratio or pareto_maxdamage
TOPK="${TOPK:-3}"                          # with maxdamage, candidates to re-evaluate (ignored by pareto_maxdamage)
WARM_START="${WARM_START:-true}"           # initial population half from the previous front (true) or random (false)
NOMINAL_REPS="${NOMINAL_REPS:-}"           # none arm: nominals per window from the same state (empty = 1); noise floor
MAX_WINDOWS="${MAX_WINDOWS:-}"             # cap on the orchestrator windows (empty = all); for characterisation runs
GOALS_OVERRIDE="${GOALS_OVERRIDE:-}"       # waypoints relative to the start pose, replacing the scenario ones
ORCH_EXTRA="${ORCH_EXTRA:-}"               # extra orchestrator options (e.g. --repeat-genome ...)
DDS_PROFILE="${DDS_PROFILE:-}"             # Fast DDS XML profile for the whole chain (see above); empty = none
CONTROLLO_ETA="${CONTROLLO_ETA:-}"         # threshold (s) of the cloud age check (--controllo-eta); empty = off
DDS_FLAG=""; ETA_FLAG=""                   # orchestrator options from DDS_PROFILE and CONTROLLO_ETA (below)
MOLA_BINARIO_G1="${MOLA_BINARIO_G1:-0}"    # 1 = every MOLA on the install_G1 binary (--mola-binario-g1); 0 = off
SONDA="${SONDA:-0}"                        # 1 = cloud and UDP socket probe during every repetition
REGISTRI_DAL_VIVO="${REGISTRI_DAL_VIVO:-0}"      # 1 = live logs in <rep>/registri_dal_vivo (poses, controller, Isaac commands)
REG_FLAG=""; [[ "$REGISTRI_DAL_VIVO" == "1" ]] && REG_FLAG="--registri-dal-vivo"
STRUMENTI_NODO="${STRUMENTI_NODO:-0}"      # 1 = node with --strumenti-ritardo <rep>/ritardo_nodo.csv; 0 = off
G1_FLAG=""; [[ "$MOLA_BINARIO_G1" == "1" ]] && G1_FLAG="--mola-binario-g1"
# Options off by default: with default values the orchestrator command line is
# unchanged (they are prepended to ORCH_EXTRA only when enabled).
PROVE_VINCITORE="${PROVE_VINCITORE:-0}"    # N identical trials of the winner (--prove-vincitore N); 0 = none
RIALLINEA="${RIALLINEA:-0}"                # 1 = --riallinea-dopo-assestamento (realigned twin)
REGISTRA_ASSETTO="${REGISTRA_ASSETTO:-0}"  # 1 = --registra-assetto (Isaac attitude command in the rollouts)
TWIN_INIT_ASSETTO="${TWIN_INIT_ASSETTO:-}" # conservato, attaccante_relativo or attaccante = --twin-init-assetto
CONFRONTO_VINCITORE="${CONFRONTO_VINCITORE:-0}"   # 1 = --confronto-vincitore (selecting twin and comparison (a))
RAMPA="${RAMPA:-}"                         # genome ramp R (--rampa-genoma R to node and orchestrator); empty or 1 = off
D31_FLAGS=""; RAMPA_NODO=""
[[ -z "$RAMPA" || "$RAMPA" =~ ^[1-9][0-9]*$ ]] || { echo "RAMPA: intero >= 1, vuoto = spenta (dato: $RAMPA)"; exit 1; }
if [[ -n "$RAMPA" && "$RAMPA" != "1" ]]; then
  D31_FLAGS="$D31_FLAGS --rampa-genoma $RAMPA"; RAMPA_NODO=" --rampa-genoma $RAMPA"
fi
# Old genome kept until the robot moves (--genoma-in-moto S to node and
# orchestrator): S in cm, empty or 0 = off; not together with RAMPA.
GENOMA_IN_MOTO="${GENOMA_IN_MOTO:-}"
[[ "$GENOMA_IN_MOTO" =~ ^0+(\.0*)?$ ]] && GENOMA_IN_MOTO=""
if [[ -n "$GENOMA_IN_MOTO" ]]; then
  [[ "$GENOMA_IN_MOTO" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "GENOMA_IN_MOTO: soglia in cm > 0, vuoto o 0 = spenta (dato: $GENOMA_IN_MOTO)"; exit 1; }
  [[ -n "$RAMPA_NODO" ]] && { echo "GENOMA_IN_MOTO e RAMPA insieme: una sola delle due (stesso canale del nodo)"; exit 1; }
  D31_FLAGS="$D31_FLAGS --genoma-in-moto $GENOMA_IN_MOTO"; RAMPA_NODO=" --genoma-in-moto $GENOMA_IN_MOTO"
fi
# Chamfer per stream and per phase P/N in the node and in the history
# (--chamfer-per-fase to node and orchestrator; already on with GENOMA_IN_MOTO).
CHAMFER_PER_FASE="${CHAMFER_PER_FASE:-0}"
case "$CHAMFER_PER_FASE" in 0) : ;; 1) D31_FLAGS="$D31_FLAGS --chamfer-per-fase"; RAMPA_NODO="$RAMPA_NODO --chamfer-per-fase" ;;
  *) echo "CHAMFER_PER_FASE: 0 o 1 (dato: $CHAMFER_PER_FASE)"; exit 1 ;; esac
# input and output clouds of the first K clouds of every genome, inside the repetition
SALVA_NUVOLE="${SALVA_NUVOLE:-0}"; SALVA_NUVOLE_K="${SALVA_NUVOLE_K:-20}"   # 1 = node with --salva-nuvole <rep>/nuvole --salva-nuvole-k K
case "$SALVA_NUVOLE" in 0|1) : ;; *) echo "SALVA_NUVOLE: 0 o 1 (dato: $SALVA_NUVOLE)"; exit 1 ;; esac
[[ "$SALVA_NUVOLE_K" =~ ^[1-9][0-9]*$ ]] || { echo "SALVA_NUVOLE_K: intero >= 1 (dato: $SALVA_NUVOLE_K)"; exit 1; }
[[ "$PROVE_VINCITORE" =~ ^[0-9]+$ ]] || { echo "PROVE_VINCITORE: intero >= 0 (dato: $PROVE_VINCITORE)"; exit 1; }
[[ "$PROVE_VINCITORE" != "0" ]] && D31_FLAGS="$D31_FLAGS --prove-vincitore $PROVE_VINCITORE"
case "$RIALLINEA" in 0) : ;; 1) D31_FLAGS="$D31_FLAGS --riallinea-dopo-assestamento" ;;
  *) echo "RIALLINEA: 0 o 1 (dato: $RIALLINEA)"; exit 1 ;; esac
case "$REGISTRA_ASSETTO" in 0) : ;; 1) D31_FLAGS="$D31_FLAGS --registra-assetto" ;;
  *) echo "REGISTRA_ASSETTO: 0 o 1 (dato: $REGISTRA_ASSETTO)"; exit 1 ;; esac
case "$TWIN_INIT_ASSETTO" in ""|azzerato) : ;; conservato|attaccante_relativo|attaccante) D31_FLAGS="$D31_FLAGS --twin-init-assetto $TWIN_INIT_ASSETTO" ;;
  *) echo "TWIN_INIT_ASSETTO: vuoto, azzerato, conservato, attaccante_relativo o attaccante (dato: $TWIN_INIT_ASSETTO)"; exit 1 ;; esac
case "$CONFRONTO_VINCITORE" in 0) : ;; 1) D31_FLAGS="$D31_FLAGS --confronto-vincitore" ;;
  *) echo "CONFRONTO_VINCITORE: 0 o 1 (dato: $CONFRONTO_VINCITORE)"; exit 1 ;; esac
[[ -n "$D31_FLAGS" ]] && ORCH_EXTRA="${D31_FLAGS# }${ORCH_EXTRA:+ $ORCH_EXTRA}"
SONDA_PID=""; SONDA_UDP_PID=""
ATTACK_FROM="${ATTACK_FROM:-spawn}"        # attack on from MOLA start (spawn) or from motion start (motion)
DRIFT_RESET_ON="${DRIFT_RESET_ON:-}"       # time drift reset: genome (default), drive or keep (never)
MAX_DRIFT_LAG_S="${MAX_DRIFT_LAG_S:-}"     # cap on the node time drift (s at 0.8 m/s); empty = none
ISAAC_ARGS="${ISAAC_ARGS:-}"               # extra options for run_isaac_standalone.py
ISAAC_READY_SEC="${ISAAC_READY_SEC:-300}"  # maximum wait for Isaac start (s)
CAMPAIGN_DIR="${CAMPAIGN_DIR:-data/attack/campaign}"   # results root
# SERIE_B=1 sets the defaults below (each one can still be overridden).
SERIE_B="${SERIE_B:-}"
if [[ "$SERIE_B" == "1" ]]; then
  ICP_QUALITY_LOG="${ICP_QUALITY_LOG:-1}"
  WINDOW_RNG="${WINDOW_RNG:-seedseq}"
  SEEDS="${SEEDS:-1001 2001 3001}"
  PERC_TERMS="${PERC_TERMS:-1}"
  WARM_FRONT_AFTER_REEVAL="${WARM_FRONT_AFTER_REEVAL:-1}"
  # --fast-ops, synchronous start on the publisher with guard; D_min 140 ms
  MIN_LATENCY_MS="${MIN_LATENCY_MS:-140}"
  FAST_OPS="${FAST_OPS:-1}"
  REPORT_MATCHED="${REPORT_MATCHED:-1}"
  SYNC_WAIT="${SYNC_WAIT:-publisher}"
  SYNC_GUARD="${SYNC_GUARD:-1}"
  # per-metre and end-of-run damage quantities
  DANNO_METRO="${DANNO_METRO:-1}"
  # time drift (gene 14) kept across metres
  case "$ARM" in random|nsga3|fixed) DRIFT_RESET_ON="${DRIFT_RESET_ON:-keep}" ;; esac
  # MOLA DEBUG in every run: same configuration for victim, shadow, attacker and
  # twins, also in the none arm
  MOLA_DEBUG="${MOLA_DEBUG:-1}"
fi
DRIFT_RESET_ON="${DRIFT_RESET_ON:-genome}"
MIN_LATENCY_MS="${MIN_LATENCY_MS:-85}"
MOLA_DEBUG="${MOLA_DEBUG:-}"               # 1 = MOLA DEBUG and MOLA_LO_DEBUG_ICP_QUALITY=1 for all MOLA (via environment)
DANNO_METRO="${DANNO_METRO:-}"             # 1 = --danno-metro to the orchestrator (logging only)
FAST_OPS="${FAST_OPS:-}"                   # 1 = --fast-ops to the node (bit-identical output)
REPORT_MATCHED="${REPORT_MATCHED:-}"       # 1 = --report-matched to the node (matched readers in the status)
SYNC_WAIT="${SYNC_WAIT:-}"                 # orchestrator --sync-wait: empty (orchestrator default, graph), graph or publisher
SYNC_GUARD="${SYNC_GUARD:-}"               # 1 = --sync-guard to the orchestrator
ICP_QUALITY_LOG="${ICP_QUALITY_LOG:-}"     # 1 = MOLA_LO_DEBUG_ICP_QUALITY=1 for all MOLA (one line per scan)
WINDOW_RNG="${WINDOW_RNG:-}"               # random arm: per-window generator (empty = orchestrator default, sum; seedseq)
SEEDS="${SEEDS:-}"                         # seeds, one per repetition (e.g. "1001 2001 3001"); empty = 1000 + repetition
PERC_TERMS="${PERC_TERMS:-}"               # 1 = node also publishes pure Chamfer, structural term and r (--perc-terms)
STRUCTURAL="${STRUCTURAL:-}"               # structural term of P: empty or net (node default), gross (removed + added)
SAMPLING="${SAMPLING:-}"                   # random/nsga3: distribution of random genomes (empty or legacy, uniform, sparse)
RANDOM_FIRST="${RANDOM_FIRST:-}"           # random arm: single random with budget, at most K tries (empty = pop x gen)
TWIN_FILTER="${TWIN_FILTER:-}"             # exclusion of anomalous twins from selection (empty or none, am2, al, am2_al)
OPERATOR_MODE="${OPERATOR_MODE:-}"         # node operators: empty or random, insieme
ZONE_ATTACCANTE="${ZONE_ATTACCANTE:-}"     # 1 = attacker-chosen zones (--zone-attaccante, requires insieme)
TRANSPORT="${TRANSPORT:-default}"          # QoS of the MOLA LiDAR entry: default (bridge best effort) or reliable
LOCALHOST_ONLY="${LOCALHOST_ONLY:-}"       # 1 = ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST for processes launched here
DISCOVERY_LOCALE="${DISCOVERY_LOCALE:-0}"  # 1 = ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST for the whole chain
WARM_FRONT_AFTER_REEVAL="${WARM_FRONT_AFTER_REEVAL:-}"   # nsga3: 1 = warm-start front after re-evaluations

case "$SCENARIO" in
  straight) GOALS="5,0" ;;
  L)        GOALS="5,0; 5,2.5; 10,2.5" ;;
  rect)     GOALS="4,0; 4,2.5; 0,2.5; 0,0" ;;
esac
[[ -n "$GOALS_OVERRIDE" ]] && GOALS="$GOALS_OVERRIDE"
case "$SCENARIO" in
  straight|L|rect) : ;;
  *) echo "scenario sconosciuto: $SCENARIO"; exit 1 ;;
esac

case "$ARM" in
  none)     EXTRA="--passthrough${NOMINAL_REPS:+ --nominal-reps $NOMINAL_REPS}${MAX_WINDOWS:+ --max-windows $MAX_WINDOWS}" ;;
  gaussian) EXTRA="--search gaussian" ;;
  random)   EXTRA="--search random --pop $POP --gen $GEN --genome-group $GROUP" ;;
  nsga3)    EXTRA="--search nsga3 --pop $POP --gen $GEN --genome-group $GROUP" ;;
  fixed)    EXTRA="--search fixed --genome-group $GROUP" ;;
esac
BUDGET_SUFFIX=""
if [[ -n "$CHAMFER_BUDGET" && ( "$ARM" == "random" || "$ARM" == "nsga3" ) ]]; then
  EXTRA="$EXTRA --chamfer-budget $CHAMFER_BUDGET"
  BUDGET_SUFFIX="_eps${CHAMFER_BUDGET}"
fi
# The drift cap is a node parameter: it goes into the folder name (lag0.3)
# because it distinguishes the arms of the sweep.
if [[ -n "$MAX_DRIFT_LAG_S" ]]; then
  BUDGET_SUFFIX="${BUDGET_SUFFIX}_lag${MAX_DRIFT_LAG_S}"
fi
# Non-default modes go into the folder name, for every arm.
MODE_SUFFIX=""
[[ "$ATTACK_FROM" == "motion" ]] && MODE_SUFFIX="${MODE_SUFFIX}_motion"
[[ "$DRIFT_RESET_ON" == "drive" ]] && MODE_SUFFIX="${MODE_SUFFIX}_rdrive"
[[ "$DRIFT_RESET_ON" == "keep" ]] && MODE_SUFFIX="${MODE_SUFFIX}_rkeep"
[[ "$SELECT" == "maxdamage" ]] && MODE_SUFFIX="${MODE_SUFFIX}_sel2"
[[ "$SELECT" == "pareto_maxdamage" ]] && MODE_SUFFIX="${MODE_SUFFIX}_sel3"
[[ "$WARM_START" == "true" ]] && MODE_SUFFIX="${MODE_SUFFIX}_warm"
if [[ "$WARM_START" == "true" ]]; then WARM_FLAG="--warm-start"; else WARM_FLAG="--no-warm-start"; fi
[[ "$VICTIM" == "restart" ]] && MODE_SUFFIX="${MODE_SUFFIX}_vrestart"
[[ "$STOP_ON" == "true" ]] && MODE_SUFFIX="${MODE_SUFFIX}_stoptrue"
# End-of-window ramp: in the name, with k and vmin only if they differ from the defaults.
WINDOW_STOP_FLAGS=""
case "$WINDOW_STOP" in
  ""|legacy) : ;;
  ramp)
    MODE_SUFFIX="${MODE_SUFFIX}_stopramp${STOP_RAMP_K:+_k${STOP_RAMP_K}}${STOP_RAMP_VMIN:+_vmin${STOP_RAMP_VMIN}}"
    WINDOW_STOP_FLAGS="--window-stop ramp${STOP_RAMP_K:+ --stop-ramp-k $STOP_RAMP_K}${STOP_RAMP_VMIN:+ --stop-ramp-vmin $STOP_RAMP_VMIN}" ;;
  *) echo "WINDOW_STOP sconosciuto: $WINDOW_STOP (vuoto, legacy o ramp)"; exit 1 ;;
esac
[[ -n "$NOMINAL_REPS" ]] && MODE_SUFFIX="${MODE_SUFFIX}_nomreps${NOMINAL_REPS}"
# SERIE_B options: in the folder name only if they differ from the default.
SERIEB_FLAGS=""
case "$WINDOW_RNG" in
  ""|sum) : ;;
  seedseq) [[ "$ARM" == "random" ]] && { SERIEB_FLAGS="$SERIEB_FLAGS --window-rng seedseq"; MODE_SUFFIX="${MODE_SUFFIX}_seedseq"; } ;;
  *) echo "WINDOW_RNG sconosciuto: $WINDOW_RNG (vuoto, sum o seedseq)"; exit 1 ;;
esac
case "$SAMPLING" in
  ""|legacy) : ;;
  uniform|sparse) [[ "$ARM" == "random" || "$ARM" == "nsga3" ]] && { SERIEB_FLAGS="$SERIEB_FLAGS --sampling $SAMPLING"; MODE_SUFFIX="${MODE_SUFFIX}_samp${SAMPLING}"; } ;;
  *) echo "SAMPLING sconosciuto: $SAMPLING (vuoto, legacy, uniform o sparse)"; exit 1 ;;
esac
if [[ -n "$RANDOM_FIRST" ]]; then
  [[ "$ARM" == "random" && "$RANDOM_FIRST" =~ ^[1-9][0-9]*$ ]] || { echo "RANDOM_FIRST vale solo per il braccio random, intero positivo: $RANDOM_FIRST"; exit 1; }
  # The "first valid applied" rule matches the selection only with
  # pareto_maxdamage: the orchestrator rejects the other selections.
  [[ "$SELECT" == "pareto_maxdamage" ]] || { echo "RANDOM_FIRST richiede SELECT=pareto_maxdamage (dato: $SELECT)"; exit 1; }
  SERIEB_FLAGS="$SERIEB_FLAGS --random-first-admissible $RANDOM_FIRST"
  MODE_SUFFIX="${MODE_SUFFIX}_first${RANDOM_FIRST}"
fi
case "$TWIN_FILTER" in
  ""|none) : ;;
  am2|al|am2_al) SERIEB_FLAGS="$SERIEB_FLAGS --twin-filter $TWIN_FILTER"; MODE_SUFFIX="${MODE_SUFFIX}_tf${TWIN_FILTER}" ;;
  *) echo "TWIN_FILTER sconosciuto: $TWIN_FILTER (vuoto, none, am2, al o am2_al)"; exit 1 ;;
esac
# the G bench (--g-banco) runs with --victim restart and no nominals: no
# no-nominal-rollout option there
[[ "$SERIE_B" == "1" && "$ARM" != "none" && " ${ORCH_EXTRA:-} " != *" --g-banco "* ]] && SERIEB_FLAGS="$SERIEB_FLAGS --no-nominal-rollout"   # no nominal of the twin: damage from the baseline only
[[ "$SERIE_B" == "1" && "$ARM" != "none" ]] && SERIEB_FLAGS="$SERIEB_FLAGS --chiudi-attacco-scoperto"   # victim without pose = attack detected, run closed
# before declaring an attack detected the chain is checked, also with the adapter
# events (unless ORCH_EXTRA already names their file)
if [[ "$SERIE_B" == "1" && "$ARM" != "none" && " $ORCH_EXTRA " != *" --adapter-events "* ]]; then
  SERIEB_FLAGS="$SERIEB_FLAGS --adapter-events ${ADAPTER_EVENTS:-data/attack/verifica/adattatore_eventi.jsonl}"
fi
[[ "$SERIE_B" == "1" && "$ARM" != "none" ]] && SERIEB_FLAGS="$SERIEB_FLAGS --mola-se-ne-accorge"   # twin MOLA crashes or has no pose > 1 s = candidate not admissible
[[ "$SERIE_B" == "1" && "$ARM" != "none" ]] && SERIEB_FLAGS="$SERIEB_FLAGS --controllo-catena"   # chain fault checked before declaring the attack detected
[[ "$SERIE_B" == "1" && "$ARM" != "none" ]] && SERIEB_FLAGS="$SERIEB_FLAGS --incompleti-regola"   # timeout and stopped_early classified by cause
# --rallentamento-mola (guasto, scoperto or risultato): the victim's MOLA time is a
# result of the attack, not a chain fault. Empty = risultato with SERIE_B=1 in the
# attack arms, otherwise the orchestrator default (guasto)
RALLENTAMENTO_MOLA="${RALLENTAMENTO_MOLA:-}"
case "$RALLENTAMENTO_MOLA" in
  ""|guasto|scoperto|risultato)
    if [[ "$SERIE_B" == "1" && "$ARM" != "none" ]]; then
      SERIEB_FLAGS="$SERIEB_FLAGS --rallentamento-mola ${RALLENTAMENTO_MOLA:-risultato}"
    fi ;;
  *) echo "RALLENTAMENTO_MOLA sconosciuto: $RALLENTAMENTO_MOLA (vuoto, guasto, scoperto o risultato)"; exit 1 ;;
esac
FRAME_LOG_ORCH=""   # per-repetition --frame-log-nodo (below)
DA_RIFARE=()        # repetitions closed by a chain fault (da_rifare.json)
if [[ -n "$SEEDS" ]]; then
  read -r -a SEED_LIST <<< "$SEEDS"
  (( ${#SEED_LIST[@]} >= REPS )) || { echo "SEEDS ha ${#SEED_LIST[@]} semi per $REPS ripetizioni"; exit 1; }
  for s in "${SEED_LIST[@]}"; do
    [[ "$s" =~ ^[0-9]+$ ]] || { echo "SEEDS: '$s' non e' un intero non negativo"; exit 1; }
  done
  [[ $(printf '%s\n' "${SEED_LIST[@]}" | sort | uniq -d | wc -l) -eq 0 ]] || { echo "SEEDS con semi duplicati: $SEEDS"; exit 1; }
fi
# MOLA DEBUG: exported here, inherited by every instance launched by the
# orchestrator (MolaRunner.spawn copies the environment)
case "$MOLA_DEBUG" in
  ""|0) : ;;
  1)
    ICP_QUALITY_LOG=1
    [[ -n "${MOLA_VERBOSITY_MOLA_LO:-}" && "$MOLA_VERBOSITY_MOLA_LO" != "DEBUG" ]] && \
      echo "MOLA_DEBUG=1: MOLA_VERBOSITY_MOLA_LO era '$MOLA_VERBOSITY_MOLA_LO', portata a DEBUG"
    export MOLA_VERBOSITY_MOLA_LO=DEBUG ;;
  *) echo "MOLA_DEBUG sconosciuto: $MOLA_DEBUG (vuoto, 0 o 1)"; exit 1 ;;
esac
case "$ICP_QUALITY_LOG" in
  "") : ;;
  1)
    [[ -n "${MOLA_LO_DEBUG_ICP_QUALITY:-}" && "$MOLA_LO_DEBUG_ICP_QUALITY" != "1" ]] && \
      echo "ICP_QUALITY_LOG=1: MOLA_LO_DEBUG_ICP_QUALITY era '$MOLA_LO_DEBUG_ICP_QUALITY', portata a 1"
    export MOLA_LO_DEBUG_ICP_QUALITY=1 ;;
  *) echo "ICP_QUALITY_LOG sconosciuto: $ICP_QUALITY_LOG (vuoto o 1)"; exit 1 ;;
esac
# A value other than 0 and 1 would end up in the name (_pt<value>) without
# --perc-terms to the node.
case "$PERC_TERMS" in
  ""|0|1) : ;;
  *) echo "PERC_TERMS sconosciuto: $PERC_TERMS (vuoto, 0 o 1)"; exit 1 ;;
esac
# Warm-start front after re-evaluations: NSGA-III only. In the name only if it
# differs from the default (_wfr without SERIE_B, _wfr0 with it).
case "$WARM_FRONT_AFTER_REEVAL" in
  ""|0) [[ "$SERIE_B" == "1" && "$ARM" == "nsga3" ]] && MODE_SUFFIX="${MODE_SUFFIX}_wfr0" ;;
  1)
    if [[ "$ARM" == "nsga3" ]]; then
      SERIEB_FLAGS="$SERIEB_FLAGS --warm-front-after-reeval"
      [[ "$SERIE_B" != "1" ]] && MODE_SUFFIX="${MODE_SUFFIX}_wfr"
    fi ;;
  *) echo "WARM_FRONT_AFTER_REEVAL sconosciuto: $WARM_FRONT_AFTER_REEVAL (vuoto, 0 o 1)"; exit 1 ;;
esac
case "$STRUCTURAL" in
  ""|net) [[ -n "$STRUCTURAL" ]] && SERIEB_FLAGS="$SERIEB_FLAGS --structural $STRUCTURAL" ;;
  gross) SERIEB_FLAGS="$SERIEB_FLAGS --structural gross" ;;
  *) echo "STRUCTURAL sconosciuto: $STRUCTURAL (vuoto, net o gross)"; exit 1 ;;
esac
# "insieme" operator mode: on with SERIE_B=1 in the arms that use the generator;
# no effect in the none and gaussian arms.
if [[ "$SERIE_B" == "1" && -z "$OPERATOR_MODE" ]]; then
  case "$ARM" in random|nsga3|fixed) OPERATOR_MODE=insieme ;; esac
fi
OPERATOR_NODE_FLAGS=""
case "$OPERATOR_MODE" in
  ""|random) : ;;
  insieme)
    [[ "$ARM" == "none" || "$ARM" == "gaussian" ]] \
      && { echo "OPERATOR_MODE=insieme non ha effetto nel braccio $ARM: lasciarlo vuoto"; exit 1; }
    OPERATOR_NODE_FLAGS=" --operator-mode insieme"
    SERIEB_FLAGS="$SERIEB_FLAGS --metro-al-nodo" ;;
  *) echo "OPERATOR_MODE sconosciuto: $OPERATOR_MODE (vuoto, random o insieme)"; exit 1 ;;
esac
# Attacker-chosen zones: per-metre snapshot, zones per genome with a hash; the node
# applies them and does not choose. On by default with SERIE_B=1 in the generator arms.
if [[ "$SERIE_B" == "1" && -z "$ZONE_ATTACCANTE" ]]; then
  case "$ARM" in random|nsga3|fixed) ZONE_ATTACCANTE=1 ;; esac
fi
# SERIE_B=1, attack arms: only the "insieme" mode with attacker zones; with other
# operators or zones off the run does not start.
if [[ "$SERIE_B" == "1" ]]; then
  case "$ARM" in
    random|nsga3|fixed)
      [[ "$OPERATOR_MODE" == "insieme" ]] \
        || { echo "SERIE_B=1, braccio $ARM: OPERATOR_MODE deve essere insieme (dato: ${OPERATOR_MODE:-vuoto}): run fermata"; exit 1; }
      [[ "$ZONE_ATTACCANTE" == "1" ]] \
        || { echo "SERIE_B=1, braccio $ARM: zone dell'attaccante spente (ZONE_ATTACCANTE=${ZONE_ATTACCANTE:-vuoto}): run fermata"; exit 1; } ;;
  esac
fi
case "$ZONE_ATTACCANTE" in
  ""|0) : ;;
  1)
    [[ "$OPERATOR_MODE" == "insieme" ]] \
      || { echo "ZONE_ATTACCANTE=1 richiede OPERATOR_MODE=insieme (dato: ${OPERATOR_MODE:-vuoto})"; exit 1; }
    OPERATOR_NODE_FLAGS="$OPERATOR_NODE_FLAGS --zone-attaccante"
    SERIEB_FLAGS="$SERIEB_FLAGS --zone-attaccante" ;;
  *) echo "ZONE_ATTACCANTE sconosciuto: $ZONE_ATTACCANTE (vuoto, 0 o 1)"; exit 1 ;;
esac
case "$OPERATOR_MODE:$ZONE_ATTACCANTE" in
  insieme:1) SORGENTE_ZONE="attaccante (fotografia per metro, zone per genoma con hash)" ;;
  insieme:*) SORGENTE_ZONE="prima nuvola di ogni flusso (versione I senza zone dell'attaccante)" ;;
  *)         SORGENTE_ZONE="nessuna (operatori random, serie A)" ;;
esac
# Node options (NODE_SB_FLAGS) and synchronous start: not in the folder name
# (fast_ops does not change the output; the synchronous start is in the shadow log).
NODE_SB_FLAGS=""
case "$FAST_OPS" in
  ""|0) : ;;
  1) NODE_SB_FLAGS="$NODE_SB_FLAGS --fast-ops" ;;
  *) echo "FAST_OPS sconosciuto: $FAST_OPS (vuoto, 0 o 1)"; exit 1 ;;
esac
case "$REPORT_MATCHED" in
  ""|0) : ;;
  1) NODE_SB_FLAGS="$NODE_SB_FLAGS --report-matched" ;;
  *) echo "REPORT_MATCHED sconosciuto: $REPORT_MATCHED (vuoto, 0 o 1)"; exit 1 ;;
esac
case "$SYNC_WAIT" in
  "") : ;;
  graph) SERIEB_FLAGS="$SERIEB_FLAGS --sync-wait graph" ;;
  publisher)
    # publisher reads victim_pub_matched from the status: needs --report-matched on the node
    [[ "$REPORT_MATCHED" == "1" || " ${NODE_EXTRA:-} " == *" --report-matched "* ]] \
      || { echo "SYNC_WAIT=publisher richiede REPORT_MATCHED=1 (o --report-matched in NODE_EXTRA)"; exit 1; }
    SERIEB_FLAGS="$SERIEB_FLAGS --sync-wait publisher" ;;
  *) echo "SYNC_WAIT sconosciuto: $SYNC_WAIT (vuoto, graph o publisher)"; exit 1 ;;
esac
case "$SYNC_GUARD" in
  ""|0) : ;;
  1) SERIEB_FLAGS="$SERIEB_FLAGS --sync-guard" ;;
  *) echo "SYNC_GUARD sconosciuto: $SYNC_GUARD (vuoto, 0 o 1)"; exit 1 ;;
esac
# Logging only (fitness and selection unchanged): not in the folder name.
case "$DANNO_METRO" in
  ""|0) : ;;
  1) if [[ "$ARM" == "random" || "$ARM" == "nsga3" || "$ARM" == "fixed" ]]; then
       SERIEB_FLAGS="$SERIEB_FLAGS --danno-metro"
     fi ;;
  *) echo "DANNO_METRO sconosciuto: $DANNO_METRO (vuoto, 0 o 1)"; exit 1 ;;
esac
# SERIE_B=1: only a valid candidate can win, and admissibility is decided by the
# Chamfer distance (budget from CHAMFER_BUDGET).
if [[ "$SERIE_B" == "1" ]]; then
  [[ "$SELECT" == "ratio" ]] && { echo "SERIE_B=1 richiede SELECT=maxdamage o pareto_maxdamage (dato: ratio)"; exit 1; }
  SERIEB_FLAGS="$SERIEB_FLAGS --select-only-valid --perceptibility chamfer"
fi
# Transport (environment variables only, MOLA YAML unchanged). reliable:
# MOLA_LIDAR_TOPIC_TYPE carries, after the type, the qos block of the LiDAR entry;
# MOLA's substitution is textual and the entry keys sit at 10 spaces in the
# re-emitted text (with 8 the parser truncates the document without error).
QOS_LIDAR_TYPE=$'PointCloud2\n          qos:\n            reliability: reliable\n            depth: 50'
case "$TRANSPORT" in
  default) : ;;
  reliable)
    export MOLA_LIDAR_TOPIC_TYPE="$QOS_LIDAR_TYPE"
    MODE_SUFFIX="${MODE_SUFFIX}_reliable" ;;
  *) echo "TRANSPORT sconosciuto: $TRANSPORT (default o reliable)"; exit 1 ;;
esac
export SERIEB_TRANSPORT="$TRANSPORT"
case "$LOCALHOST_ONLY" in
  "") : ;;
  1) [[ -z "$DDS_PROFILE" ]] || { echo "LOCALHOST_ONLY=1 incompatibile con DDS_PROFILE (rmw raddoppierebbe i trasporti del profilo): usare DISCOVERY_LOCALE=1"; exit 1; }
     export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST; MODE_SUFFIX="${MODE_SUFFIX}_localhost" ;;
  *) echo "LOCALHOST_ONLY sconosciuto: $LOCALHOST_ONLY (vuoto o 1)"; exit 1 ;;
esac
SCOPERTA_MOTIVO=""
case "$DISCOVERY_LOCALE" in
  0) : ;;
  1) if [[ -n "$DDS_PROFILE" ]]; then
       grep -q "<initialPeersList>" "$DDS_PROFILE" 2>/dev/null && grep -q "127\.0\.0\.1" "$DDS_PROFILE" \
         || { echo "DISCOVERY_LOCALE=1 con DDS_PROFILE=$DDS_PROFILE senza scoperta locale nel profilo (initialPeersList 127.0.0.1): run fermata"; exit 1; }
       export ROS_AUTOMATIC_DISCOVERY_RANGE=SYSTEM_DEFAULT
       SCOPERTA_MOTIVO="profilo DDS con la scoperta locale dentro: SYSTEM_DEFAULT perche' rmw non tocchi il profilo"
     else
       export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
       SCOPERTA_MOTIVO="senza profilo DDS: scoperta locale da rmw (LOCALHOST)"
     fi
     unset ROS_LOCALHOST_ONLY ;;
  *) echo "DISCOVERY_LOCALE sconosciuto: $DISCOVERY_LOCALE (0 o 1)"; exit 1 ;;
esac
# discovery range in a process environment (empty = not set)
range_di() { tr '\0' '\n' < "/proc/$1/environ" 2>/dev/null | sed -n 's/^ROS_AUTOMATIC_DISCOVERY_RANGE=//p'; }
# Configuration in the folder name: always with SERIE_B=1 (seeds, Chamfer terms,
# operator mode), otherwise only when set.
SEEDS_TAG=""
[[ -n "$SEEDS" ]] && SEEDS_TAG="$(IFS=-; echo "${SEED_LIST[*]}")"
if [[ "$SERIE_B" == "1" ]]; then
  MODE_SUFFIX="${MODE_SUFFIX}_serieB_s${SEEDS_TAG}_pt${PERC_TERMS:-0}_op${OPERATOR_MODE:-random}"
else
  [[ -n "$SEEDS" ]] && MODE_SUFFIX="${MODE_SUFFIX}_s${SEEDS_TAG}"
  [[ "$PERC_TERMS" == "1" ]] && MODE_SUFFIX="${MODE_SUFFIX}_pt1"
fi
[[ "$STRUCTURAL" == "gross" ]] && MODE_SUFFIX="${MODE_SUFFIX}_sgross"
[[ -n "$GENOMA_IN_MOTO" ]] && MODE_SUFFIX="${MODE_SUFFIX}_gim${GENOMA_IN_MOTO}"
case "$ARM" in
  none|gaussian|random|nsga3|fixed) : ;;
  *) echo "braccio sconosciuto: $ARM"; exit 1 ;;
esac

if pgrep -f "^python3( -u)? .*run_isaac_standalone\.py" >/dev/null || pgrep -x isaacsim >/dev/null; then
  echo "Isaac Sim e' gia' in esecuzione: lo script avvia e chiude Isaac a ogni ripetizione."
  echo "Chiuderlo prima di lanciare la campagna."
  exit 1
fi
if pgrep -f "^python3( -u)? .*perturbation_node\.py" >/dev/null; then
  echo "perturbation_node gia' in esecuzione: lo script lo avvia e chiude a ogni ripetizione"
  echo "(due nodi pubblicherebbero sullo stesso topic). Chiuderlo prima di lanciare la campagna."
  exit 1
fi
if ! pgrep -f "^python3( -u)? .*add_intensity_node\.py" >/dev/null; then
  echo "add_intensity_node non in esecuzione: e' un prerequisito della campagna."
  exit 1
fi
# SERIE_B=1: versions of the interpreter that launches the node (python3 from PATH,
# as in node_start) must match the reference, otherwise the run does not start.
AMBIENTE_RIFERIMENTO="${AMBIENTE_RIFERIMENTO:-config/ambiente_nodo_serieB.txt}"
if [[ "$SERIE_B" == "1" ]]; then
  python3 scripts/ambiente_nodo.py controlla --riferimento "$AMBIENTE_RIFERIMENTO" \
    || { echo "SERIE_B=1: ambiente del perturbation_node diverso dal riferimento: run fermata"; exit 1; }
fi
# With LOCALHOST_ONLY=1 or DISCOVERY_LOCALE=1 the adapter, launched outside this
# script, must have the same discovery range (read from its environment) and, with
# DISCOVERY_LOCALE=1, no ROS_LOCALHOST_ONLY=1.
if [[ "$LOCALHOST_ONLY" == "1" || "$DISCOVERY_LOCALE" == "1" ]]; then
  for pid in $(pgrep -f "^python3( -u)? .*add_intensity_node\.py"); do
    if ! tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -qx "ROS_AUTOMATIC_DISCOVERY_RANGE=$ROS_AUTOMATIC_DISCOVERY_RANGE"; then
      echo "LOCALHOST_ONLY=1 o DISCOVERY_LOCALE=1 ma add_intensity_node (pid $pid) non ha ROS_AUTOMATIC_DISCOVERY_RANGE=$ROS_AUTOMATIC_DISCOVERY_RANGE: rilanciarlo con la variabile."
      exit 1
    fi
    if [[ "$DISCOVERY_LOCALE" == "1" ]] && tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -qx "ROS_LOCALHOST_ONLY=1"; then
      echo "DISCOVERY_LOCALE=1 ma add_intensity_node (pid $pid) ha ROS_LOCALHOST_ONLY=1 (forza LOCALHOST): rilanciarlo senza."
      exit 1
    fi
  done
fi
# DDS profile: same file for every process of the chain. Error if it is missing or
# if DEFAULT_FASTRTPS_PROFILES.xml is in the working directory (here or the
# adapter's): Fast DDS would load the same profiles twice.
DDS_FLAG=""
if [[ -n "$DDS_PROFILE" ]]; then
  [[ -f "$DDS_PROFILE" ]] || { echo "DDS_PROFILE=$DDS_PROFILE: file inesistente: run fermata"; exit 1; }
  DDS_PROFILE="$(realpath "$DDS_PROFILE")"
  [[ -e "$PWD/DEFAULT_FASTRTPS_PROFILES.xml" ]] \
    && { echo "DDS_PROFILE: $PWD/DEFAULT_FASTRTPS_PROFILES.xml presente (profili caricati due volte): run fermata"; exit 1; }
  for pid in $(pgrep -f "^python3( -u)? .*add_intensity_node\.py"); do
    tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -qxF "FASTRTPS_DEFAULT_PROFILES_FILE=$DDS_PROFILE" \
      || { echo "DDS_PROFILE: add_intensity_node (pid $pid) senza FASTRTPS_DEFAULT_PROFILES_FILE=$DDS_PROFILE: rilanciarlo con la variabile."; exit 1; }
    [[ -e "/proc/$pid/cwd/DEFAULT_FASTRTPS_PROFILES.xml" ]] \
      && { echo "DDS_PROFILE: DEFAULT_FASTRTPS_PROFILES.xml nella cartella di lavoro dell'adattatore (pid $pid): run fermata"; exit 1; }
  done
  export FASTRTPS_DEFAULT_PROFILES_FILE="$DDS_PROFILE"
  DDS_FLAG="--dds-profile $DDS_PROFILE"
  echo "profilo DDS: $DDS_PROFILE (md5 $(md5sum < "$DDS_PROFILE" | cut -d' ' -f1))"
fi
# Offline check of the MOLA configuration with the effective MOLA_LIDAR_TOPIC_TYPE:
# the LiDAR entry of victim and attacker must have topic and qos, and the rest of
# the document must stay the same.
if [[ "$TRANSPORT" == "reliable" ]]; then
  for t in /carter/lidar_victim /carter/lidar_attacker; do
    python3 scripts/controlla_qos_yaml.py --topic "$t" --reliability reliable --depth 50 \
      || { echo "controllo della QoS LiDAR fallito per $t: lancio fermato"; exit 1; }
  done
fi
echo "== trasporto: TRANSPORT=$TRANSPORT, ROS_AUTOMATIC_DISCOVERY_RANGE=${ROS_AUTOMATIC_DISCOVERY_RANGE:-<non impostata>}"

ISAAC_PID=""
NODE_PID=""
ORCH_PID=""

# In a script, commands started with & ignore SIGINT: without "trap - INT" the
# INT sent by stop_pid would not arrive and the processes would only be stopped
# with TERM, skipping their orderly shutdown.

# Stop a process: first the given signal, then TERM, then KILL.
stop_pid() {   # $1 = pid, $2 = first signal, $3 = seconds to wait
  local pid="$1"
  [[ -z "$pid" ]] && return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill -"$2" "$pid" 2>/dev/null || true
    for _ in $(seq 1 "$3"); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 15); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -KILL "$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
}

isaac_start() {   # $1 = log file, $2 = live log folder (empty = off)
  ( trap - INT
    if [[ -n "${2:-}" ]]; then export REGISTRI_DAL_VIVO_DIR="$2"; else unset REGISTRI_DAL_VIVO_DIR; fi
    exec python3 -u isaac/run_isaac_standalone.py $ISAAC_ARGS ) > "$1" 2>&1 &
  ISAAC_PID=$!
  local t0=$SECONDS
  until grep -q "\[standalone\] pronto" "$1" 2>/dev/null; do
    if ! kill -0 "$ISAAC_PID" 2>/dev/null; then
      echo "Isaac terminato durante l'avvio (log: $1)"; return 1
    fi
    if (( SECONDS - t0 > ISAAC_READY_SEC )); then
      echo "Isaac non pronto dopo $ISAAC_READY_SEC s (log: $1)"; return 1
    fi
    sleep 2
  done
  sleep 5   # first clouds and first pose
}

isaac_stop() { stop_pid "$ISAAC_PID" INT 30; ISAAC_PID=""; }

# The node starts with the attack off: the orchestrator enables it for the
# rollouts that need it.
node_start() {   # $1 = log file, $2 = gaussian noise sigma (empty = genome)
  local extra=""
  [[ -n "$2" ]] && extra="--gaussian-sigma $2"
  [[ -n "$MAX_DRIFT_LAG_S" ]] && extra="$extra --max-drift-lag-s $MAX_DRIFT_LAG_S"
  [[ -n "${NODE_EXTRA:-}" ]] && extra="$extra $NODE_EXTRA"   # node test options
  # delay instrumentation in the repetition folder (unless NODE_EXTRA already asks for it)
  [[ "$STRUMENTI_NODO" == "1" && " ${NODE_EXTRA:-} " != *" --strumenti-ritardo "* ]] \
    && extra="$extra --strumenti-ritardo $OUT/ritardo_nodo.csv"
  [[ "$PERC_TERMS" == "1" ]] && extra="$extra --perc-terms"
  [[ -n "$STRUCTURAL" ]] && extra="$extra --structural $STRUCTURAL"
  [[ -n "$NODE_SB_FLAGS" ]] && extra="$extra$NODE_SB_FLAGS"
  [[ -n "$RAMPA_NODO" ]] && extra="$extra$RAMPA_NODO"   # genome ramp or genome-in-motion: same value as the orchestrator
  # victim generator state for the realignment (checked by the orchestrator at start)
  [[ "$RIALLINEA" == "1" && " ${NODE_EXTRA:-} " != *" --stato-generatore "* ]] && extra="$extra --stato-generatore"
  [[ "$SALVA_NUVOLE" == "1" && " ${NODE_EXTRA:-} " != *" --salva-nuvole "* ]] \
    && extra="$extra --salva-nuvole $OUT/nuvole --salva-nuvole-k $SALVA_NUVOLE_K"
  if [[ -n "$OPERATOR_NODE_FLAGS" ]]; then
    extra="$extra$OPERATOR_NODE_FLAGS"
    # per-cloud zone log next to the per-frame log
    [[ "$SERIE_B" == "1" && " ${NODE_EXTRA:-} " != *" --struct-log "* ]] && extra="$extra --struct-log $OUT/struct_log.csv"
  fi
  # SERIE_B=1: effective delay of every cloud towards victim and shadow in the run's
  # per-frame log (unless NODE_EXTRA already asks for it)
  if [[ "$SERIE_B" == "1" ]]; then
    [[ " ${NODE_EXTRA:-} " == *" --frame-log "* ]] || extra="$extra --frame-log $OUT/frame_log.csv"
    [[ " ${NODE_EXTRA:-} " == *" --frame-log-delay "* ]] || extra="$extra --frame-log-delay"
  fi
  ( trap - INT; exec python3 -u src/nodes/perturbation_node.py --min-latency-ms "$MIN_LATENCY_MS" \
    --start-disabled $extra ) > "$1" 2>&1 &
  NODE_PID=$!
}

node_stop() { stop_pid "$NODE_PID" INT 10; NODE_PID=""; }

# Probe (SONDA=1): attached before the orchestrator and never restarted during the
# repetition (--sync-wait publisher counts the readers present at start).
sonda_start() {   # $1 = output folder
  [[ "$SONDA" == "1" ]] || return 0
  local topics="--topic /carter/lidar_victim --topic /carter/lidar_attacker --topic /carter/lidar_perturbed"
  [[ " $ORCH_EXTRA " =~ \ --shadow-topic[\ =]([^ ]+) ]] && topics="$topics --topic ${BASH_REMATCH[1]}"
  mkdir -p "$1"
  ( trap - INT; exec python3 -u scripts/sonda_udp.py --out "$1" ) > "$1/sonda_udp.log" 2>&1 &
  SONDA_UDP_PID=$!
  ( trap - INT; exec python3 -u scripts/sonda_nuvole.py --out "$1" $topics ) > "$1/sonda_nuvole.log" 2>&1 &
  SONDA_PID=$!
  for _ in $(seq 100); do [[ -f "$1/pronta" ]] && break; sleep 0.1; done
  [[ -f "$1/pronta" ]] && echo "sonda agganciata ($1)" || echo "sonda non agganciata in 10 s (vedi $1/sonda_nuvole.log)"
}
sonda_stop() { stop_pid "$SONDA_PID" INT 10; SONDA_PID=""; stop_pid "$SONDA_UDP_PID" INT 5; SONDA_UDP_PID=""; }

cleanup() {
  stop_pid "$ORCH_PID" INT 30; ORCH_PID=""
  sonda_stop
  node_stop
  isaac_stop
}
trap cleanup EXIT
trap 'echo "== interrotto"; exit 130' INT
trap 'echo "== terminato"; exit 143' TERM HUP

# Latency of the freshly started node, read from the status it publishes (the
# status only arrives after the first clouds): waits up to 60 s.
check_latency() {
  local lat="" t0=$SECONDS
  while (( SECONDS - t0 < 60 )); do
    lat="$(timeout 10 ros2 topic echo --once --field data /attack/status std_msgs/msg/String 2>/dev/null \
      | python3 -c 'import json,sys; print(json.loads(sys.stdin.readline()).get("min_latency_ms", 0))' 2>/dev/null || true)"
    [[ -n "$lat" ]] && break
    sleep 2
  done
  if ! python3 -c "import sys; sys.exit(0 if abs(float('${lat:-nan}') - $MIN_LATENCY_MS) < 1e-6 else 1)" 2>/dev/null; then
    echo "perturbation_node senza la latenza minima richiesta (letta: '${lat:-nessuno stato}', attesa: $MIN_LATENCY_MS ms)."
    return 1
  fi
}

# Move the folder and log of a repetition to a new name.
move_rep() {   # $1 = base, $2 = name, $3 = new name
  local tag="$3"
  [[ -e "$1/$tag" ]] && tag="${tag}_$(date +%Y%m%d_%H%M%S)"
  [[ -e "$1/$2" ]] && mv "$1/$2" "$1/$tag"
  for suf in .log _isaac.log _pert.log; do
    [[ -e "$1/$2$suf" ]] && mv "$1/$2$suf" "$1/$tag$suf"
  done
  return 0
}

if [[ "$ARM" == "gaussian" ]]; then
  VARIANTS="$SIGMAS"
else
  VARIANTS="-"
fi

for SIGMA in $VARIANTS; do
  case "$ARM" in
    none)          DIR="none_g${GROUP}${MODE_SUFFIX}${RUN_SUFFIX}" ;;
    gaussian)      DIR="gaussian_s${SIGMA}${MODE_SUFFIX}${RUN_SUFFIX}" ;;
    random|nsga3)  DIR="${ARM}_g${GROUP}_p${POP}x${GEN}${BUDGET_SUFFIX}${MODE_SUFFIX}${RUN_SUFFIX}" ;;
    fixed)         DIR="fixed_g${GROUP}${MODE_SUFFIX}${RUN_SUFFIX}" ;;
  esac
  [[ "$SIGMA" == "-" ]] && SIGMA=""
  OUT_BASE="$CAMPAIGN_DIR/$SCENARIO/$DIR"
  mkdir -p "$OUT_BASE"

  for i in $(seq 1 "$REPS"); do
    NAME="rep_$(printf '%02d' "$i")"
    if [[ -n "$SEEDS" ]]; then SEED_I="${SEED_LIST[$((i - 1))]}"; else SEED_I=$((1000 + i)); fi
    OUT="$OUT_BASE/$NAME"
    if [[ -f "$OUT/completata" ]]; then
      echo "== $OUT gia' completata, salto"
      continue
    fi
    # Two Isaac stalls on the same repetition are a recorded outcome: no further
    # retries. The repetition is redone by hand, renaming rep_NN_stallo2.
    if [[ -d "$OUT_BASE/${NAME}_stallo2" ]]; then
      echo "== $OUT ha gia' due stalli registrati (${NAME}_stallo2): salto"
      continue
    fi
    # SERIE_B=1: node per-frame log (the one in NODE_EXTRA, if any) read by the
    # orchestrator for the chain check
    if [[ "$SERIE_B" == "1" && "$ARM" != "none" ]]; then
      if [[ " ${NODE_EXTRA:-} " =~ \ --frame-log\ ([^ ]+) ]]; then
        FRAME_LOG_ORCH="--frame-log-nodo ${BASH_REMATCH[1]}"
      else
        FRAME_LOG_ORCH="--frame-log-nodo $OUT/frame_log.csv"
      fi
    fi
    # Cloud age check: the per-frame log is needed in the baseline too
    ETA_FLAG=""
    if [[ -n "$CONTROLLO_ETA" ]]; then
      if [[ -z "$FRAME_LOG_ORCH" ]]; then
        if [[ " ${NODE_EXTRA:-} " =~ \ --frame-log\ ([^ ]+) ]]; then
          FRAME_LOG_ORCH="--frame-log-nodo ${BASH_REMATCH[1]}"
        elif [[ "$SERIE_B" == "1" ]]; then
          FRAME_LOG_ORCH="--frame-log-nodo $OUT/frame_log.csv"
        else
          echo "CONTROLLO_ETA richiede il registro per frame del nodo (SERIE_B=1 o --frame-log in NODE_EXTRA)"; exit 1
        fi
      fi
      ETA_FLAG="--controllo-eta $CONTROLLO_ETA"
    fi
    if [[ -e "$OUT" || -e "$OUT_BASE/$NAME.log" ]]; then
      echo "== $OUT incompleta da un lancio precedente: rinominata e rifatta"
      move_rep "$OUT_BASE" "$NAME" "${NAME}_incompleta_$(date +%Y%m%d_%H%M%S)"
    fi
    for attempt in 1 2; do
      echo "== $(date +%T) $SCENARIO / $DIR / ripetizione $i (tentativo $attempt)  ->  $OUT"
      REG_DIR=""
      if [[ "$REGISTRI_DAL_VIVO" == "1" ]]; then REG_DIR="$OUT/registri_dal_vivo"; mkdir -p "$REG_DIR"; fi
      isaac_start "$OUT_BASE/${NAME}_isaac.log" "$REG_DIR" || exit 1
      node_start "$OUT_BASE/${NAME}_pert.log" "$SIGMA"
      check_latency || exit 1
      if [[ "$SERIE_B" == "1" ]]; then
        # SERIE_B=1: budget and orchestrator options next to the results
        # (same arguments as the line below).
        mkdir -p "$OUT"
        { echo "CHAMFER_BUDGET=${CHAMFER_BUDGET:-} SELECT=$SELECT SERIE_B=$SERIE_B"
          echo "python3 -u src/optimization/attack_orchestrator.py --goals \"$GOALS\" --start-pose=\"$START_POSE\"" \
               "--horizon 1.0 --settle-sec 5.0 --stop-on \"$STOP_ON\" $WINDOW_STOP_FLAGS --attack-from \"$ATTACK_FROM\"" \
               "--drift-reset-on \"$DRIFT_RESET_ON\" --victim \"$VICTIM\" --select \"$SELECT\" --topk \"$TOPK\" $WARM_FLAG" \
               "--seed \"$SEED_I\" --trace --reeval --out \"$OUT\" $EXTRA $SERIEB_FLAGS $FRAME_LOG_ORCH $DDS_FLAG $ETA_FLAG $G1_FLAG $REG_FLAG $ORCH_EXTRA"
          echo "OPERATOR_MODE=${OPERATOR_MODE:-random} ZONE_ATTACCANTE=${ZONE_ATTACCANTE:-0} SORGENTE_ZONE=$SORGENTE_ZONE"
        } > "$OUT/comando.txt"
        # node interpreter versions and libraries loaded by the freshly started node
        python3 scripts/ambiente_nodo.py versioni --riferimento "${AMBIENTE_RIFERIMENTO:-config/ambiente_nodo_serieB.txt}" ${NODE_PID:+--pid "$NODE_PID"} \
          > "$OUT/ambiente.txt" 2>&1 || echo "versioni dell'ambiente non scritte ($OUT/ambiente.txt)"
        [[ -z "$DDS_PROFILE" ]] || echo "profilo DDS: FASTRTPS_DEFAULT_PROFILES_FILE=$DDS_PROFILE md5 $(md5sum < "$DDS_PROFILE" | cut -d' ' -f1)" >> "$OUT/ambiente.txt"
        [[ "$MOLA_BINARIO_G1" != "1" ]] || echo "binario di MOLA: install_G1 per tutte le MOLA (MOLA_BINARIO_G1=1, --mola-binario-g1; variabili in history.json, mola_binario_g1)" >> "$OUT/ambiente.txt"
        [[ "$SONDA" != "1" ]] || echo "sonda: SONDA=1, uscita in $OUT/sonda" >> "$OUT/ambiente.txt"
        [[ "$REGISTRI_DAL_VIVO" != "1" ]] || echo "registri dal vivo: REGISTRI_DAL_VIVO=1, uscita in $OUT/registri_dal_vivo" >> "$OUT/ambiente.txt"
        [[ "$STRUMENTI_NODO" != "1" ]] || echo "strumenti del ritardo del nodo: STRUMENTI_NODO=1, uscita in $OUT/ritardo_nodo*.csv" >> "$OUT/ambiente.txt"
        [[ -z "$GENOMA_IN_MOTO" ]] || echo "genoma in moto (variante E1): GENOMA_IN_MOTO=$GENOMA_IN_MOTO, --genoma-in-moto $GENOMA_IN_MOTO a nodo e orchestratore (kmin e kmax di default)" >> "$OUT/ambiente.txt"
        [[ "$CHAMFER_PER_FASE" != "1" ]] || echo "Chamfer per fase: CHAMFER_PER_FASE=1, --chamfer-per-fase a nodo e orchestratore (history: chamfer_fasi)" >> "$OUT/ambiente.txt"
      fi
      if [[ "$DISCOVERY_LOCALE" == "1" ]]; then
        # value read from the environment of the freshly started processes and of the adapter
        mkdir -p "$OUT"
        { printf 'scoperta: DISCOVERY_LOCALE=1, ROS_AUTOMATIC_DISCOVERY_RANGE=%s (%s; ROS_LOCALHOST_ONLY tolta; orchestratore, MOLA e sonda la ereditano);' "$ROS_AUTOMATIC_DISCOVERY_RANGE" "$SCOPERTA_MOTIVO"
          printf ' isaac pid %s: %s;' "$ISAAC_PID" "$(range_di "$ISAAC_PID")"
          printf ' nodo pid %s: %s;' "$NODE_PID" "$(range_di "$NODE_PID")"
          for pid in $(pgrep -f "^python3( -u)? .*add_intensity_node\.py"); do printf ' adattatore pid %s: %s;' "$pid" "$(range_di "$pid")"; done
          echo; } >> "$OUT/ambiente.txt"
      fi
      sonda_start "$OUT/sonda"
      set +e
      ( trap - INT; exec python3 -u src/optimization/attack_orchestrator.py \
        --goals "$GOALS" --start-pose="$START_POSE" --horizon 1.0 --settle-sec 5.0 --stop-on "$STOP_ON" $WINDOW_STOP_FLAGS \
        --attack-from "$ATTACK_FROM" --drift-reset-on "$DRIFT_RESET_ON" --victim "$VICTIM" \
        --select "$SELECT" --topk "$TOPK" $WARM_FLAG \
        --seed "$SEED_I" --trace --reeval --out "$OUT" $EXTRA $SERIEB_FLAGS $FRAME_LOG_ORCH $DDS_FLAG $ETA_FLAG $G1_FLAG $REG_FLAG $ORCH_EXTRA ) \
        > >(tee "$OUT_BASE/${NAME}.log") 2>&1 &
      ORCH_PID=$!
      wait "$ORCH_PID"
      rc=$?
      ORCH_PID=""
      set -e
      sonda_stop
      node_stop
      isaac_stop
      if [[ $rc -eq 0 ]]; then
        date +%FT%T > "$OUT/completata"
        break
      fi
      if [[ $rc -ne 0 && -f "$OUT/da_rifare.json" ]]; then
        # chain fault (--controllo-catena): not a result; the repetition stays without
        # "completata" and is redone with the same seed
        echo "== guasto della catena nella ripetizione $i (seme $SEED_I): da rifare ($OUT/da_rifare.json)"
        DA_RIFARE+=("$OUT")
        break
      fi
      [[ $rc -ne 3 ]] && { echo "== orchestratore uscito con codice $rc: ripetizione $i non completata"; break; }
      # Isaac stall: the attempt is kept, renamed, and the repetition restarts.
      echo "== stallo di Isaac nella ripetizione $i (tentativo $attempt)"
      move_rep "$OUT_BASE" "$NAME" "${NAME}_stallo${attempt}"
    done
  done
done
if (( ${#DA_RIFARE[@]} )); then
  # code 4: repetitions to redo with the same seed (scripts/coda_serieB.sh relaunches them)
  printf '== ripetizione da rifare (guasto della catena): %s\n' "${DA_RIFARE[@]}"
  echo "== campagna $SCENARIO / $ARM / gruppo $GROUP chiusa con ${#DA_RIFARE[@]} ripetizioni da rifare"
  exit 4
fi
echo "== campagna $SCENARIO / $ARM / gruppo $GROUP completata"
