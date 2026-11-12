#!/usr/bin/env bash
# Run queue: the conditions picked from a list, in blocks per seed, one simulation at a
# time, resuming from where it stopped.
#
# Usage (from the repository root, in a dedicated tmux window):
#   scripts/coda_serieB.sh --lista                         conditions, estimated hours and disk
#   scripts/coda_serieB.sh --stato [CONDIZIONI...]         queue state and summary
#   scripts/coda_serieB.sh --a-secco CONDIZIONI...         print checks and commands, launch nothing
#   scripts/coda_serieB.sh CONDIZIONI...                   launch (or resume) the queue
#   scripts/coda_serieB.sh --semi "1001 2001" CONDIZIONI...
# CONDIZIONI: ids from the list (baseline, nsga3_senzabudget, nsga3_eps5 ... massimo),
# patterns with * ('nsga3_*', 'casuale_*') or "tutte". Within each seed block the order
# is always the design order, whatever the order on the command line.
# Relaunching the same command after an interruption resumes from the first unfinished run.
#
# Every run is a run_campaign.sh straight with SERIE_B=1 (direction goals, 5 windows plus
# 3 held out, D_min 140 ms, spawn, ramp k 1.0, REL, gross, --twin-init attacker_est,
# frame log with delay, shadow crc), Chamfer budget from CHAMFER_BUDGET. Before every
# run: chain checks (bench library), Isaac watchdog, disk space, adapter restart, code
# snapshot. During the run: PROVA_IN_CORSO.json lock, maximum time, true odom x, expected
# options in the command. After: esito_run.json in the run folder.
#
# Outcomes: fatta (also "attack detected"); guasto_catena (code 4 of run_campaign.sh or
# the orchestrator's da_rifare.json with --controllo-catena, run_end chain fault in the
# history, any other exit code, Isaac killed by the watchdog, maximum time, adapter lost,
# shadow diverging in at least one metre: shadow_check.identita_da_rep) and oltre_ritardo
# (code 3 of the delay check): repeated with the same seed in a new folder
# s<seed>_rip<n>, at most 2 times; fuori_spazio (x beyond X_MAX_M): not repeated;
# interrotta, bloccata, configurazione, disco and ritardo_* (codes 4 and 5 of the delay
# check, or check failed): the queue stops with a [!] entry.
#
# Delay check (block [4]): CONTROLLO_RITARDO, default
# "python3 scripts/controllo_ritardo_run.py {run}"; {run} (run folder) and {frame_log}
# are substituted, without placeholders the folder is appended. Codes: 0 compliant,
# 3 over the delay limit, 4 invalid configuration, 5 not measurable. CONTROLLO_RITARDO=""
# turns it off (with a warning). Never run on the baseline.
#
# Damage baseline of the attacked runs: BASELINE_SERIEB (mandatory, {seme} = seed of the
# run). Without it the queue does not launch attacked runs and stops if the variable is
# missing or the file does not exist.
#
# Files: runs in $RADICE/<condition>/s<seed>[_rip<n>]/ (comando.txt, eventi.txt,
# ambiente.txt, fotografia_codice.txt, stato_catena_prima.txt, frame_log.csv,
# shadow_crc.csv, campagna/, esito_run.json, controllo_ritardo.txt, riepilogo_run.txt and
# .json: victim rejections and pose losses, [4], end of run, Chamfer and damage); state in
# $RADICE/coda/stato_coda.json; daily summary in
# $RADICE/coda/riepiloghi/riepilogo_AAAA-MM-GG.txt (with free GB on the repository disk
# and the --lista estimate for the remaining runs); queue events in $RADICE/coda/eventi.txt,
# run output in $RADICE/coda/coda.log.
#
# Chain checks (from the environment, off by default, passed to run_campaign.sh and
# written in the run command): CONTROLLO_ETA=<s> (cloud age), DDS_PROFILE=<xml> (Fast DDS
# profile), SONDA=1 (cloud and UDP probe), MOLA_BINARIO_G1=1 (every MOLA on install_G1).
#
# Winner options (from the environment, off by default, passed to run_campaign.sh and
# looked for in the run command only when enabled): PROVE_VINCITORE=<N> (identical trials
# of the winner), RIALLINEA=1 (twin realigned after settling), REGISTRA_ASSETTO=1,
# TWIN_INIT_ASSETTO=conservato|attaccante, CONFRONTO_VINCITORE=1. Selection within the
# Chamfer budget uses SELECT=pareto_maxdamage (TOPK ignored).
#
# Null genome (from the environment, 0 or 1, off by default): GENOMA_NULLO=1 passes
# --genoma-nullo-se-nessun-ammissibile to the orchestrator of the attacked runs (in
# ORCH_EXTRA) and looks for it in the command: a window without an admissible genome
# applies the null genome instead of holding the victim.
#
# Limits study (from the environment, off by default; with any of them on, RADICE must
# contain "limiti", and with the multiplier also x<K>): MOLTIPLICATORE_LIMITI=K (number
# >= 1, 1 = off) with FAMIGLIE_LIMITI (deriva,rumore,geometria,strutturale,dropout, default
# all) passes --moltiplicatore-limiti K --famiglie-limiti F to node and orchestrator of the
# attacked runs, looks for them in the commands and appends _limx<K>_<families> to the
# campaign name (K with p instead of the dot); BLOCCA_G14=1 passes --blocca-gene 14=-1 to
# the orchestrator (g14 at -1 in every candidate) and appends _g14bloccato;
# REGOLA_CRASH_LIMITI=1, fixed genome only (massimo): a crash of the victim MOLA process
# (victim without pose with mola-cli terminated) = outcome crash_processo, run repeated
# once; a second crash at the same level (same K and genome) = crash_da_attacco (final),
# otherwise guasto_catena. Rejection caused by the attack (process alive, ICP rejected)
# and the other victim-without-pose cases stay "fatta: attacco scoperto", with the marker
# in the details.
#
# Stopping: Ctrl-C in the window (the current run closes in order and stays "interrotta";
# on relaunch it is redone in a new folder). MOLA YAML and launch files are never touched.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
SE="$HERE/$(basename "${BASH_SOURCE[0]}")"
cd "$ROOT"
PY="python3 -I $HERE/coda_serieB_stato.py"

# --- Configuration (environment variables) ---
RADICE="${RADICE:-$ROOT/data/attack/serieB}"
CODA_DIR="$RADICE/coda"
STATO_FILE="$CODA_DIR/stato_coda.json"
RIEPILOGHI="$CODA_DIR/riepiloghi"
LOG_CODA="$CODA_DIR/coda.log"
SEMI="${SEMI:-1001 2001 3001}"                    # seeds, one block each
FINESTRE="${FINESTRE:-5}"                          # number of windows N
EXTRA_FINESTRE="${EXTRA_FINESTRE:-3}"              # extra windows after N
POP="${POP:-32}"; GEN="${GEN:-16}"; GRUPPO="${GRUPPO:-3}"   # NSGA-III population, generations, gene group
SELECT="${SELECT:-maxdamage}"; TOPK="${TOPK:-2}"   # selection rule and re-evaluated top-k
export MIN_LATENCY_MS="${MIN_LATENCY_MS:-140}"     # node minimum latency (ms)
SETTLE_CLOUDS="${SETTLE_CLOUDS:-20}"; REOPEN_CLOUDS="${REOPEN_CLOUDS:-20}"
PAUSE_SEC="${PAUSE_SEC:-1800}"                     # baseline pauses (s)
K_RANDPURO="${K_RANDPURO:-64}"                     # pure random: at most K draws
SAMPLING="${SAMPLING:-}"                           # empty = legacy
GEN_MAX="${GEN_MAX:-0,1,0,1,1,1,1,1,0,0,1,1,1,1,1,1,1}"   # genome of the massimo condition
# mandatory: path of the damage baseline for the attacked runs
BASELINE_PATH="${BASELINE_SERIEB:-}"               # {seme} = seed of the run
X_MAX_M="${X_MAX_M:-7.0}"; CONTROLLO_X_S="${CONTROLLO_X_S:-30}"
CONTROLLO_RITARDO="${CONTROLLO_RITARDO-python3 scripts/controllo_ritardo_run.py {run}}"   # delay check (block [4]); "" = off
TIMEOUT_RITARDO_S="${TIMEOUT_RITARDO_S:-1800}"
RUN_END_GUASTO="${RUN_END_GUASTO:-guasto_catena}"  # expression on the history run_end.reason
OPZIONI_ATTESE_EXTRA="${OPZIONI_ATTESE_EXTRA:-}"   # extra options to find in the orchestrator command
TEMPO_MAX_FATTORE="${TEMPO_MAX_FATTORE:-1.5}"
RIAVVIO_ADATTATORE="${RIAVVIO_ADATTATORE:-sempre}" # sempre | preventivo (only past 4 h)
PAUSA_FRA_RUN_S="${PAUSA_FRA_RUN_S:-30}"
PILOTA_DISCO="${PILOTA_DISCO:-$ROOT/data/attack/verifica/2026-10-26_pilota_G3_dmin140}"
BASELINE_DISCO="${BASELINE_DISCO:-$ROOT/data/attack/verifica/2026-10-25_baseline_direction_s1001_sosta0}"
ORE_T6="${ORE_T6:-$ROOT/data/attack/dossier/2026-10-23_serieB/T6_ore_G3_N5/ore_condizioni.csv}"
WATCHDOG_FASE3="${WATCHDOG_FASE3:-}"               # Isaac watchdog path: if set, the queue starts it when missing
export VAULT_LOG_PREFISSO="${VAULT_LOG_PREFISSO:-Log serie B - }"
COSTRUISCI="scripts/costruisci_baseline_direction.py"
CONTROLLO_ETA="${CONTROLLO_ETA:-}"                 # threshold (s) of the cloud age check; empty = off
CONTROLLO_OMBRA_VIVA="${CONTROLLO_OMBRA_VIVA:-0}"   # 1 = --controllo-ombra-viva to the orchestrator; 0 = off
ORIGINE_COERENTE="${ORIGINE_COERENTE:-0}"           # 1 = --origine-coerente-vittima to the orchestrator; 0 = off
OPZ_0311_DESC=""                                    # in the command text only when enabled: with 0 it is unchanged
[[ "$CONTROLLO_OMBRA_VIVA" == "1" ]] && OPZ_0311_DESC+=" CONTROLLO_OMBRA_VIVA=1"
[[ "$ORIGINE_COERENTE" == "1" ]] && OPZ_0311_DESC+=" ORIGINE_COERENTE=1"
GENOMA_NULLO="${GENOMA_NULLO:-0}"                   # 1 = --genoma-nullo-se-nessun-ammissibile to the orchestrator; 0 = off
OPZ_0511_DESC=""                                    # in the command text only when enabled: with 0 it is unchanged
[[ "$GENOMA_NULLO" == "1" ]] && OPZ_0511_DESC+=" GENOMA_NULLO=1"
# limits study: values checked here, errors reported by the run checks; in the
# command text and in the campaign name only when enabled
MOLTIPLICATORE_LIMITI="${MOLTIPLICATORE_LIMITI:-1}"   # K >= 1; 1 = off
FAMIGLIE_LIMITI="${FAMIGLIE_LIMITI:-deriva,rumore,geometria,strutturale,dropout}"
BLOCCA_G14="${BLOCCA_G14:-0}"                       # 1 = --blocca-gene 14=-1 to the orchestrator; 0 = off
REGOLA_CRASH_LIMITI="${REGOLA_CRASH_LIMITI:-0}"     # 1 = victim crash rule; 0 = off
LIM_ERR=""; LIM_MOL=0; LIM_K=""; LIM_FAM=""; LIM_SUFF=""; G14_SUFF=""; OPZ_LIMITI_DESC=""; LIM_RIGA=""
IFS=$'\t' read -r LIM_ESITO LIM_K LIM_FAM LIM_SUFF < <(python3 -I -c '
import math, sys
ammesse = ("deriva", "rumore", "geometria", "strutturale", "dropout")
k, fam = sys.argv[1], sys.argv[2]
try:
    kv = float(k)
    ok = math.isfinite(kv) and kv >= 1 and k.strip() != "" and all(c in "0123456789." for c in k)
except ValueError:
    ok = False
if not ok:
    print("ERR\tMOLTIPLICATORE_LIMITI: numero >= 1, solo cifre e punto (dato: %s)" % k)
    sys.exit()
f = [x.strip() for x in fam.split(",")]
if "" in f or any(x not in ammesse for x in f) or len(set(f)) != len(f):
    print("ERR\tFAMIGLIE_LIMITI: fra %s, separate da virgole, senza ripetizioni (dato: %s)" % (",".join(ammesse), fam))
    sys.exit()
ks = str(int(kv)) if kv == int(kv) else repr(kv)
fc = ",".join(x for x in ammesse if x in f)
print("OK\t%s\t%s\t_limx%s_%s" % (ks, fc, ks.replace(".", "p"), fc.replace(",", "-")))
' "$MOLTIPLICATORE_LIMITI" "$FAMIGLIE_LIMITI")
if [[ "$LIM_ESITO" == "OK" ]]; then
  if [[ "$LIM_K" != "1" ]]; then
    LIM_MOL=1
    OPZ_LIMITI_DESC+=" MOLTIPLICATORE_LIMITI=$LIM_K FAMIGLIE_LIMITI=$LIM_FAM"
    LIM_RIGA+="moltiplicatore x$LIM_K su $LIM_FAM (suffisso $LIM_SUFF); "
  else LIM_SUFF=""; fi
else
  LIM_ERR="${LIM_K:-MOLTIPLICATORE_LIMITI o FAMIGLIE_LIMITI non controllati}"; LIM_K=""; LIM_FAM=""; LIM_SUFF=""
fi
if [[ "$BLOCCA_G14" == "1" ]]; then
  G14_SUFF="_g14bloccato"; OPZ_LIMITI_DESC+=" BLOCCA_G14=1"
  LIM_RIGA+="g14 bloccato a -1 in tutti i candidati (suffisso $G14_SUFF); "
fi
if [[ "$REGOLA_CRASH_LIMITI" == "1" ]]; then
  OPZ_LIMITI_DESC+=" REGOLA_CRASH_LIMITI=1"
  LIM_RIGA+="regola del crash della vittima accesa (una ripetizione, poi crash_da_attacco o guasto); "
fi
# horizon ablation: ORIZZONTE_M = window length in metres, 1 = off. When on, same
# 5 m straight line, so FINESTRE * ORIZZONTE_M = 5; root containing "orizzonte"
# and H<horizon> in the name; suffix _H<horizon> in the campaign.
ORIZZONTE_M="${ORIZZONTE_M:-1}"; ORZ_ERR=""; ORZ_SUFF=""; ORZ_DESC=""; ORZ_TIMEOUT=""
if [[ "$ORIZZONTE_M" != "1" ]]; then
  if ! [[ "$ORIZZONTE_M" =~ ^[0-9]+([.][0-9]+)?$ ]]; then ORZ_ERR="ORIZZONTE_M: numero > 0, solo cifre e punto (dato: $ORIZZONTE_M)"
  elif ! python3 -I -c 'import sys; h=float(sys.argv[1]); n=int(sys.argv[2]); sys.exit(0 if h > 0 and abs(n*h - 5.0) < 1e-9 else 1)' "$ORIZZONTE_M" "$FINESTRE"; then
    ORZ_ERR="ORIZZONTE_M=$ORIZZONTE_M con FINESTRE=$FINESTRE: il rettilineo e' di 5 m (FINESTRE * ORIZZONTE_M = 5)"
  else ORZ_SUFF="_H${ORIZZONTE_M//./p}"; ORZ_DESC=" ORIZZONTE_M=$ORIZZONTE_M"
    # rollout drive timeout: 45 s per metre of window (orchestrator default 45 s at 1 m)
    ORZ_TIMEOUT=$(python3 -I -c 'import sys; print(f"{45.0*float(sys.argv[1]):g}")' "$ORIZZONTE_M"); fi
fi
LIM_ON=0
[[ "$LIM_MOL" == "1" || "$BLOCCA_G14" != "0" || "$REGOLA_CRASH_LIMITI" != "0" || -n "$LIM_ERR" ]] && LIM_ON=1
SONDA="${SONDA:-0}"; MOLA_BINARIO_G1="${MOLA_BINARIO_G1:-0}"
# winner options: run_campaign.sh environment and expected options, only when enabled
D31_ENV=(); D31_ATTESE=(); D31_DESC=""; D31_NODO_ATTESE=()
if [[ -n "${PROVE_VINCITORE:-}" && "${PROVE_VINCITORE}" != "0" ]]; then
  D31_ENV+=(PROVE_VINCITORE="$PROVE_VINCITORE"); D31_ATTESE+=("--prove-vincitore $PROVE_VINCITORE"); D31_DESC+=" PROVE_VINCITORE=$PROVE_VINCITORE"
fi
if [[ "${RIALLINEA:-0}" == "1" ]]; then
  D31_ENV+=(RIALLINEA=1); D31_ATTESE+=("--riallinea-dopo-assestamento"); D31_DESC+=" RIALLINEA=1"
  D31_NODO_ATTESE+=("--stato-generatore")
fi
if [[ "${SALVA_NUVOLE:-0}" == "1" ]]; then
  D31_ENV+=(SALVA_NUVOLE=1 SALVA_NUVOLE_K="${SALVA_NUVOLE_K:-20}")
  D31_NODO_ATTESE+=("--salva-nuvole" "--salva-nuvole-k ${SALVA_NUVOLE_K:-20}")
  D31_DESC+=" SALVA_NUVOLE=1 SALVA_NUVOLE_K=${SALVA_NUVOLE_K:-20}"
fi
if [[ "${REGISTRA_ASSETTO:-0}" == "1" ]]; then
  D31_ENV+=(REGISTRA_ASSETTO=1); D31_ATTESE+=("--registra-assetto"); D31_DESC+=" REGISTRA_ASSETTO=1"
fi
if [[ -n "${TWIN_INIT_ASSETTO:-}" && "${TWIN_INIT_ASSETTO}" != "azzerato" ]]; then
  D31_ENV+=(TWIN_INIT_ASSETTO="$TWIN_INIT_ASSETTO"); D31_ATTESE+=("--twin-init-assetto $TWIN_INIT_ASSETTO"); D31_DESC+=" TWIN_INIT_ASSETTO=$TWIN_INIT_ASSETTO"
fi
if [[ "${CONFRONTO_VINCITORE:-0}" == "1" ]]; then
  D31_ENV+=(CONFRONTO_VINCITORE=1); D31_ATTESE+=("--confronto-vincitore"); D31_DESC+=" CONFRONTO_VINCITORE=1"
fi
if [[ -n "${RAMPA:-}" && "${RAMPA}" != "1" ]]; then
  D31_ENV+=(RAMPA="$RAMPA"); D31_ATTESE+=("--rampa-genoma $RAMPA"); D31_NODO_ATTESE+=("--rampa-genoma $RAMPA")
  D31_DESC+=" RAMPA=$RAMPA"
fi
# genome in motion: threshold S in cm, empty or 0 = off; run_campaign rejects it with RAMPA
if [[ -n "${GENOMA_IN_MOTO:-}" && ! "${GENOMA_IN_MOTO}" =~ ^0+(\.0*)?$ ]]; then
  D31_ENV+=(GENOMA_IN_MOTO="$GENOMA_IN_MOTO"); D31_ATTESE+=("--genoma-in-moto $GENOMA_IN_MOTO")
  D31_NODO_ATTESE+=("--genoma-in-moto $GENOMA_IN_MOTO"); D31_DESC+=" GENOMA_IN_MOTO=$GENOMA_IN_MOTO"
fi
if [[ "${CHAMFER_PER_FASE:-0}" == "1" ]]; then
  D31_ENV+=(CHAMFER_PER_FASE=1); D31_ATTESE+=("--chamfer-per-fase"); D31_NODO_ATTESE+=("--chamfer-per-fase")
  D31_DESC+=" CHAMFER_PER_FASE=1"
fi

# --- Arguments ---
A_SECCO=0; MODO=coda; VOCI=(); ESEGUI=()
while (( $# )); do
  case "$1" in
    --a-secco) A_SECCO=1 ;;
    --lista) MODO=lista ;;
    --stato) MODO=stato ;;
    --semi) SEMI="${2:?--semi richiede un elenco}"; shift ;;
    --esegui-run) MODO=run; ESEGUI=("${2:?}" "${3:?}" "${4:?}"); shift 3 ;;
    -h|--aiuto) awk 'NR > 1 && !/^#/ {exit} NR > 1 {sub(/^# ?/, ""); print}' "$SE"; exit 0 ;;
    -*) echo "opzione sconosciuta: $1"; exit 2 ;;
    *) VOCI+=("$1") ;;
  esac
  shift
done
[[ -z "${ROS_DISTRO:-}" && -f /opt/ros/jazzy/setup.bash ]] && { set +u; source /opt/ros/jazzy/setup.bash; set -u; }

stima() { $PY stima --pilota "$PILOTA_DISCO" --baseline-dir "$BASELINE_DISCO" --ore "$ORE_T6" \
            --finestre "$FINESTRE" --extra "$EXTRA_FINESTRE" --pop "$POP" --gen "$GEN" --topk "$TOPK" --k-max "$K_RANDPURO" "$@"; }

# ======================================================================
# One run (child process in its own session): $1 condition, $2 seed, $3 folder
# Exit codes without esito_run.json: 20 checks failed, 21 baseline already present.
# ======================================================================
esegui_run() {
  COND="$1"; SEME="$2"; OUT="$3"; [[ "$OUT" == /* ]] || OUT="$ROOT/$OUT"
  BRACCIO=""; BUDGET=""; TIPO=""   # globals: also needed at closing (trap EXIT)
  IFS=$'\t' read -r _ BRACCIO BUDGET TIPO _ < <($PY condizioni | awk -F'\t' -v c="$COND" '$1==c')
  [[ -n "${BRACCIO:-}" ]] || { echo "condizione sconosciuta: $COND"; exit 2; }
  [[ "$BUDGET" == "-" ]] && BUDGET=""
  COMANDO_PROVA="$SE --esegui-run $COND $SEME $OUT (FINESTRE=$FINESTRE MIN_LATENCY_MS=$MIN_LATENCY_MS POP=$POP GEN=$GEN SELECT=$SELECT TOPK=$TOPK SAMPLING=$SAMPLING BASELINE_PATH=$BASELINE_PATH CONTROLLO_RITARDO=$CONTROLLO_RITARDO CONTROLLO_ETA=$CONTROLLO_ETA${OPZ_LIMITI_DESC}${ORZ_DESC}${OPZ_0311_DESC}${OPZ_0511_DESC} DDS_PROFILE=${DDS_PROFILE:-} SONDA=$SONDA MOLA_BINARIO_G1=$MOLA_BINARIO_G1${D31_DESC})"
  # shellcheck source=prova_banco_serieB_lib.sh
  source "$HERE/prova_banco_serieB_lib.sh"
  local BASE_CSV="" BPATH="${BASELINE_PATH//\{seme\}/$SEME}" TEMPO_MAX_S
  TEMPO_MAX_S=$($PY tempo-max --ore "$ORE_T6" --finestre "$FINESTRE" --extra "$EXTRA_FINESTRE" --fattore "$TEMPO_MAX_FATTORE" "$COND")
  local SUFF; SUFF="$(basename "$OUT")"; SUFF="${SUFF#s"$SEME"}"
  [[ "$TIPO" == "baseline" ]] && BASE_CSV="$ROOT/data/attack/baseline/rettilineo_direction_N${FINESTRE}_s${SEME}_dmin${MIN_LATENCY_MS}_sosta${PAUSE_SEC%.*}${SUFF}.csv"

  echo "== run $COND, seme $SEME -> $OUT $( [[ $A_SECCO == 1 ]] && echo '(a secco)')"
  echo "braccio $BRACCIO, budget ${BUDGET:-nessuno}, tempo massimo $TEMPO_MAX_S s"
  if [[ "$LIM_ON" == "1" && "$TIPO" == "baseline" ]]; then echo "studio dei limiti: senza effetto sulla baseline (nessun attacco)"
  elif [[ "$LIM_ON" == "1" && -n "$LIM_RIGA" ]]; then echo "studio dei limiti: ${LIM_RIGA%; }"
  elif [[ "$LIM_ON" == "1" ]]; then echo "studio dei limiti: valori non validi, vedi i controlli falliti"; fi
  controlli_iniziali 1
  for o in '"--goal-mode"' '"--extra-windows"' '"--victim-never-clean"' '"--fresh-cloud-after-restore"' '"--settle-clouds"' \
           '"--shadow-check"' '"--shadow-crc-log"' '"--victim-shadow-env"' '"--mola-stop-signal"' '"--damage-ref"' \
           '"--twin-init"' '"--twin-state"' '"--chiudi-attacco-scoperto"' '"--sync-guard"' '"--random-first-admissible"' \
           '"--controllo-catena"' '"--mola-se-ne-accorge"' '"--danno-metro"'; do
    grep -q -- "$o" src/optimization/attack_orchestrator.py || fallito "$o assente dall'orchestratore"
  done
  for o in '"--frame-log-delay"' '"--fresh-cloud-after-restore"' '"--victim-never-clean"' '"--shadow-crc-log"' \
           '"--measures"' '"--shared-nn"' '"--twin-clock-base"'; do
    grep -q -- "$o" src/nodes/perturbation_node.py || fallito "$o assente dal nodo"
  done
  [[ -f "$COSTRUISCI" ]] || fallito "$COSTRUISCI assente"
  # chain checks enabled from the environment: options present in the code
  if [[ "$CONTROLLO_OMBRA_VIVA" == "1" ]]; then
    grep -q -- '"--controllo-ombra-viva"' src/optimization/attack_orchestrator.py || fallito "--controllo-ombra-viva assente dall'orchestratore (CONTROLLO_OMBRA_VIVA)"
  elif [[ "$CONTROLLO_OMBRA_VIVA" != "0" ]]; then fallito "CONTROLLO_OMBRA_VIVA: 0 o 1 (dato: $CONTROLLO_OMBRA_VIVA)"; fi
  if [[ "$ORIGINE_COERENTE" == "1" ]]; then
    grep -q -- '"--origine-coerente-vittima"' src/optimization/attack_orchestrator.py || fallito "--origine-coerente-vittima assente dall'orchestratore (ORIGINE_COERENTE)"
  elif [[ "$ORIGINE_COERENTE" != "0" ]]; then fallito "ORIGINE_COERENTE: 0 o 1 (dato: $ORIGINE_COERENTE)"; fi
  if [[ "$GENOMA_NULLO" == "1" ]]; then
    grep -q -- '"--genoma-nullo-se-nessun-ammissibile"' src/optimization/attack_orchestrator.py || fallito "--genoma-nullo-se-nessun-ammissibile assente dall'orchestratore (GENOMA_NULLO)"
  elif [[ "$GENOMA_NULLO" != "0" ]]; then fallito "GENOMA_NULLO: 0 o 1 (dato: $GENOMA_NULLO)"; fi
  # limits study: values, options in the code, dedicated root
  [[ -n "$LIM_ERR" ]] && fallito "$LIM_ERR"
  [[ -n "$ORZ_ERR" ]] && fallito "$ORZ_ERR"
  if [[ "$ORIZZONTE_M" != "1" ]]; then
    [[ "$RADICE" == *orizzonte* && "$(basename "$RADICE")" == *"H${ORIZZONTE_M//./p}"* ]] \
      || fallito "ORIZZONTE_M=$ORIZZONTE_M: la radice deve contenere 'orizzonte' e H${ORIZZONTE_M//./p} nell'ultima cartella (dato: $RADICE)"
    [[ "$TIPO" == "baseline" ]] && fallito "ORIZZONTE_M acceso con la baseline: la baseline del danno resta quella dei 3 semi"
    echo "orizzonte: finestre da $ORIZZONTE_M m, $FINESTRE finestre (suffisso $ORZ_SUFF)"
  fi
  if [[ "$LIM_MOL" == "1" ]]; then
    grep -q -- '"--moltiplicatore-limiti"' src/optimization/attack_orchestrator.py || fallito "--moltiplicatore-limiti assente dall'orchestratore (MOLTIPLICATORE_LIMITI)"
    grep -q -- '"--moltiplicatore-limiti"' src/nodes/perturbation_node.py || fallito "--moltiplicatore-limiti assente dal nodo (MOLTIPLICATORE_LIMITI)"
    grep -qF 'FAMIGLIE_LIMITI = ("deriva", "rumore", "geometria", "strutturale", "dropout")' src/perturbations/perturbation_generator.py \
      || fallito "famiglie dei limiti del generatore diverse da quelle della coda (FAMIGLIE_LIMITI)"
  fi
  if [[ "$BLOCCA_G14" == "1" ]]; then
    grep -q -- '"--blocca-gene"' src/optimization/attack_orchestrator.py || fallito "--blocca-gene assente dall'orchestratore (BLOCCA_G14)"
    if [[ "$TIPO" == "massimo" ]] && ! [[ "$(cut -d, -f15 <<< "$GEN_MAX")" =~ ^-1(\.0*)?$ ]]; then
      fallito "BLOCCA_G14=1 con il massimo: il gene 14 di GEN_MAX deve essere -1 (dato: $(cut -d, -f15 <<< "$GEN_MAX"))"
    fi
  elif [[ "$BLOCCA_G14" != "0" ]]; then fallito "BLOCCA_G14: 0 o 1 (dato: $BLOCCA_G14)"; fi
  if [[ "$REGOLA_CRASH_LIMITI" == "1" ]]; then
    grep -q -- '"--regola-crash"' scripts/coda_serieB_stato.py || fallito "--regola-crash assente da coda_serieB_stato.py (REGOLA_CRASH_LIMITI)"
    # fixed genome only: with a search the genome of the repeated attempt differs
    [[ "$TIPO" == "ricerca" || "$TIPO" == "randpuro" ]] \
      && fallito "REGOLA_CRASH_LIMITI=1 vale solo a genoma fisso (massimo), non con $COND"
    grep -q -- '"processo_mola_vittima"' src/optimization/attack_orchestrator.py || fallito "processo_mola_vittima assente dall'orchestratore (REGOLA_CRASH_LIMITI)"
  elif [[ "$REGOLA_CRASH_LIMITI" != "0" ]]; then fallito "REGOLA_CRASH_LIMITI: 0 o 1 (dato: $REGOLA_CRASH_LIMITI)"; fi
  [[ "$LIM_ON" == "1" && "$RADICE" != *limiti* ]] \
    && fallito "studio dei limiti acceso con una RADICE senza 'limiti' ($RADICE): radice separata dalle run del seme 1001"
  # K also in the root name (last folder, e.g. .../<root>_limiti/senza_deriva_x4):
  # x<K> with p instead of the dot, not followed by other digits
  [[ "$LIM_MOL" == "1" && ! "$(basename "$RADICE")" =~ x${LIM_K//./p}([^0-9p]|$) ]] \
    && fallito "MOLTIPLICATORE_LIMITI=$LIM_K: l'ultima cartella della RADICE deve contenere x${LIM_K//./p} ($RADICE)"
  if [[ -n "$CONTROLLO_ETA" ]]; then
    grep -q -- '"--controllo-eta"' src/optimization/attack_orchestrator.py || fallito "--controllo-eta assente dall'orchestratore (CONTROLLO_ETA)"
    grep -qF 'CONTROLLO_ETA' scripts/run_campaign.sh || fallito "CONTROLLO_ETA assente da run_campaign.sh"
    [[ "$TIPO" == "baseline" ]] && fallito "CONTROLLO_ETA con la baseline della coda: manca --victim-shadow-env (usare prova_banco_serieB_baseline_definitiva.sh o togliere CONTROLLO_ETA)"
  fi
  if [[ -n "${DDS_PROFILE:-}" ]]; then
    grep -q -- '"--dds-profile"' src/optimization/attack_orchestrator.py || fallito "--dds-profile assente dall'orchestratore (DDS_PROFILE)"
    grep -qF 'DDS_PROFILE' scripts/run_campaign.sh || fallito "DDS_PROFILE assente da run_campaign.sh"
  fi
  if [[ "$MOLA_BINARIO_G1" == "1" ]]; then
    grep -q -- '"--mola-binario-g1"' src/optimization/attack_orchestrator.py || fallito "--mola-binario-g1 assente dall'orchestratore (MOLA_BINARIO_G1)"
    grep -qF 'MOLA_BINARIO_G1' scripts/run_campaign.sh || fallito "MOLA_BINARIO_G1 assente da run_campaign.sh"
  fi
  if [[ "$SONDA" == "1" ]]; then
    for x in scripts/sonda_nuvole.py scripts/sonda_udp.py; do [[ -f "$x" ]] || fallito "$x assente (SONDA)"; done
    grep -qF 'sonda_start' scripts/run_campaign.sh || fallito "avvio della sonda assente da run_campaign.sh (SONDA)"
  fi
  # SERIE_B=1 in run_campaign.sh: MOLA-notices rule, chain check and code 4
  for o in "--mola-se-ne-accorge" "--controllo-catena" "da_rifare.json" "exit 4"; do
    grep -qF -- "$o" scripts/run_campaign.sh || fallito "'$o' assente da run_campaign.sh"
  done
  [[ "$CONTROLLO_RITARDO" != *controllo_ritardo_run.py* || -f scripts/controllo_ritardo_run.py ]] \
    || fallito "scripts/controllo_ritardo_run.py assente (CONTROLLO_RITARDO)"
  # node interpreter as in the reference (run_campaign.sh checks it again with SERIE_B=1)
  python3 scripts/ambiente_nodo.py controlla --riferimento "${AMBIENTE_RIFERIMENTO:-config/ambiente_nodo_serieB.txt}" \
    || fallito "ambiente del perturbation_node diverso dal riferimento (au-env)"
  if [[ "$TIPO" == "baseline" ]]; then
    if [[ -e "$BASE_CSV" || -e "${BASE_CSV%.csv}.json" ]]; then
      echo "baseline gia' presente: $BASE_CSV (non si rifa' e non si sovrascrive)"
      [[ "$A_SECCO" != "1" ]] && exit 21
    else echo "ok: polilinea di uscita libera: $BASE_CSV"; fi
  else
    if [[ -z "$BPATH" ]]; then fallito "BASELINE_SERIEB non impostata: baseline del danno in attesa della decisione baseline-A2"
    elif [[ -f "$BPATH" ]]; then echo "ok: baseline del danno $BPATH"
    else fallito "baseline del danno assente: $BPATH (BASELINE_SERIEB)"; fi
  fi
  if [[ "$TIPO" != "baseline" && -n "$CONTROLLO_RITARDO" ]]; then echo "controllo del ritardo: $CONTROLLO_RITARDO"
  elif [[ "$TIPO" != "baseline" ]]; then echo "attenzione: CONTROLLO_RITARDO vuoto, blocco [4] non controllato in coda"; fi
  if [[ "$A_SECCO" != "1" && "$CONTROLLI_OK" != "1" ]]; then echo "controlli falliti: run non lanciata"; exit 20; fi

  # run_campaign.sh environment (N = 5, D_min 140; FAST_OPS, REPORT_MATCHED,
  # SYNC_WAIT=publisher and SYNC_GUARD=1 come from SERIE_B=1, OPERATOR_MODE from its default).
  local D="$OUT" NODE_X ORCH_X ARM_RC="$BRACCIO" P_RC="$POP" G_RC="$GEN" SUFF_LIM=""
  [[ "$TIPO" != "baseline" ]] && SUFF_LIM="${LIM_SUFF}${G14_SUFF}"   # limits study: empty when off
  NODE_X="--frame-log $D/frame_log.csv --frame-log-delay --fresh-cloud-after-restore"
  ORCH_X="--goal-mode direction --max-windows $FINESTRE --extra-windows $EXTRA_FINESTRE --settle-sec 2.0 --settle-clouds $SETTLE_CLOUDS --reopen-clouds $REOPEN_CLOUDS --fresh-cloud-after-restore --horizon $ORIZZONTE_M${ORZ_TIMEOUT:+ --drive-timeout $ORZ_TIMEOUT} --adapter-events data/attack/verifica/adattatore_eventi.jsonl"
  local CAMP_ENV=(CAMPAIGN_DIR="$D/campagna" ISAAC_ARGS="--headless" ATTACK_FROM=spawn STOP_ON=estimate VICTIM=continuous
                  WINDOW_STOP=ramp STOP_RAMP_K=1.0 MAX_DRIFT_LAG_S=0.5 START_POSE="$START_POSE" SERIE_B=1 SEEDS="$SEME"
                  ICP_QUALITY_LOG=1 TRANSPORT=reliable STRUCTURAL=gross MIN_LATENCY_MS="$MIN_LATENCY_MS"
                  MOLA_DEBUG=1 DANNO_METRO=1   # SERIE_B=1 defaults, explicit: an inherited value does not turn them off
                  RUN_SUFFIX="${SUFF_LIM}${ORZ_SUFF}_coda_N${FINESTRE}_dmin${MIN_LATENCY_MS}" CHAMFER_BUDGET="$BUDGET")
  if [[ "$TIPO" == "baseline" ]]; then
    CAMP_ENV+=(SELECT="$SELECT" TOPK="$TOPK" WARM_START=false)
    ORCH_X="$ORCH_X --pause-sec $PAUSE_SEC"
    P_RC=""; G_RC=""
  else
    NODE_X="$NODE_X --victim-never-clean --shadow-crc-log $D/shadow_crc.csv --measures thread --shared-nn --twin-clock-base"
    ORCH_X="$ORCH_X --victim-never-clean --shadow-check --shadow-crc-log $D/shadow_crc.csv --victim-shadow-env --mola-stop-signal int --damage-ref baseline --baseline-path ${BPATH:-BASELINE_SERIEB_MANCANTE} --twin-init attacker_est --twin-state ombra_g --uniform-sim-time true"
    CAMP_ENV+=(DRIFT_RESET_ON=keep)   # drift kept across metres, explicit: an inherited value does not turn it off
    [[ -n "$SAMPLING" ]] && CAMP_ENV+=(SAMPLING="$SAMPLING")
    case "$TIPO" in
      ricerca)  CAMP_ENV+=(SELECT="$SELECT" TOPK="$TOPK") ;;
      randpuro) CAMP_ENV+=(SELECT=pareto_maxdamage TOPK="$TOPK" RANDOM_FIRST="$K_RANDPURO"); P_RC=1; G_RC=1 ;;
      massimo)  CAMP_ENV+=(SELECT="$SELECT" TOPK="$TOPK" WARM_START=false)
                ORCH_X="$ORCH_X --fixed-genome=$GEN_MAX --admissibility none --max-proc-ms 0"; P_RC=""; G_RC="" ;;
    esac
  fi
  [[ "$TIPO" != "baseline" && "$CONTROLLO_OMBRA_VIVA" == "1" ]] && ORCH_X="$ORCH_X --controllo-ombra-viva"
  [[ "$ORIGINE_COERENTE" == "1" ]] && ORCH_X="$ORCH_X --origine-coerente-vittima"
  [[ "$TIPO" != "baseline" && "$GENOMA_NULLO" == "1" ]] && ORCH_X="$ORCH_X --genoma-nullo-se-nessun-ammissibile"
  if [[ "$TIPO" != "baseline" && "$LIM_MOL" == "1" ]]; then
    ORCH_X="$ORCH_X --moltiplicatore-limiti $LIM_K --famiglie-limiti $LIM_FAM"
    NODE_X="$NODE_X --moltiplicatore-limiti $LIM_K --famiglie-limiti $LIM_FAM"
  fi
  [[ "$TIPO" != "baseline" && "$BLOCCA_G14" == "1" ]] && ORCH_X="$ORCH_X --blocca-gene 14=-1"
  CAMP_ENV+=(NODE_EXTRA="$NODE_X" ORCH_EXTRA="$ORCH_X")
  # chain checks, explicit like the others (values from the queue environment)
  CAMP_ENV+=(CONTROLLO_ETA="$CONTROLLO_ETA" DDS_PROFILE="${DDS_PROFILE:-}" SONDA="$SONDA" MOLA_BINARIO_G1="$MOLA_BINARIO_G1")
  (( ${#D31_ENV[@]} )) && CAMP_ENV+=("${D31_ENV[@]}")
  local CAMP_ARGS=(straight "$ARM_RC" "$GRUPPO" 1 $P_RC $G_RC)
  # options that must appear in the orchestrator command (rep_01/comando.txt)
  local ATTESE=("--goal-mode direction" "--max-windows $FINESTRE" "--extra-windows $EXTRA_FINESTRE" "--sync-wait publisher"
                "--sync-guard" "--select-only-valid" "--perceptibility chamfer" "--fresh-cloud-after-restore" "--seed \"$SEME\"")
  [[ "$TIPO" != "baseline" ]] && ATTESE+=("--incompleti-regola" "--chiudi-attacco-scoperto" "--no-nominal-rollout" "--twin-state ombra_g" "--damage-ref baseline"
                                          "--mola-se-ne-accorge" "--controllo-catena" "--danno-metro")
  [[ -n "$BUDGET" ]] && ATTESE+=("--chamfer-budget $BUDGET")
  [[ "$ORIZZONTE_M" != "1" ]] && ATTESE+=("--horizon $ORIZZONTE_M" "--drive-timeout $ORZ_TIMEOUT")
  [[ "$TIPO" == "randpuro" ]] && ATTESE+=("--random-first-admissible $K_RANDPURO")
  [[ -n "$CONTROLLO_ETA" ]] && ATTESE+=("--controllo-eta $CONTROLLO_ETA")
  [[ "$TIPO" != "baseline" && "$CONTROLLO_OMBRA_VIVA" == "1" ]] && ATTESE+=("--controllo-ombra-viva")
  [[ "$ORIGINE_COERENTE" == "1" ]] && ATTESE+=("--origine-coerente-vittima")
  [[ "$TIPO" != "baseline" && "$GENOMA_NULLO" == "1" ]] && ATTESE+=("--genoma-nullo-se-nessun-ammissibile")
  [[ "$TIPO" != "baseline" && "$LIM_MOL" == "1" ]] && ATTESE+=("--moltiplicatore-limiti $LIM_K" "--famiglie-limiti $LIM_FAM")
  [[ "$TIPO" != "baseline" && "$BLOCCA_G14" == "1" ]] && ATTESE+=("--blocca-gene 14=-1")
  [[ -n "${DDS_PROFILE:-}" ]] && ATTESE+=("--dds-profile ${DDS_PROFILE}")
  [[ "$MOLA_BINARIO_G1" == "1" ]] && ATTESE+=("--mola-binario-g1")
  (( ${#D31_ATTESE[@]} )) && ATTESE+=("${D31_ATTESE[@]}")
  local x; for x in $OPZIONI_ATTESE_EXTRA; do ATTESE+=("$x"); done
  local NODO_ATTESE=("--min-latency-ms $MIN_LATENCY_MS" "--fast-ops" "--frame-log-delay" "--report-matched")
  (( ${#D31_NODO_ATTESE[@]} )) && NODO_ATTESE+=("${D31_NODO_ATTESE[@]}")
  [[ "$TIPO" != "baseline" && "$LIM_MOL" == "1" ]] && NODO_ATTESE+=("--moltiplicatore-limiti $LIM_K" "--famiglie-limiti $LIM_FAM")
  local REP_GLOB="$D/campagna/straight/*_coda_N${FINESTRE}_dmin${MIN_LATENCY_MS}/rep_01"

  if [[ "$A_SECCO" == "1" ]]; then
    echo "+ mkdir -p $D/campagna; prima della run:"
    echo "+ fotografia del codice in $D/fotografia_codice.txt (git rev-parse HEAD, git status --porcelain=v1 -b, sha256sum dei file di src scripts isaac config tests, git diff HEAD)"
    echo "+ scripts/verifica/stato_catena.sh > $D/stato_catena_prima.txt; timeout 30 ros2 daemon stop; timeout 30 ros2 daemon start"
    prova_inizio "serieB_${COND}_s${SEME}" "$TEMPO_MAX_S"
    if [[ "$RIAVVIO_ADATTATORE" == "sempre" ]]; then riavvia_adattatore "$D/adattatore.log"; else riavvio_preventivo "$D/adattatore.log"; fi
    ambiente_su_file "$D/ambiente.txt"
    printf '+ env'; printf ' %q' "${CAMP_ENV[@]}"; printf ' scripts/run_campaign.sh'; printf ' %s' "${CAMP_ARGS[@]}"; printf ' > %s 2>&1\n' "$D/campagna.log"
    echo "+ durante la run: watchdog di Isaac ogni $CONTROLLO_WD_S s; ogni $CONTROLLO_X_S s python3 $COSTRUISCI --controllo-x $REP_GLOB --x-max $X_MAX_M"
    printf '+ appena esiste rep_01/comando.txt, opzioni attese:'; printf ' [%s]' "${ATTESE[@]}"; echo
    printf '+ nel comando del perturbation_node:'; printf ' [%s]' "${NODO_ATTESE[@]}"; echo
    if [[ "$TIPO" == "baseline" ]]; then
      echo "+ python3 $COSTRUISCI $REP_GLOB --csv $BASE_CSV --json $D/baseline_direction_rapporto.json --finestre $FINESTRE --x-max $X_MAX_M --seme $SEME --stato serieB > $D/baseline_direction.txt 2>&1"
    elif [[ -n "$CONTROLLO_RITARDO" ]]; then
      echo "+ timeout $TIMEOUT_RITARDO_S $(comando_ritardo "$D")  > $D/controllo_ritardo.txt 2>&1"
    else
      echo "+ (controllo del ritardo non agganciato)"
    fi
    echo "+ python3 scripts/coda_serieB_stato.py storia $REP_GLOB --guasto-re '$RUN_END_GUASTO' (con da_rifare.json); ombra $REP_GLOB (metro con ombra divergente = guasto catena); classifica (codice 4 di run_campaign.sh = guasto catena) -> $D/esito_run.json"
    [[ "$REGOLA_CRASH_LIMITI" == "1" && "$TIPO" != "baseline" ]] && echo "+ classifica con --regola-crash: vittima senza posa con mola-cli terminato = crash_processo"
    echo "+ python3 scripts/coda_serieB_stato.py riepilogo-run $D --json $D/riepilogo_run.json > $D/riepilogo_run.txt"
    [[ "$CONTROLLI_OK" != "1" ]] && echo "a secco: il lancio vero si fermerebbe ai controlli falliti sopra"
    return 0
  fi

  mkdir -p "$D/campagna"
  echo "$COMANDO_PROVA" > "$D/comando.txt"
  CAMP_PID=""; SEGNALE=""; ESITO_SCRITTO=0; RC_CAMP=""; RC_BASE=""; RC_RIT=""; CONFIG_ERR=""; BLOCCATA=""
  trap 'chiusura_run' EXIT
  trap 'SEGNALE=INT; nota "== run interrotta (INT)"; exit 130' INT
  trap 'SEGNALE=TERM; nota "== run terminata (TERM)"; exit 143' TERM HUP
  prova_inizio "serieB_${COND}_s${SEME}" "$TEMPO_MAX_S" || exit 20
  fotografia_codice "$D/fotografia_codice.txt"
  bash scripts/verifica/stato_catena.sh > "$D/stato_catena_prima.txt" 2>&1
  { timeout 30 ros2 daemon stop; timeout 30 ros2 daemon start; } >> "$D/stato_catena_prima.txt" 2>&1
  if [[ "$RIAVVIO_ADATTATORE" == "sempre" ]]; then riavvia_adattatore "$D/adattatore.log" || BLOCCATA="adattatore non riavviato"
  else riavvio_preventivo "$D/adattatore.log" || BLOCCATA="adattatore non riavviato"; fi
  [[ -n "$BLOCCATA" ]] && { nota "IMPREVISTO: $BLOCCATA: run non lanciata"; exit 1; }
  ambiente_su_file "$D/ambiente.txt"

  nota "run_campaign.sh ${CAMP_ARGS[*]} (CHAMFER_BUDGET=${BUDGET:-nessuno}) -> $D"
  ( trap - INT; exec env "${CAMP_ENV[@]}" scripts/run_campaign.sh "${CAMP_ARGS[@]}" ) > "$D/campagna.log" 2>&1 &
  CAMP_PID=$!
  local t_x=$SECONDS conf_ok=0 nodo_ok=0 rep riga cmdf
  while kill -0 "$CAMP_PID" 2>/dev/null; do
    watchdog_in_prova || exit 1
    if (( conf_ok == 0 )); then
      for cmdf in $REP_GLOB/comando.txt; do
        [[ -f "$cmdf" ]] || continue
        conf_ok=1
        for x in "${ATTESE[@]}"; do grep -qF -- "$x" "$cmdf" || CONFIG_ERR="${CONFIG_ERR:+$CONFIG_ERR; }manca '$x' nel comando dell'orchestratore"; done
      done
    fi
    if (( nodo_ok == 0 )); then
      riga="$(pgrep -af "^python3( -u)? .*perturbation_node\.py" | head -1)"
      if [[ -n "$riga" ]]; then
        nodo_ok=1
        for x in "${NODO_ATTESE[@]}"; do [[ " $riga " == *" $x "* ]] || CONFIG_ERR="${CONFIG_ERR:+$CONFIG_ERR; }manca '$x' nel comando del nodo"; done
      fi
    fi
    if [[ -n "$CONFIG_ERR" ]]; then
      MOTIVO_INTERRUZIONE="configurazione: $CONFIG_ERR"
      nota "IMPREVISTO: $MOTIVO_INTERRUZIONE: la run si chiude"
      exit 1
    fi
    if (( SECONDS - t_x >= CONTROLLO_X_S )); then
      t_x=$SECONDS
      for rep in $REP_GLOB; do
        [[ -d "$rep" ]] || continue
        if ! riga="$(python3 "$COSTRUISCI" --controllo-x "$rep" --x-max "$X_MAX_M")"; then
          MOTIVO_INTERRUZIONE="x odom vera oltre $X_MAX_M m ($riga)"
          nota "IMPREVISTO: $MOTIVO_INTERRUZIONE: la run si chiude"
          exit 1
        fi
      done
    fi
    sleep 5
  done
  wait "$CAMP_PID"; RC_CAMP=$?; CAMP_PID=""
  nota "run_campaign.sh uscito con $RC_CAMP"
  inizio_chiusura   # from here the maximum time no longer changes the outcome
  rep="$(ls -d $REP_GLOB 2>/dev/null | head -1)"
  if [[ "$TIPO" == "baseline" ]]; then
    if [[ -n "$rep" ]]; then
      python3 "$COSTRUISCI" "$rep" --csv "$BASE_CSV" --json "$D/baseline_direction_rapporto.json" --finestre "$FINESTRE" \
        --x-max "$X_MAX_M" --seme "$SEME" --stato serieB > "$D/baseline_direction.txt" 2>&1
      RC_BASE=$?
      [[ $RC_BASE -eq 0 ]] && cp -n "$D/baseline_direction_rapporto.json" "${BASE_CSV%.csv}.json"
    else RC_BASE=99; fi
  elif [[ -n "$CONTROLLO_RITARDO" && "$RC_CAMP" == "0" ]]; then
    timeout "$TIMEOUT_RITARDO_S" bash -c "$(comando_ritardo "$D")" > "$D/controllo_ritardo.txt" 2>&1
    RC_RIT=$?
    nota "controllo del ritardo: codice $RC_RIT ($D/controllo_ritardo.txt)"
  fi
  exit 0
}

# Delay check command for folder $1.
comando_ritardo() {
  local c="$CONTROLLO_RITARDO"
  if [[ "$c" == *"{run}"* || "$c" == *"{frame_log}"* ]]; then
    c="${c//\{run\}/$1}"; c="${c//\{frame_log\}/$1/frame_log.csv}"
  else c="$c $1"; fi
  echo "$c"
}

# Code snapshot next to the run: commit, git status, sha256 of the source files, git diff.
fotografia_codice() {   # $1 = file
  local files
  if git -C "$ROOT" rev-parse HEAD >/dev/null 2>&1; then
    files=$(git -C "$ROOT" ls-files -c -m -o --exclude-standard -- src scripts isaac config tests 2>/dev/null | grep -v __pycache__ | sort -u)
  else
    files=$(find src scripts isaac config tests -type f ! -path '*__pycache__*' 2>/dev/null | sort)
  fi
  { echo "data: $(date +%FT%T)"; echo "radice: $ROOT"
    echo "== commit"; git -C "$ROOT" rev-parse HEAD 2>&1
    echo "== git status --porcelain=v1 -b"; git -C "$ROOT" status --porcelain=v1 -b 2>&1
    echo "== sha256 (src scripts isaac config tests, tracciati, modificati e non tracciati)"
    for f in $files; do [[ -f "$f" ]] && sha256sum "$f"; done
    echo "== git diff HEAD"; git -C "$ROOT" diff HEAD -- src scripts isaac config tests 2>&1
  } > "$1"
}

# Run closing: campaign, MOLA of the run, esito_run.json, then prova_fine (lib).
chiusura_run() {
  local rc=$?
  [[ "$PROVA_MIA" == "1" ]] || return 0
  trap '' INT TERM HUP
  inizio_chiusura
  if [[ -n "$CAMP_PID" ]]; then
    [[ "$SEGNALE" == "TERM" || -n "$MOTIVO_INTERRUZIONE" ]] && kill -TERM "$CAMP_PID" 2>/dev/null
    attendi_poi_term "$CAMP_PID" 130 || nota "IMPREVISTO: run_campaign.sh (pid $CAMP_PID) ancora vivo dopo la chiusura"
    CAMP_PID=""
  fi
  chiudi_mola_della_prova
  scrivi_esito
  # victim rejections and pose losses, [4], end of run, Chamfer and damage (daily summary)
  $PY riepilogo-run "$OUT" --json "$OUT/riepilogo_run.json" > "$OUT/riepilogo_run.txt" 2>&1 \
    || nota "riepilogo della run non riuscito ($OUT/riepilogo_run.txt)"
  prova_fine "$rc"
}

scrivi_esito() {
  local rep wd fatti="$OUT/esito_fatti.json"
  rep="$(ls -d "$OUT"/campagna/straight/*/rep_01 2>/dev/null | head -1)"
  wd="$(conta_interventi_watchdog)"; wd=$(( ${wd:-0} - ${PROVA_WD0:-0} ))
  # values passed via environment: no run string inside the Python code
  F_CARTELLA="$OUT" F_COND="$COND" F_SEME="$SEME" F_TIPO="$TIPO" F_RC="${RC_CAMP:-}" F_REP="${rep:-}" \
  F_STORIA="$($PY storia "${rep:-}" --guasto-re "$RUN_END_GUASTO" 2>/dev/null || echo '{}')" F_WD="$wd" \
  F_SEGNALE="$SEGNALE" F_MOTIVO="$MOTIVO_INTERRUZIONE" F_CONFIG="$CONFIG_ERR" F_BLOCCATA="$BLOCCATA" \
  F_ADATT="$(pid_adattatore)" F_RC_BASE="${RC_BASE:-}" F_RC_RIT="${RC_RIT:-}" \
  F_OMBRA="$($PY ombra "${rep:-}" 2>/dev/null || echo '{}')" \
  python3 -I - "$fatti" <<'EOF'
import json, os, sys
e = os.environ
intero = lambda v: int(v) if v not in ("", None) else None
rep = e["F_REP"]
try:
    storia = json.loads(e["F_STORIA"] or "{}")
except ValueError:
    storia = {"errore": "storia non leggibile"}
try:
    ombra = json.loads(e["F_OMBRA"] or "{}")
except ValueError:
    ombra = {"esito": "non_misurata", "testo": "ombra: uscita del controllo non leggibile"}
f = {"cartella": e["F_CARTELLA"], "condizione": e["F_COND"], "seme": e["F_SEME"],
     "baseline": e["F_TIPO"] == "baseline", "rc": intero(e["F_RC"]), "rep": rep,
     "completata": bool(rep) and os.path.exists(os.path.join(rep, "completata")),
     "storia": storia, "watchdog_interventi": intero(e["F_WD"]) or 0,
     "tempo_scaduto": os.path.exists(os.path.join(e["F_CARTELLA"], ".tempo_massimo_scaduto")),
     "segnale": e["F_SEGNALE"], "motivo_interruzione": e["F_MOTIVO"],
     "configurazione": e["F_CONFIG"], "bloccata": e["F_BLOCCATA"],
     "adattatore_vivo": bool(e["F_ADATT"]),
     "baseline_rc": intero(e["F_RC_BASE"]), "ritardo_rc": intero(e["F_RC_RIT"]), "ombra": ombra}
with open(sys.argv[1], "w") as fh:
    json.dump(f, fh, indent=1, ensure_ascii=False)
EOF
  local RCR=(); [[ "$REGOLA_CRASH_LIMITI" == "1" && "$TIPO" != "baseline" ]] && RCR=(--regola-crash)
  nota "esito: $($PY classifica "$fatti" --uscita "$OUT/esito_run.json" ${RCR[@]+"${RCR[@]}"})"
}

# ======================================================================
# Isaac watchdog: active before every run; started by the queue if WATCHDOG_FASE3 is set.
# ======================================================================
assicura_watchdog() {
  local st; st="$(watchdog_stato)"
  [[ "$st" == "ok" && -e "$SENTINELLA_FASE3" ]] && return 0
  if [[ -z "$WATCHDOG_FASE3" || ! -f "$WATCHDOG_FASE3" ]]; then
    nota "watchdog di Isaac non attivo ($st) e WATCHDOG_FASE3 non dato: la coda si ferma"; return 1
  fi
  if [[ "$A_SECCO" == "1" ]]; then
    echo "+ touch $SENTINELLA_FASE3; tmux new-window -d -t tesi -n wdfase3 'ROOT=$ROOT bash $WATCHDOG_FASE3'; attesa del battito (90 s)"; return 0
  fi
  touch "$SENTINELLA_FASE3"
  tmux has-session -t tesi 2>/dev/null || tmux new-session -d -s tesi
  tmux new-window -d -t tesi -n wdfase3 "ROOT=$ROOT bash $WATCHDOG_FASE3"
  local t0=$SECONDS
  while (( SECONDS - t0 < 90 )); do
    [[ "$(watchdog_stato)" == "ok" ]] && { nota "watchdog di Isaac avviato dalla coda"; return 0; }
    sleep 5
  done
  nota "IMPREVISTO: watchdog di Isaac avviato ma senza battito dopo 90 s: la coda si ferma"; return 1
}

# free GB on the repository disk and --lista estimate for the remaining runs
RIEP_DISCO=(--disco "$ROOT" --pilota "$PILOTA_DISCO" --baseline-dir "$BASELINE_DISCO" --finestre "$FINESTRE"
            --extra "$EXTRA_FINESTRE" --pop "$POP" --gen "$GEN" --topk "$TOPK" --k-max "$K_RANDPURO")
riepilogo() { $PY riepilogo "${RIEP_DISCO[@]}" ${DISCO_MIN_GB:+--margine-gb "$DISCO_MIN_GB"} "$STATO_FILE" "$RIEPILOGHI" "${RUNS[@]}" >/dev/null 2>&1 || true; }

# ======================================================================
# Modes
# ======================================================================
if [[ "$MODO" == "run" ]]; then
  esegui_run "${ESEGUI[@]}"
  exit $?
fi

if [[ "$MODO" == "lista" ]]; then
  echo "Condizioni della serie B (D5; preregistrazione v2 e v3). Semi: $SEMI. Finestre $FINESTRE (+$EXTRA_FINESTRE trattenute)."
  printf '%-20s %-7s %-7s %s\n' id braccio budget descrizione
  $PY condizioni | awk -F'\t' '{printf "%-20s %-7s %-7s %s\n", $1, $2, $3, $5}'
  echo
  echo "Stima per un seme (ore da T6, disco dal pilota G3):"
  mapfile -t UNO < <($PY seleziona --semi "${SEMI%% *}" tutte)
  stima "${UNO[@]}"
  exit 0
fi

(( ${#VOCI[@]} )) || [[ "$MODO" == "stato" ]] || { echo "nessuna condizione: scripts/coda_serieB.sh --lista per l'elenco"; exit 2; }
(( ${#VOCI[@]} )) || VOCI=(tutte)
mapfile -t RUNS < <($PY seleziona --semi "$SEMI" "${VOCI[@]}") || true
(( ${#RUNS[@]} )) || { $PY seleziona --semi "$SEMI" "${VOCI[@]}"; exit 2; }

if [[ "$MODO" == "stato" ]]; then
  echo "== stato della coda ($STATO_FILE)"
  $PY mostra "$STATO_FILE" "${RUNS[@]}"
  echo; $PY riepilogo --stampa "${RIEP_DISCO[@]}" "$STATO_FILE" "$RIEPILOGHI" "${RUNS[@]}"
  exit 0
fi

# Library functions for watchdog, log and queue notes (OUT = queue folder).
OUT="$CODA_DIR"
# shellcheck source=prova_banco_serieB_lib.sh
source "$HERE/prova_banco_serieB_lib.sh"

echo "== coda della serie B $( [[ $A_SECCO == 1 ]] && echo '(a secco: nessun lancio, nessuno stato scritto)') - $(date +%F' '%T)"
echo "radice: $RADICE; semi: $SEMI; run scelte: ${#RUNS[@]}"
echo "-- stato delle run scelte"
$PY mostra "$STATO_FILE" "${RUNS[@]}"
mapfile -t DA_FARE < <(for r in "${RUNS[@]}"; do $PY prossima "$STATO_FILE" "$r"; done)
# Damage baseline: mandatory if any of the runs to do is attacked
BASE_ERR=""
for r in "${DA_FARE[@]}"; do
  [[ "${r#*:}" == "baseline" ]] && continue
  if [[ -z "$BASELINE_PATH" ]]; then
    BASE_ERR="BASELINE_SERIEB non impostata: la baseline media dei 3 semi attende la decisione baseline-A2"; break
  fi
  f="${BASELINE_PATH//\{seme\}/${r%%:*}}"
  [[ -f "$f" ]] || { BASE_ERR="baseline del danno assente: $f (BASELINE_SERIEB, run ${r#*:} s${r%%:*})"; break; }
done
if [[ -n "$BASE_ERR" ]]; then echo "[!] $BASE_ERR: la coda si ferma prima della prima run"
else echo "baseline del danno: ${BASELINE_PATH:-non serve (solo baseline)}"; fi
echo "-- stima di ore e disco per le ${#DA_FARE[@]} run ancora da fare"
(( ${#DA_FARE[@]} )) && stima "${DA_FARE[@]}"
LIBERO=$(df --output=avail -BG "$ROOT" | tail -1 | tr -dc 0-9)
if (( ${#DA_FARE[@]} )); then
  read -r GB_STIMA GB_MAX < <(stima --json "${DA_FARE[@]}" | python3 -c 'import json,sys; t=json.load(sys.stdin)["totale"]["mb"]; print(round(t[0]/1000), round(t[1]/1000))')
  echo "disco: ${LIBERO} GB liberi; servono circa ${GB_STIMA} GB (al piu' ${GB_MAX} GB) piu' ${DISCO_MIN_GB} GB di margine"
  (( LIBERO < GB_STIMA + DISCO_MIN_GB )) && echo "ATTENZIONE: lo spazio non basta per tutta la selezione: la coda si fermera' prima della run che non ci sta"
fi

if [[ "$A_SECCO" == "1" ]]; then
  assicura_watchdog || true
  i=0
  for r in "${DA_FARE[@]}"; do
    i=$((i + 1)); SEME="${r%%:*}"; COND="${r#*:}"
    CART="$($PY cartella --radice "$RADICE" "$COND" "$SEME")"
    echo; echo "######## run $i/${#DA_FARE[@]}: seme $SEME, $COND"
    echo "+ controllo del disco prima della run: GB liberi >= GB massimi della run + $DISCO_MIN_GB"
    echo "+ $PY inizia $STATO_FILE $COND $SEME $CART"
    echo "+ setsid bash $SE --esegui-run $COND $SEME $CART >> $LOG_CODA 2>&1"
    bash "$SE" --a-secco --esegui-run "$COND" "$SEME" "$CART" 2>&1 | sed 's/^/    /'
    echo "+ $PY chiudi $STATO_FILE $COND $SEME $CART/esito_run.json; riepilogo in $RIEPILOGHI/riepilogo_$(date +%F).txt"
    # folder of the next repetition: CART may already be a repetition (_rip<n>)
    if [[ "$CART" =~ ^(.*)_rip([0-9]+)$ ]]; then CB="${BASH_REMATCH[1]}"; NR=$(( BASH_REMATCH[2] + 1 )); else CB="$CART"; NR=1; fi
    echo "+ guasto_catena o oltre_ritardo (rit4): stessa run, seme $SEME, in ${CB}_rip${NR}; al piu' 2 ripetizioni per run, guasti precedenti compresi"
    echo "+ ritardo_configurazione, ritardo_non_misurabile o ritardo_errore (controllo del ritardo 4, 5 o altro): voce [!] e la coda si ferma"
    [[ "$REGOLA_CRASH_LIMITI" == "1" ]] && echo "+ crash_processo (REGOLA_CRASH_LIMITI=1): stessa run in ${CB}_rip${NR} una volta; crash di nuovo con lo stesso genoma = crash_da_attacco (finale), con un altro genoma = guasto_catena"
  done
  exit 0
fi

# --- Actual launch ---
mkdir -p "$CODA_DIR" "$RIEPILOGHI"
exec 8>"$CODA_DIR/coda.lock"
flock -n 8 || { echo "un'altra coda della serie B e' gia' in corso ($CODA_DIR/coda.lock)"; exit 1; }
while IFS= read -r r; do [[ -n "$r" ]] && nota "run rimasta in corso da una coda caduta: $r (sara' rifatta in una cartella nuova)"; done < <($PY ripara "$STATO_FILE")
if [[ -n "$BASE_ERR" ]]; then
  nota "[!] $BASE_ERR: la coda si ferma prima della prima run"
  vault_riga "- $(date +%H:%M) [!] coda della serie B non avviata: $BASE_ERR" ""
  exit 1
fi
nota "== coda avviata: ${#RUNS[@]} run scelte (${#DA_FARE[@]} da fare), semi $SEMI, condizioni ${VOCI[*]}"
vault_riga "- $(date +%H:%M) coda della serie B avviata: ${#DA_FARE[@]} run da fare (${VOCI[*]}; semi $SEMI), stato in \`${STATO_FILE#$ROOT/}\`" ""
riepilogo

FIGLIO=""; FERMA=""
trap '[[ -n "$FIGLIO" ]] && kill -INT -- "-$FIGLIO" 2>/dev/null; FERMA=INT' INT
trap '[[ -n "$FIGLIO" ]] && kill -TERM "$FIGLIO" 2>/dev/null; FERMA=TERM' TERM HUP
RC_CODA=0
while :; do
  [[ -n "$FERMA" ]] && { nota "coda fermata a mano ($FERMA)"; RC_CODA=130; break; }
  P="$($PY prossima "$STATO_FILE" "${RUNS[@]}")"
  [[ -z "$P" ]] && { nota "== coda finita: nessuna run da fare"; break; }
  SEME="${P%%:*}"; COND="${P#*:}"
  assicura_watchdog || { RC_CODA=1; break; }
  LIBERO=$(df --output=avail -BG "$ROOT" | tail -1 | tr -dc 0-9)
  GB_RUN=$(stima --json "$P" | python3 -c 'import json,sys; print(int(json.load(sys.stdin)["totale"]["mb"][1]/1000 + 1))')
  if (( LIBERO < GB_RUN + DISCO_MIN_GB )); then
    nota "IMPREVISTO: disco: ${LIBERO} GB liberi, la run $COND s$SEME ne puo' usare fino a ${GB_RUN} (+${DISCO_MIN_GB} di margine): la coda si ferma"
    RC_CODA=1; break
  fi
  CART="$($PY cartella --radice "$RADICE" "$COND" "$SEME")"
  $PY inizia "$STATO_FILE" "$COND" "$SEME" "$CART"
  riepilogo
  nota "run $COND, seme $SEME -> ${CART#$ROOT/}"
  ( trap - INT; exec setsid bash "$SE" --esegui-run "$COND" "$SEME" "$CART" ) >> "$LOG_CODA" 2>&1 &
  FIGLIO=$!
  while :; do wait "$FIGLIO"; RC_F=$?; kill -0 "$FIGLIO" 2>/dev/null || break; done
  FIGLIO=""
  ES="$CART/esito_run.json"
  if [[ ! -f "$ES" ]]; then
    mkdir -p "$CODA_DIR/esiti"; ES="$CODA_DIR/esiti/$(date +%Y%m%d_%H%M%S)_${COND}_s${SEME}.json"
    case "$RC_F" in
      21) echo "{\"esito\": \"saltata\", \"motivo\": \"polilinea della baseline gia' presente\", \"cartella\": null}" > "$ES" ;;
      20) echo "{\"esito\": \"bloccata\", \"motivo\": \"controlli prima della run falliti (coda.log)\", \"cartella\": null}" > "$ES" ;;
      *)  if [[ -e "$CART/.tempo_massimo_scaduto" ]]; then M='{"esito": "guasto_catena", "motivo": "tempo massimo, run chiusa a forza senza esito"'
          else M='{"esito": "bloccata", "motivo": "run chiusa senza esito (codice '"$RC_F"', eventi.txt della run)"'; fi
          echo "$M, \"cartella\": \"$CART\"}" > "$ES" ;;
    esac
  fi
  NUOVO="$($PY chiudi "$STATO_FILE" "$COND" "$SEME" "$ES")"
  ESITO="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["esito"] + ": " + d.get("motivo", ""))' "$ES")"
  nota "run $COND s$SEME: $ESITO; stato della run: $NUOVO"
  riepilogo
  case "${ESITO%%:*}" in
    interrotta|bloccata|configurazione|disco)
      vault_riga "- $(date +%H:%M) coda della serie B FERMA: $COND s$SEME, $ESITO (\`${CART#$ROOT/}\`)" ""
      RC_CODA=1; break ;;
    ritardo_configurazione|ritardo_non_misurabile|ritardo_errore)
      vault_riga "- $(date +%H:%M) [!] coda della serie B FERMA sul blocco [4]: $COND s$SEME, $ESITO (\`${CART#$ROOT/}/controllo_ritardo.txt\`)" ""
      RC_CODA=1; break ;;
  esac
  [[ "$NUOVO" == "guasto_definitivo" ]] && vault_riga "- $(date +%H:%M) coda della serie B: $COND s$SEME guasto dopo 3 tentativi, si passa oltre" ""
  [[ -n "$FERMA" ]] && continue
  sleep "$PAUSA_FRA_RUN_S"
done
riepilogo
nota "riepilogo: $RIEPILOGHI/riepilogo_$(date +%F).txt"
vault_riga "- $(date +%H:%M) coda della serie B chiusa (codice $RC_CODA): riepilogo in \`${RIEPILOGHI#$ROOT/}/riepilogo_$(date +%F).txt\`" ""
exit "$RC_CODA"
