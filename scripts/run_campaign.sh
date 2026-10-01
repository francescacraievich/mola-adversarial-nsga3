#!/usr/bin/env bash
# Ripetizioni di un braccio della campagna su uno scenario.
#
# A ogni ripetizione lo script avvia Isaac Sim da terminale
# (isaac/run_isaac_standalone.py, opzioni extra in ISAAC_ARGS, es. --headless)
# e il perturbation_node con --min-latency-ms $MIN_LATENCY_MS (default 85; per
# il braccio gaussiano anche --gaussian-sigma), lancia l'orchestratore e alla
# fine chiude orchestratore, nodo e Isaac: ogni processo vive una sola run e
# tutti i bracci pagano la stessa latenza per scan. L'add_intensity_node resta
# un prerequisito in esecuzione.
#
# Il braccio gaussiano gira per ogni valore di SIGMAS (metri, default
# "0.01 0.02 0.05 0.10"), una sottocartella per valore: gaussian_s<sigma>.
# Cartelle: none_g<gruppo>, gaussian_s<sigma>, <arm>_g<gruppo>_p<pop>x<gen>
# per random e nsga3 (configurazioni diverse non si mescolano).
#
# Una ripetizione e' completa quando l'orchestratore esce con codice 0: lo
# script scrive allora il file "completata" nella sua cartella, e le
# ripetizioni con quel file vengono saltate (lo script si puo' rilanciare dopo
# un'interruzione). Una cartella senza "completata" (run interrotta: la
# history.json viene scritta dopo ogni finestra, quindi puo' esistere anche
# parziale) viene rinominata rep_NN_incompleta_<data> e la ripetizione rifatta.
# Se l'orchestratore si ferma per stallo di Isaac (codice 3) la ripetizione
# viene rilanciata una volta; la prova fallita resta come rep_NN_stallo1
# (rep_NN_stallo2 se fallisce anche la seconda, e si passa oltre).
#
# Uso:
#   scripts/run_campaign.sh <scenario> <arm> <gruppo> <ripetizioni> [pop] [gen]
#
#   scenario: straight | L | rect   (waypoint definiti sotto)
#   arm:      none | gaussian | random | nsga3
#   gruppo:   1 | 2 | 3
#
# Esempi:
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
START_POSE="${START_POSE:--5.99,-1.0,0}"   # posa iniziale assoluta x,y,yaw_deg
MIN_LATENCY_MS="${MIN_LATENCY_MS:-85}"     # latenza minima per scan del perturbation_node
SIGMAS="${SIGMAS:-0.01 0.02 0.05 0.10}"    # deviazioni standard del braccio gaussiano (m)
ISAAC_ARGS="${ISAAC_ARGS:-}"               # opzioni extra per run_isaac_standalone.py
ISAAC_READY_SEC="${ISAAC_READY_SEC:-300}"  # attesa massima dell'avvio di Isaac
CAMPAIGN_DIR="${CAMPAIGN_DIR:-data/attack/campaign}"   # radice dei risultati

case "$SCENARIO" in
  straight) GOALS="5,0" ;;
  L)        GOALS="5,0; 5,2.5; 10,2.5" ;;
  rect)     GOALS="4,0; 4,2.5; 0,2.5; 0,0" ;;
  *) echo "scenario sconosciuto: $SCENARIO"; exit 1 ;;
esac

case "$ARM" in
  none)     EXTRA="--passthrough" ;;
  gaussian) EXTRA="--search gaussian" ;;
  random)   EXTRA="--search random --pop $POP --gen $GEN --genome-group $GROUP" ;;
  nsga3)    EXTRA="--search nsga3 --pop $POP --gen $GEN --genome-group $GROUP" ;;
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

ISAAC_PID=""
NODE_PID=""
ORCH_PID=""

# In uno script i comandi lanciati con & partono con SIGINT ignorato: senza
# "trap - INT" l'INT di stop_pid non arriverebbe e i processi verrebbero
# chiusi solo con TERM, senza la loro chiusura ordinata.

# Termina un processo: prima il segnale dato, poi TERM, poi KILL.
stop_pid() {   # $1 = pid, $2 = primo segnale, $3 = secondi di attesa
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

isaac_start() {   # $1 = file di log
  ( trap - INT; exec python3 -u isaac/run_isaac_standalone.py $ISAAC_ARGS ) > "$1" 2>&1 &
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
  sleep 5   # prime nuvole e prima posa
}

isaac_stop() { stop_pid "$ISAAC_PID" INT 30; ISAAC_PID=""; }

# Il nodo parte ad attacco spento: e' l'orchestratore ad accenderlo per i
# rollout che lo richiedono.
node_start() {   # $1 = file di log, $2 = sigma del rumore gaussiano (vuoto = genoma)
  local extra=""
  [[ -n "$2" ]] && extra="--gaussian-sigma $2"
  ( trap - INT; exec python3 -u src/nodes/perturbation_node.py --min-latency-ms "$MIN_LATENCY_MS" \
    --start-disabled $extra ) > "$1" 2>&1 &
  NODE_PID=$!
}

node_stop() { stop_pid "$NODE_PID" INT 10; NODE_PID=""; }

cleanup() {
  stop_pid "$ORCH_PID" INT 30; ORCH_PID=""
  node_stop
  isaac_stop
}
trap cleanup EXIT
trap 'echo "== interrotto"; exit 130' INT
trap 'echo "== terminato"; exit 143' TERM HUP

# Latenza del nodo appena avviato, letta dallo stato che pubblica (lo stato
# arriva solo dopo le prime nuvole): fino a 60 s di attesa.
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

# Sposta cartella e log di una ripetizione sotto un nuovo nome.
move_rep() {   # $1 = base, $2 = nome, $3 = nuovo nome
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
    none)          DIR="none_g${GROUP}" ;;
    gaussian)      DIR="gaussian_s${SIGMA}" ;;
    random|nsga3)  DIR="${ARM}_g${GROUP}_p${POP}x${GEN}" ;;
  esac
  [[ "$SIGMA" == "-" ]] && SIGMA=""
  OUT_BASE="$CAMPAIGN_DIR/$SCENARIO/$DIR"
  mkdir -p "$OUT_BASE"

  for i in $(seq 1 "$REPS"); do
    NAME="rep_$(printf '%02d' "$i")"
    OUT="$OUT_BASE/$NAME"
    if [[ -f "$OUT/completata" ]]; then
      echo "== $OUT gia' completata, salto"
      continue
    fi
    if [[ -e "$OUT" || -e "$OUT_BASE/$NAME.log" ]]; then
      echo "== $OUT incompleta da un lancio precedente: rinominata e rifatta"
      move_rep "$OUT_BASE" "$NAME" "${NAME}_incompleta_$(date +%Y%m%d_%H%M%S)"
    fi
    for attempt in 1 2; do
      echo "== $(date +%T) $SCENARIO / $DIR / ripetizione $i (tentativo $attempt)  ->  $OUT"
      isaac_start "$OUT_BASE/${NAME}_isaac.log" || exit 1
      node_start "$OUT_BASE/${NAME}_pert.log" "$SIGMA"
      check_latency || exit 1
      set +e
      ( trap - INT; exec python3 -u src/optimization/attack_orchestrator.py \
        --goals "$GOALS" --start-pose="$START_POSE" --horizon 1.0 --settle-sec 2.0 \
        --seed "$((1000 + i))" --trace --reeval --out "$OUT" $EXTRA ) \
        > >(tee "$OUT_BASE/${NAME}.log") 2>&1 &
      ORCH_PID=$!
      wait "$ORCH_PID"
      rc=$?
      ORCH_PID=""
      set -e
      node_stop
      isaac_stop
      if [[ $rc -eq 0 ]]; then
        date +%FT%T > "$OUT/completata"
        break
      fi
      [[ $rc -ne 3 ]] && { echo "== orchestratore uscito con codice $rc: ripetizione $i non completata"; break; }
      # Stallo di Isaac: la prova resta, rinominata, e la ripetizione riparte.
      echo "== stallo di Isaac nella ripetizione $i (tentativo $attempt)"
      move_rep "$OUT_BASE" "$NAME" "${NAME}_stallo${attempt}"
    done
  done
done
echo "== campagna $SCENARIO / $ARM / gruppo $GROUP completata"
