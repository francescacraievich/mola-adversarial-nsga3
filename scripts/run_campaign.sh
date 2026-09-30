#!/usr/bin/env bash
# Ripetizioni di un braccio della campagna su uno scenario.
#
# Isaac Sim viene avviato da terminale (isaac/run_isaac_standalone.py, opzioni
# extra in ISAAC_ARGS, es. --headless) all'inizio di ogni ripetizione e chiuso
# alla fine: ogni processo vive una sola run. Se l'orchestratore si ferma per
# stallo di Isaac (codice 3) la ripetizione viene rilanciata una volta; la
# cartella della prova fallita resta come rep_NN_stallo1 (rep_NN_stallo2 se
# fallisce anche la seconda, e lo script passa alla ripetizione successiva).
#
# Prerequisiti in esecuzione: add_intensity_node, perturbation_node con
# --min-latency-ms $MIN_LATENCY_MS (default 85; piu' --gaussian-sigma per il
# braccio gaussiano): tutti i bracci pagano la stessa latenza per scan, e lo
# script si rifiuta di partire se il nodo in esecuzione ha un valore diverso.
# Ogni ripetizione riporta il
# robot alla posa iniziale (--start-pose) e riparte da zero; una ripetizione
# gia' presente (history.json) viene saltata, cosi' lo script si puo'
# rilanciare dopo un'interruzione.
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
#   scripts/run_campaign.sh rect none 3 10
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
ISAAC_ARGS="${ISAAC_ARGS:-}"               # opzioni extra per run_isaac_standalone.py
ISAAC_READY_SEC="${ISAAC_READY_SEC:-300}"  # attesa massima dell'avvio di Isaac
# CAMPAIGN_DIR: radice dei risultati (default data/attack/campaign)

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

ISAAC_PID=""

isaac_start() {   # $1 = file di log
  python3 -u isaac/run_isaac_standalone.py $ISAAC_ARGS > "$1" 2>&1 &
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

isaac_stop() {
  [[ -z "$ISAAC_PID" ]] && return 0
  if kill -0 "$ISAAC_PID" 2>/dev/null; then
    kill -INT "$ISAAC_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$ISAAC_PID" 2>/dev/null || break; sleep 1; done
    kill -TERM "$ISAAC_PID" 2>/dev/null || true
    for _ in $(seq 1 15); do kill -0 "$ISAAC_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL "$ISAAC_PID" 2>/dev/null || true
  fi
  wait "$ISAAC_PID" 2>/dev/null || true
  ISAAC_PID=""
}
trap isaac_stop EXIT

# Latenza del nodo in esecuzione, letta dallo stato che pubblica (il nodo
# pubblica lo stato solo dopo le prime nuvole: si controlla con Isaac avviato).
check_latency() {
  local lat
  lat="$(timeout 10 ros2 topic echo --once --field data /attack/status std_msgs/msg/String 2>/dev/null \
    | python3 -c 'import json,sys; print(json.loads(sys.stdin.readline()).get("min_latency_ms", 0))' 2>/dev/null || true)"
  if ! python3 -c "import sys; sys.exit(0 if abs(float('${lat:-nan}') - $MIN_LATENCY_MS) < 1e-6 else 1)" 2>/dev/null; then
    echo "perturbation_node senza la latenza minima richiesta (letta: '${lat:-nessuno stato}', attesa: $MIN_LATENCY_MS ms)."
    echo "Rilanciarlo con: python3 -u src/nodes/perturbation_node.py --min-latency-ms $MIN_LATENCY_MS"
    return 1
  fi
}

OUT_BASE="${CAMPAIGN_DIR:-data/attack/campaign}/$SCENARIO/${ARM}_g${GROUP}"
mkdir -p "$OUT_BASE"

for i in $(seq 1 "$REPS"); do
  NAME="rep_$(printf '%02d' "$i")"
  OUT="$OUT_BASE/$NAME"
  if [[ -f "$OUT/history.json" ]]; then
    echo "== $OUT gia' presente, salto"
    continue
  fi
  for attempt in 1 2; do
    echo "== $SCENARIO / $ARM / gruppo $GROUP / ripetizione $i (tentativo $attempt)  ->  $OUT"
    isaac_start "$OUT_BASE/${NAME}_isaac.log" || exit 1
    check_latency || exit 1
    set +e
    python3 -u src/optimization/attack_orchestrator.py \
      --goals "$GOALS" --start-pose="$START_POSE" --horizon 1.0 --settle-sec 2.0 \
      --seed "$((1000 + i))" --trace --reeval --out "$OUT" $EXTRA \
      2>&1 | tee "$OUT_BASE/${NAME}.log"
    rc=${PIPESTATUS[0]}
    set -e
    isaac_stop
    [[ $rc -ne 3 ]] && break
    # Stallo di Isaac: la prova resta, rinominata, e la ripetizione riparte.
    echo "== stallo di Isaac nella ripetizione $i (tentativo $attempt)"
    TAG="${NAME}_stallo${attempt}"
    [[ -e "$OUT_BASE/$TAG" ]] && TAG="${TAG}_$(date +%Y%m%d_%H%M%S)"   # da un lancio precedente
    mv "$OUT" "$OUT_BASE/$TAG"
    mv "$OUT_BASE/${NAME}.log" "$OUT_BASE/${TAG}.log"
    mv "$OUT_BASE/${NAME}_isaac.log" "$OUT_BASE/${TAG}_isaac.log"
  done
done
echo "== campagna $SCENARIO / $ARM / gruppo $GROUP completata"
