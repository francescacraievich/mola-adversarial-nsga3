#!/usr/bin/env bash
# Sorveglianza della campagna notturna sul rettilineo.
#
# Ogni PERIOD secondi (default 600): conta le ripetizioni completate per
# braccio (cartelle rep_NN con il file "completata" scritto da run_campaign.sh;
# la history.json da sola non basta, perche' viene scritta anche a meta' run)
# e scrive una riga in data/attack/verifica/28_notte_watchdog.txt. Se nessun
# run_campaign.sh e' vivo e i conteggi attesi non sono raggiunti, rilancia la
# sequenza in una nuova finestra tmux: le ripetizioni completate vengono
# saltate e quelle interrotte rinominate (rep_NN_incompleta_<data>) e rifatte
# da run_campaign.sh. Quando tutti i conteggi sono raggiunti scrive "campagna
# completa" ed esce. Al piu' MAX_RELAUNCH rilanci.
#
# Uso (in una finestra tmux della sessione tesi):
#   scripts/watchdog_notte.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$HERE/.."
cd "$ROOT"

PERIOD="${PERIOD:-600}"
MAX_RELAUNCH="${MAX_RELAUNCH:-5}"
SESSION="${SESSION:-tesi}"
BASE="data/attack/campaign/straight"
LOG="data/attack/verifica/28_notte_watchdog.txt"
OUT_LOG="data/attack/verifica/28_notte.txt"
SIGMAS="0.01 0.02 0.05 0.10"

# Sequenza della notte: la stessa riga lanciata a mano nella finestra campagna.
SEQ="export ISAAC_ARGS=--headless; { scripts/run_campaign.sh straight none 1 10; SIGMAS=\"$SIGMAS\" scripts/run_campaign.sh straight gaussian 1 3; scripts/run_campaign.sh straight nsga3 3 10 8 4; } 2>&1 | tee -a $OUT_LOG"

# braccio -> ripetizioni attese
declare -A EXPECTED=([none_g1]=10 [nsga3_g3_p8x4]=10)
for s in $SIGMAS; do EXPECTED[gaussian_s$s]=3; done
ORDER="none_g1 $(for s in $SIGMAS; do printf 'gaussian_s%s ' "$s"; done)nsga3_g3_p8x4"

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

count_done() {   # $1 = cartella del braccio
  local n=0 d
  for d in "$BASE/$1"/rep_[0-9][0-9]; do
    [[ -f "$d/completata" ]] && n=$((n + 1))
  done
  echo "$n"
}

campaign_alive() { pgrep -f "^bash scripts/run_campaign\.sh" >/dev/null; }

relaunches=0
log "avvio sorveglianza (periodo ${PERIOD} s, massimo ${MAX_RELAUNCH} rilanci)"
while true; do
  line=""
  complete=1
  for arm in $ORDER; do
    n=$(count_done "$arm")
    line+=" $arm $n/${EXPECTED[$arm]}"
    (( n < EXPECTED[$arm] )) && complete=0
  done
  alive="no"; campaign_alive && alive="si"
  log "completate:$line  run_campaign vivo: $alive"

  if (( complete )); then
    log "campagna completa"
    exit 0
  fi

  # Due controlli a 30 s di distanza: fra un comando e l'altro della sequenza
  # nessun run_campaign.sh e' vivo per un istante.
  if [[ "$alive" == "no" ]] && { sleep 30; ! campaign_alive; }; then
    if (( relaunches >= MAX_RELAUNCH )); then
      log "campagna ferma e ${MAX_RELAUNCH} rilanci gia' fatti: esco senza rilanciare"
      exit 1
    fi
    relaunches=$((relaunches + 1))
    win="campagna_r$relaunches"
    tmux new-window -d -t "$SESSION" -n "$win" \
      "source /opt/ros/jazzy/setup.bash && cd $ROOT && $SEQ"
    log "nessun run_campaign.sh vivo e conteggi non raggiunti: rilancio $relaunches nella finestra $win"
  fi
  sleep "$PERIOD"
done
