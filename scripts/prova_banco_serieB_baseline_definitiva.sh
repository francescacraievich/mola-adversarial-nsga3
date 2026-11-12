#!/usr/bin/env bash
# Definitive baseline in direction mode (variant of prova_banco_serieB_baseline_direction.sh):
# N = 5 windows, D_min 140 ms, shadow = victim check (--shadow-check, CRC log).
# One run without attack (arm none, continuous victim) with --goal-mode direction and N guided
# windows, then the polyline used by --damage-ref baseline.
#
# Settings:
#   --goal-mode direction, ramp k 1.0 (WINDOW_STOP=ramp, STOP_RAMP_K=1.0), no arrival,
#   --max-windows N; --extra-windows 3 (victim_held windows do not count; none without attack);
#   pauses --pause-sec 1800 as in the existing baselines;
#   same minimum delay for every arm (MIN_LATENCY_MS), per-cloud log with --frame-log and
#   --frame-log-delay; --fresh-cloud-after-restore in node and orchestrator;
#   settling --settle-sec 2.0 (ORCH_EXTRA comes after the --settle-sec 5.0 of run_campaign.sh
#   and argparse keeps the last one);
#   reliable transport (TRANSPORT=reliable), structural term gross (STRUCTURAL=gross),
#   ICP_QUALITY_LOG=1, SERIE_B=1 with SEEDS=<seed>.
# Without attack the seed does not enter the rollouts (attack_orchestrator.py uses args.seed only
# in the search branches, after the passthrough continue): it only names the run.
#
# x limit: every CONTROLLO_X_S s the maximum true odom x in the repetition traces (written at the
# end of each rollout) is compared with X_MAX_M (7.0 m): beyond it the test ends as INTERROTTA.
# The check lags at most one rollout (TRUE_CAP = 2 m of path per rollout).
# At the end scripts/costruisci_baseline_direction.py repeats the check on the traces, counts the
# windows, looks for pose losses and only if everything holds writes the polyline (new name,
# never overwritten) to data/attack/baseline/rettilineo_direction_N<N>_s<seed>_dmin<D>_sosta<P>.csv.
#
# Output: data/attack/verifica/<date>_baseline_definitiva_s<seed>_N<N>_dmin<D>_sosta<P>/ (new):
# comando.txt, eventi.txt, ambiente.txt, frame_log.csv, shadow_crc.csv,
# campagna/straight/none_g3..._baseline_direction_N<N>/rep_01/, campagna.log,
# baseline_direction.txt, baseline_direction_rapporto.json.
#
# Usage:
#   scripts/prova_banco_serieB_baseline_definitiva.sh --a-secco
#   scripts/prova_banco_serieB_baseline_definitiva.sh            # seed 1001, 5 windows
#   SEME=2001 scripts/prova_banco_serieB_baseline_definitiva.sh
#   PAUSE_SEC=0 STATO="..." scripts/prova_banco_serieB_baseline_definitiva.sh
#
# Settling is counted in clouds as in attacked runs with --victim-never-clean:
# --settle-clouds SETTLE_CLOUDS and --reopen-clouds REOPEN_CLOUDS (20 and 20, provisional).
# The CSV name carries D_min and the pause: a baseline with another D_min or pause does not
# take the name of the definitive one.
set -uo pipefail
export MIN_LATENCY_MS="${MIN_LATENCY_MS:-140}"   # D_min of the runs (ms)

A_SECCO=0
[[ "${1:-}" == "--a-secco" ]] && A_SECCO=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"
SEME="${SEME:-1001}"
FINESTRE="${FINESTRE:-5}"                     # N = 5
PAUSE_SEC="${PAUSE_SEC:-1800}"               # pauses of the existing baselines (s)
SETTLE_CLOUDS="${SETTLE_CLOUDS:-20}"         # as with --victim-never-clean
REOPEN_CLOUDS="${REOPEN_CLOUDS:-20}"         # provisional M (measured: 13 clouds)
STATO="${STATO:-pilota (1 seme)}"
X_MAX_M="${X_MAX_M:-7.0}"                    # maximum true odom x (m)
CONTROLLO_X_S="${CONTROLLO_X_S:-30}"
ORIGINE_COERENTE="${ORIGINE_COERENTE:-0}"    # 1 = orchestrator --origine-coerente-vittima
case "$ORIGINE_COERENTE" in 0) ORIG_X=""; ORIG_D="" ;; 1) ORIG_X=" --origine-coerente-vittima"; ORIG_D=" ORIGINE_COERENTE=1" ;; *) echo "ORIGINE_COERENTE: 0 o 1 (dato: $ORIGINE_COERENTE)"; exit 2 ;; esac
TEMPO_MAX_S="${TEMPO_MAX_S:-16200}"          # 4.5 h, about 1.5x the estimate (3 h 04 min)
OUT="${OUT:-$ROOT/data/attack/verifica/$(date +%F)_baseline_definitiva_s${SEME}_N${FINESTRE}_dmin${MIN_LATENCY_MS}_sosta${PAUSE_SEC%.*}}"
RAPPORTO="$OUT/baseline_direction_rapporto.json"
COSTRUISCI="scripts/costruisci_baseline_direction.py"
COMANDO_PROVA="$0 $* (SEME=$SEME FINESTRE=$FINESTRE PAUSE_SEC=$PAUSE_SEC SETTLE_CLOUDS=$SETTLE_CLOUDS REOPEN_CLOUDS=$REOPEN_CLOUDS X_MAX_M=$X_MAX_M$ORIG_D OUT=$OUT)"
# shellcheck source=prova_banco_serieB_lib.sh
source "$HERE/prova_banco_serieB_lib.sh"
in_sessione_propria "$@"   # leader of its own session (fallback KILL to the group)
BASELINE_CSV="${BASELINE_CSV:-$ROOT/data/attack/baseline/rettilineo_direction_N${FINESTRE}_s${SEME}_dmin${MIN_LATENCY_MS}_sosta${PAUSE_SEC%.*}.csv}"

echo "== baseline direction, seme $SEME, $FINESTRE finestre $( [[ $A_SECCO == 1 ]] && echo '(a secco: nessun lancio)')"
[[ "$SEME" =~ ^[0-9]+$ ]] || { echo "SEME non intero: $SEME"; exit 2; }
[[ "$FINESTRE" =~ ^[1-9][0-9]*$ ]] || { echo "FINESTRE non intero positivo: $FINESTRE"; exit 2; }
[[ "$PAUSE_SEC" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "PAUSE_SEC non numerico: $PAUSE_SEC"; exit 2; }
controlli_iniziali 1
case "$SEME" in 1001|2001|3001) echo "ok: seme $SEME fra quelli della serie B (D5)" ;;
  *) echo "attenzione: seme $SEME fuori da 1001, 2001, 3001 (D5)" ;; esac
[[ "$FINESTRE" == "6" ]] || echo "attenzione: FINESTRE=$FINESTRE, la scelta per la serie B e' 6"
[[ "$PAUSE_SEC" == "1800" ]] || echo "attenzione: PAUSE_SEC=$PAUSE_SEC, le baseline attuali hanno 1800 s (D3)"
for o in '"--goal-mode"' '"--extra-windows"' '"--pause-sec"' '"--fresh-cloud-after-restore"' '"--structural"'; do
  grep -q -- "$o" src/optimization/attack_orchestrator.py || fallito "$o assente dall'orchestratore"
done
for o in '"--frame-log-delay"' '"--fresh-cloud-after-restore"' '"--structural"'; do
  grep -q -- "$o" src/nodes/perturbation_node.py || fallito "$o assente dal nodo"
done
[[ -f "$COSTRUISCI" ]] || fallito "costruttore della baseline assente: $COSTRUISCI"
if [[ -e "$BASELINE_CSV" || -e "${BASELINE_CSV%.csv}.json" ]]; then
  fallito "baseline gia' esistente: $BASELINE_CSV (i file di baseline non si sovrascrivono)"
else echo "ok: baseline di uscita libera: $BASELINE_CSV"; fi
echo "finestre: $FINESTRE (direction, rampa k 1,0), sosta $PAUSE_SEC s, assestamento $SETTLE_CLOUDS nuvole (riapertura $REOPEN_CLOUDS), ritardo minimo $MIN_LATENCY_MS ms, x massima $X_MAX_M m"
echo "durata stimata: $FINESTRE x ~1822 s (sosta di 1800 s e due tratti; baseline live del 20/10: 1820-1825 s per finestra, n 16) + ~1 min di avvio: ~3 h 04 min con 6 finestre; tempo massimo $TEMPO_MAX_S s"
if [[ "$A_SECCO" != "1" && "$CONTROLLI_OK" != "1" ]]; then echo "controlli falliti: prova non lanciata"; exit 1; fi
[[ "$A_SECCO" != "1" ]] && { mkdir -p "$OUT"; echo "$COMANDO_PROVA" > "$OUT/comando.txt"; }

CAMP_PID=""; SEGNALE=""
cleanup() {
  local rc=$?
  [[ "$A_SECCO" == "1" ]] && return 0
  [[ "$PROVA_MIA" == "1" ]] || return 0   # launch refused: nothing to close
  trap '' INT TERM HUP   # shutdown is not interruptible
  inizio_chiusura
  # with Ctrl-C the campaign is already in its own shutdown and is awaited; with TERM or
  # HUP to the script, time limit, missing watchdog or x over the limit: TERM at once
  if [[ -n "$CAMP_PID" ]]; then
    [[ "$SEGNALE" == "TERM" || -n "$MOTIVO_INTERRUZIONE" ]] && kill -TERM "$CAMP_PID" 2>/dev/null
    attendi_poi_term "$CAMP_PID" 130 || nota "IMPREVISTO: run_campaign.sh (pid $CAMP_PID) ancora vivo dopo la chiusura"
    CAMP_PID=""
  fi
  chiudi_mola_della_prova
  prova_fine "$rc"
}
trap cleanup EXIT
trap 'SEGNALE=INT; nota "== interrotta"; exit 130' INT
trap 'SEGNALE=TERM; nota "== terminata"; exit 143' TERM HUP
prova_inizio "baseline_direction_s$SEME" "$TEMPO_MAX_S" || exit 1

esegui "mkdir -p $OUT/campagna"
riavvio_preventivo "$OUT/adattatore.log"
ambiente_su_file "$OUT/ambiente.txt"

CAMP_ENV=(CAMPAIGN_DIR="$OUT/campagna" ISAAC_ARGS="--headless" ATTACK_FROM=spawn STOP_ON=estimate VICTIM=continuous
          WINDOW_STOP=ramp STOP_RAMP_K=1.0 MAX_DRIFT_LAG_S=0.5 SELECT=maxdamage TOPK=2 START_POSE="$START_POSE"
          WARM_START=false SERIE_B=1 SEEDS="$SEME" ICP_QUALITY_LOG=1 TRANSPORT=reliable STRUCTURAL=gross
          MIN_LATENCY_MS="$MIN_LATENCY_MS" RUN_SUFFIX="_baseline_direction_N$FINESTRE"
          NODE_EXTRA="--frame-log $OUT/frame_log.csv --frame-log-delay --fresh-cloud-after-restore --shadow-crc-log $OUT/shadow_crc.csv"
          ORCH_EXTRA="--goal-mode direction --max-windows $FINESTRE --extra-windows 3 --settle-sec 2.0 --settle-clouds $SETTLE_CLOUDS --reopen-clouds $REOPEN_CLOUDS --fresh-cloud-after-restore --pause-sec $PAUSE_SEC --horizon 1 --shadow-check --shadow-crc-log $OUT/shadow_crc.csv --victim-shadow-env --mola-stop-signal int --adapter-events data/attack/verifica/adattatore_eventi.jsonl$ORIG_X")
REP_GLOB="$OUT/campagna/straight/none_g3*_baseline_direction_N$FINESTRE/rep_01"
rep_dir() { local d; for d in $REP_GLOB; do [[ -d "$d" ]] && { echo "$d"; return 0; }; done; return 1; }

if [[ "$A_SECCO" == "1" ]]; then
  printf '+ env'; printf ' %q' "${CAMP_ENV[@]}"; printf ' scripts/run_campaign.sh straight none 3 1 > %s 2>&1\n' "$OUT/campagna.log"
  echo "+ ogni $CONTROLLO_X_S s: python3 $COSTRUISCI --controllo-x $REP_GLOB --x-max $X_MAX_M (oltre: INTERROTTA)"
  rc=0
else
  ( trap - INT; exec env "${CAMP_ENV[@]}" scripts/run_campaign.sh straight none 3 1 ) > "$OUT/campagna.log" 2>&1 &
  CAMP_PID=$!
  t_x=$SECONDS
  while kill -0 "$CAMP_PID" 2>/dev/null; do
    watchdog_in_prova || exit 1
    if (( SECONDS - t_x >= CONTROLLO_X_S )); then
      t_x=$SECONDS
      if rep="$(rep_dir)"; then
        if ! riga="$(python3 "$COSTRUISCI" --controllo-x "$rep" --x-max "$X_MAX_M")"; then
          MOTIVO_INTERRUZIONE="x odom vera oltre $X_MAX_M m ($riga)"
          nota "IMPREVISTO: $MOTIVO_INTERRUZIONE: la prova si chiude"
          exit 1
        fi
      fi
    fi
    sleep 5
  done
  wait "$CAMP_PID"; rc=$?; CAMP_PID=""
  nota "run_campaign.sh uscito con $rc"
  # campaign done: stop the timer before building (the run outcome no longer changes)
  inizio_chiusura
fi

# Build the baseline: the CSV is written only if all checks pass.
if [[ "$A_SECCO" == "1" ]]; then
  echo "+ python3 $COSTRUISCI $REP_GLOB --csv $BASELINE_CSV --json $RAPPORTO --finestre $FINESTRE --x-max $X_MAX_M --seme $SEME --stato '$STATO' > $OUT/baseline_direction.txt 2>&1"
  echo "+ con i controlli superati: cp -n $RAPPORTO ${BASELINE_CSV%.csv}.json"
elif rep="$(rep_dir)"; then
  python3 "$COSTRUISCI" "$rep" --csv "$BASELINE_CSV" --json "$RAPPORTO" --finestre "$FINESTRE" \
    --x-max "$X_MAX_M" --seme "$SEME" --stato "$STATO" > "$OUT/baseline_direction.txt" 2>&1
  rc_b=$?
  if [[ $rc_b -eq 0 ]]; then
    cp -n "$RAPPORTO" "${BASELINE_CSV%.csv}.json"
    nota "baseline scritta: ${BASELINE_CSV#$ROOT/} (rapporto ${RAPPORTO#$ROOT/})"
  else
    PROVA_NOTE="baseline non scritta: controlli falliti o errore (codice $rc_b, ${OUT#$ROOT/}/baseline_direction.txt)"
    nota "$PROVA_NOTE"
    [[ ${rc:-0} -eq 0 ]] && rc=1
  fi
else
  PROVA_NOTE="baseline non scritta: cartella della ripetizione assente ($REP_GLOB)"
  nota "$PROVA_NOTE"
  [[ ${rc:-0} -eq 0 ]] && rc=1
fi
if [[ "$A_SECCO" == "1" && "$CONTROLLI_OK" != "1" ]]; then echo "a secco: il lancio vero si fermerebbe ai controlli falliti sopra"; exit 1; fi
nota "== baseline direction finita: $OUT"
exit "${rc:-0}"
