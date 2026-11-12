#!/usr/bin/env bash
# Shared functions of the bench tests (prova_banco_serieB_*.sh).
# Source after setting ROOT, OUT and A_SECCO (0/1).
#
# Processes are started with the same commands as scripts/run_campaign.sh
# (Isaac, perturbation_node) and the orchestrator (MOLA: launch_mola_attack.sh
# with the topic as argument and ROS_ARGS, TRACES, LIDAR_POSE_X,
# MOLA_INITIAL_SIGMA in the environment, as MolaRunner.spawn). MOLA YAML and
# launch files are left untouched: only environment variables.

ISAAC_PID=""; NODE_PID=""
declare -A MOLA_PID=()
CONTROLLI_OK=1
ADATT_CMD="python3 -u src/nodes/add_intensity_node.py --input-topic /front_3d_lidar/lidar_points --sub-scans-per-cycle 1 --output-frame base_link"
ADATT_MAX_ETA_S="${ADATT_MAX_ETA_S:-14400}"      # 4 h
DISCO_MIN_GB="${DISCO_MIN_GB:-20}"
ISAAC_READY_SEC="${ISAAC_READY_SEC:-300}"
MIN_LATENCY_MS="${MIN_LATENCY_MS:-85}"
START_POSE="${START_POSE:--5.99,-1.0,0}"
CTL="python3 $ROOT/scripts/prova_banco_serieB_ctl.py"
# DDS profile: with DDS_PROFILE=<xml>, FASTRTPS_DEFAULT_PROFILES_FILE (absolute path)
# is passed to Isaac, node, adapter and the MOLA instances of the test; empty = MOLA
# without the variable (factory transport). Checks in dds_profilo_prepara, to be
# called before starting the processes.
DDS_PROFILE="${DDS_PROFILE:-}"
dds_profilo_prepara() {
  [[ -z "$DDS_PROFILE" ]] && return 0
  [[ -f "$DDS_PROFILE" ]] || { nota "DDS_PROFILE=$DDS_PROFILE: file inesistente"; return 1; }
  DDS_PROFILE="$(realpath "$DDS_PROFILE")"
  [[ -e "$PWD/DEFAULT_FASTRTPS_PROFILES.xml" ]] \
    && { nota "DDS_PROFILE: $PWD/DEFAULT_FASTRTPS_PROFILES.xml presente (profili caricati due volte)"; return 1; }
  export DDS_PROFILE FASTRTPS_DEFAULT_PROFILES_FILE="$DDS_PROFILE"
  nota "profilo DDS: $DDS_PROFILE (md5 $(md5sum < "$DDS_PROFILE" | cut -d' ' -f1))"
}
# MOLA_LIDAR_TOPIC_TYPE value for reliable QoS: same as run_campaign.sh.
QOS_LIDAR_TYPE=$'PointCloud2\n          qos:\n            reliability: reliable\n            depth: 50'

nota() { echo "$(date +%F' '%T)  $*" | { if [[ "$A_SECCO" == "1" || ! -d "$OUT" ]]; then cat; else tee -a "$OUT/eventi.txt"; fi; }; }
fallito() { echo "CONTROLLO FALLITO: $*"; CONTROLLI_OK=0; }
# Run the command, or only print it with --a-secco.
esegui() { if [[ "$A_SECCO" == "1" ]]; then printf '+ %s\n' "$*"; else eval "$@"; fi; }
# on source: with an invalid profile the test does not start
dds_profilo_prepara || exit 1
# Local discovery: DISCOVERY_LOCALE=1 exports ROS_AUTOMATIC_DISCOVERY_RANGE for every
# process of the test: LOCALHOST without DDS_PROFILE, SYSTEM_DEFAULT with DDS_PROFILE
# (local discovery lives in the profile, and rmw leaves the profile intact only with
# SYSTEM_DEFAULT); ROS_LOCALHOST_ONLY is unset. riavvio_preventivo restarts the adapter
# with the same value, and the adapter is not reset to SUBNET at the end of the test.
# 0 = default discovery.
DISCOVERY_LOCALE="${DISCOVERY_LOCALE:-0}"
case "$DISCOVERY_LOCALE" in
  0) : ;;
  1) if [[ -n "$DDS_PROFILE" ]]; then
       grep -q "<initialPeersList>" "$DDS_PROFILE" 2>/dev/null && grep -q "127\.0\.0\.1" "$DDS_PROFILE" \
         || { nota "DISCOVERY_LOCALE=1 con DDS_PROFILE=$DDS_PROFILE senza scoperta locale nel profilo (initialPeersList 127.0.0.1)"; exit 1; }
       export ROS_AUTOMATIC_DISCOVERY_RANGE=SYSTEM_DEFAULT
       SCOPERTA_MOTIVO="profilo DDS con la scoperta locale dentro: SYSTEM_DEFAULT perche' rmw non tocchi il profilo"
     else
       export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
       SCOPERTA_MOTIVO="senza profilo DDS: scoperta locale da rmw (LOCALHOST)"
     fi
     unset ROS_LOCALHOST_ONLY
     export DISCOVERY_LOCALE
     nota "scoperta locale: ROS_AUTOMATIC_DISCOVERY_RANGE=$ROS_AUTOMATIC_DISCOVERY_RANGE per tutta la catena, ROS_LOCALHOST_ONLY tolta ($SCOPERTA_MOTIVO)" ;;
  *) nota "DISCOVERY_LOCALE sconosciuto: $DISCOVERY_LOCALE (0 o 1)"; exit 1 ;;
esac

# Processes of a simulation (same forms as run_campaign.sh and the orchestrator).
processi_simulazione() {
  {
    # anchored patterns: command lines that only mention these files (editors, tail) are ignored
    pgrep -af "^(/usr)?(/bin/)?(bash|sh) .*scripts/run_campaign\.sh" ; pgrep -af "^python3( -u)? .*attack_orchestrator\.py"
    pgrep -af "^python3( -u)? .*run_isaac_standalone\.py" ; pgrep -ax isaacsim
    pgrep -af "^python3( -u)? .*perturbation_node\.py" ; pgrep -ax mola-cli
    pgrep -ax mola-lidar-odom ; pgrep -af "^(\S*python3 )?\S*ros2 bag record"
  } 2>/dev/null | grep -v "prova_banco_serieB" || true
}
# Queue runner and launcher (shells executing those files, not readers such as less or tail).
processi_coda() { pgrep -af "^(/usr)?(/bin/)?(bash|sh) .*(runner_coda|lancia_sim)\.sh" 2>/dev/null | grep -v "prova_banco_serieB" || true; }
processi_catena() { processi_simulazione; processi_coda; }

pids_adattatore() { pgrep -f "^python3( -u)? .*add_intensity_node\.py"; }
pid_adattatore() { pids_adattatore | head -1; }

# Simulation queue and monitors acting on the bench: the CODA_IN_CORSO sentinel keeps
# alive the adapter monitor (restarts the adapter in tesi:adapt without
# ROS_AUTOMATIC_DISCOVERY_RANGE, even with Isaac playing, after 120 s) and the Isaac
# watchdog (kills Isaac if its pose is frozen). Neither has a pause file: the test
# refuses to start while they are active.
SENTINELLA_CODA="$ROOT/data/attack/verifica/CODA_IN_CORSO"
controlli_coda() {
  local proc fin
  # blockers: adapter monitor and the legacy watchdog (not the heartbeat one below)
  proc="$(pgrep -af "^(/usr)?(/bin/)?(bash|sh) .*(sorveglianza_adattatore|watchdog_isaac_fermo|adapter_watch)" 2>/dev/null | grep -v "prova_banco_serieB" || true)"
  fin="$(tmux list-windows -a -F '#{session_name}:#{window_name}' 2>/dev/null | grep -E ':(sorvadapt[0-9]*|wdisaac|sorveglianza.*)$' || true)"
  if [[ -n "$proc" || -n "$fin" || -e "$SENTINELLA_CODA" ]]; then
    fallito "coda o sorveglianze attive: riavvierebbero l'adattatore o terminerebbero Isaac durante la prova"
    [[ -n "$proc" ]] && echo "$proc" | sed 's/^/    processo: /'
    [[ -n "$fin" ]] && echo "$fin" | sed 's/^/    finestra tmux: /'
    [[ -e "$SENTINELLA_CODA" ]] && echo "    sentinella: $SENTINELLA_CODA"
    echo "  per fermarle prima della prova (a coda ferma, nessuna cancellazione):"
    [[ -e "$SENTINELLA_CODA" ]] && echo "    mv $SENTINELLA_CODA $SENTINELLA_CODA.sospesa_$(date +%Y%m%d_%H%M)"
    for w in $fin; do echo "    tmux kill-window -t $w"; done
    for pid in $(awk '{print $1}' <<< "$proc"); do echo "    kill -TERM $pid"; done
  else
    echo "ok: nessuna coda ne' sorveglianza attiva (sentinella assente)"
  fi
}

# --- Isaac watchdog with heartbeat ---
# The watchdog writes a heartbeat to FASE3_WATCHDOG.json every 60 s (pid, start time,
# heartbeat, root). A test on the chain starts only if the heartbeat is recent, the pid
# is alive with the same start time and the root is the repo running this script; the
# check is repeated during the test. The process name is not used.
SENTINELLA_FASE3="$ROOT/data/attack/verifica/FASE3_IN_CORSO"
BATTITO_FASE3="$ROOT/data/attack/verifica/FASE3_WATCHDOG.json"
BATTITO_MAX_S="${BATTITO_MAX_S:-150}"
WATCHDOG_FASE3="${WATCHDOG_FASE3:-}"   # path of the watchdog script, only for the message

# process start time (field 22 of /proc/<pid>/stat), empty if the pid does not exist
avvio_pid() { [[ -r "/proc/$1/stat" ]] || return 0; sed 's/^.*) //' "/proc/$1/stat" 2>/dev/null | awk '{print $20}'; }
# true if the pid exists and has the given start time
pid_vivo_con_avvio() { [[ -n "$1" && -n "$2" && "$(avvio_pid "$1")" == "$2" ]]; }

watchdog_stato() {   # prints "ok" or the reason for refusal
  python3 - "$BATTITO_FASE3" "$ROOT" "$BATTITO_MAX_S" <<'EOF'
import json, os, sys, time
f, root, max_s = sys.argv[1], sys.argv[2], float(sys.argv[3])
try:
    d = json.load(open(f))
except Exception as e:
    print(f"battito assente o illeggibile ({f})"); raise SystemExit(0)
pid, avvio, batt, rad = d.get("pid"), str(d.get("avvio", "")), d.get("battito"), d.get("radice", "")
try:
    st = open(f"/proc/{int(pid)}/stat").read()
    vero = st[st.rindex(")") + 2:].split()[19]
except Exception:
    vero = None
if vero is None:
    print(f"pid {pid} del watchdog non esiste"); raise SystemExit(0)
if vero != avvio:
    print(f"pid {pid} riusato da un altro processo (avvio {vero}, atteso {avvio})"); raise SystemExit(0)
eta = time.time() - float(batt or 0)
if eta >= max_s:
    print(f"battito vecchio di {eta:.0f} s (massimo {max_s:.0f} s)"); raise SystemExit(0)
if os.path.realpath(rad) != os.path.realpath(root):
    print(f"il watchdog lavora su {rad}, lo script su {root}"); raise SystemExit(0)
print("ok")
EOF
}

controlli_watchdog_fase3() {
  local st; st="$(watchdog_stato)"
  [[ "$st" == "ok" && ! -e "$SENTINELLA_FASE3" ]] && st="sentinella FASE3_IN_CORSO assente"
  if [[ "$st" == "ok" && -e "$SENTINELLA_FASE3" ]]; then
    echo "ok: watchdog di Isaac della fase 3 attivo (battito recente in $BATTITO_FASE3, radice uguale)"
    return 0
  fi
  fallito "watchdog di Isaac della fase 3 non attivo: la prova sulla catena non parte senza ($st)"
  [[ -e "$SENTINELLA_FASE3" ]] || echo "    sentinella assente: touch $SENTINELLA_FASE3"
  echo "    avviarlo (a coda ferma, con il watchdog vecchio spento): tmux new-window -t tesi -n wdfase3 'bash ${WATCHDOG_FASE3:-<percorso di watchdog_isaac_fase3.sh>}'"
}

# During the test: check at most every CONTROLLO_WD_S seconds (and on every call with
# "subito"); if the watchdog is gone the test ends as INTERROTTA.
CONTROLLO_WD_S="${CONTROLLO_WD_S:-300}"
ULTIMO_CONTROLLO_WD=0
watchdog_in_prova() {   # $1 = "subito" to force the check; rc 1 if the watchdog is missing
  [[ "$A_SECCO" == "1" ]] && return 0
  if [[ "${1:-}" != "subito" ]] && (( SECONDS - ULTIMO_CONTROLLO_WD < CONTROLLO_WD_S )); then return 0; fi
  ULTIMO_CONTROLLO_WD=$SECONDS
  local st; st="$(watchdog_stato)"
  [[ "$st" == "ok" && -e "$SENTINELLA_FASE3" ]] && return 0
  [[ "$st" == "ok" ]] && st="sentinella FASE3_IN_CORSO assente"
  MOTIVO_INTERRUZIONE="watchdog assente ($st)"
  nota "IMPREVISTO: watchdog di Isaac della fase 3 assente durante la prova: $st"
  vault_riga "- $(date +%H:%M) IMPREVISTO: prova al banco $PROVA_NOME, watchdog di Isaac della fase 3 assente ($st): la prova si chiude" ""
  return 1
}

# --- Own session (setsid) ---
# The script re-executes itself as leader of a new session, so the fallback KILL of the
# watchdog to the process group (pgid) also stops campaign, recorder and timer. The outer
# process only forwards signals: INT to the group (like Ctrl-C), TERM and HUP to the script.
in_sessione_propria() {   # call with the script's "$@"
  [[ "$A_SECCO" == "1" ]] && return 0
  [[ "$(ps -o sid= -p $$ | tr -d ' ')" == "$$" ]] && return 0
  local figlio rc=0 se
  # absolute path: the script has already cd'ed into the root, $0 may be relative
  se="${HERE:-$(cd "$(dirname "$0")" && pwd)}/$(basename "${BASH_SOURCE[-1]}")"
  [[ -f "$se" ]] || { echo "rilancio sotto setsid impossibile: script non trovato ($se)"; exit 2; }
  ( trap - INT; exec setsid bash "$se" "$@" ) &
  figlio=$!
  trap 'kill -INT -- "-$figlio" 2>/dev/null' INT
  trap 'kill -TERM "$figlio" 2>/dev/null' TERM HUP
  # wait interrupted by a forwarded signal: retry until the child is reaped
  while :; do
    wait "$figlio"; rc=$?
    kill -0 "$figlio" 2>/dev/null || break
  done
  exit "$rc"
}

# --- Test state (PROVA_IN_CORSO.json), time limit, outcome in the daily log ---
VAULT="${VAULT:-$ROOT/data/attack/registro_giornaliero}"
VAULT_LOG_PREFISSO="${VAULT_LOG_PREFISSO:-}"   # daily log is "<prefix>YYYY-MM-DD.md"
FILE_PROVA="$ROOT/data/attack/verifica/PROVA_IN_CORSO.json"
LOCK_PROVA="$ROOT/data/attack/verifica/PROVA_IN_CORSO.lock"
LOCK_CODA="$ROOT/data/attack/verifica/CODA_VAULT.lock"
LOG_NOTTE="$ROOT/data/attack/dossier/LOG_notte.txt"
SCRIPT_PID=$$
SCRIPT_AVVIO="$(avvio_pid $$)"
# initialized here: never taken from the environment
PROVA_NOME=""; PROVA_MIA=0; TIMER_PID=""; SEGNALE=""; PROVA_NOTE=""; MOTIVO_INTERRUZIONE=""

controlli_vault() {
  if [[ -z "$VAULT_LOG_PREFISSO" ]]; then
    fallito "esportare VAULT_LOG_PREFISSO (prefisso del log del giorno nel vault, es. il nome fino alla data)"
  elif [[ ! -d "$VAULT" ]]; then
    fallito "vault assente: $VAULT"
  else
    echo "ok: vault $VAULT, log del giorno ${VAULT_LOG_PREFISSO}$(date +%F).md"
  fi
}

conta_interventi_watchdog() { grep -c "\[watchdog isaac fase3\] IMPREVISTO: posa di Isaac ferma" "$LOG_NOTTE" 2>/dev/null || true; }

prova_inizio() {   # $1 = name, $2 = time limit (s); after creating OUT and installing the trap
  PROVA_TMAX="$2"; PROVA_INIZIO="$(date +%FT%T)"
  PROVA_WD0="$(conta_interventi_watchdog)"; PROVA_WD0="${PROVA_WD0:-0}"
  if [[ "$A_SECCO" == "1" ]]; then
    echo "+ $FILE_PROVA (pid, avvio, pgid, nome=$1, inizio, tempo_max_s=$2, scade_epoch, cartella), creazione esclusiva (flock) con file temporaneo e mv; timer di $2 s che manda TERM allo script"
    return 0
  fi
  mkdir -p "$(dirname "$FILE_PROVA")"
  # exclusive creation: check and write under the same lock
  local esito
  esito=$(
    exec 9>"$LOCK_PROVA"
    flock -n 9 || { echo "lock occupato ($LOCK_PROVA)"; exit 0; }
    if [[ -e "$FILE_PROVA" ]]; then
      p=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('pid',''))" "$FILE_PROVA" 2>/dev/null)
      a=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('avvio',''))" "$FILE_PROVA" 2>/dev/null)
      if [[ -n "$p" ]] && { [[ -z "$a" ]] && kill -0 "$p" 2>/dev/null || pid_vivo_con_avvio "$p" "$a"; }; then
        echo "un'altra prova al banco e' in corso (pid $p, $FILE_PROVA)"; exit 0
      fi
      mv "$FILE_PROVA" "$FILE_PROVA.orfano_$(date +%Y%m%d_%H%M%S)"
    fi
    python3 - "$FILE_PROVA.tmp" "$SCRIPT_PID" "$SCRIPT_AVVIO" "$(ps -o pgid= -p $$ | tr -d ' ')" "$1" "$PROVA_INIZIO" "$2" "$OUT" <<'EOF'
import json, sys, time
f, pid, avvio, pgid, nome, inizio, tmax, cart = sys.argv[1:9]
json.dump({"pid": int(pid), "avvio": avvio, "pgid": int(pgid), "nome": nome, "inizio": inizio,
           "tempo_max_s": int(tmax), "scade_epoch": int(time.time()) + int(tmax), "cartella": cart},
          open(f, "w"), indent=1)
EOF
    mv "$FILE_PROVA.tmp" "$FILE_PROVA" && echo ok
  )
  if [[ "$esito" != "ok" ]]; then
    nota "IMPREVISTO: $esito: lancio rifiutato"
    # the output folder is not left occupied: a new launch finds it free
    [[ -d "$OUT" ]] && mv "$OUT" "${OUT}_rifiutato_$(date +%Y%m%d_%H%M%S)" \
      && echo "cartella d'uscita spostata in ${OUT}_rifiutato_<ora> (nessun dato della prova)"
    return 1
  fi
  PROVA_NOME="$1"; PROVA_MIA=1
  # test tag in the environment of every process started from here (MOLA in their own
  # session too): used to find them by pid without touching other chains
  export PROVA_BANCO_ID="$SCRIPT_PID:$SCRIPT_AVVIO"
  # timer: on expiry, if the script is still the same (pid and start time) and not already
  # closing, record the reason and send it TERM (the trap shuts the chain down)
  ( trap 'kill $s 2>/dev/null; exit 0' TERM
    sleep "$2" & s=$!; wait "$s"
    pid_vivo_con_avvio "$SCRIPT_PID" "$SCRIPT_AVVIO" || exit 0
    [[ -e "$OUT/.in_chiusura" ]] && exit 0
    touch "$OUT/.tempo_massimo_scaduto"; kill -TERM "$SCRIPT_PID" 2>/dev/null ) >/dev/null 2>&1 &
  TIMER_PID=$!
  nota "prova $1 avviata: $FILE_PROVA, tempo massimo $2 s"
}

# Start of shutdown: from here on the timer no longer changes the outcome.
inizio_chiusura() {
  [[ "$PROVA_MIA" == "1" ]] || return 0
  touch "$OUT/.in_chiusura" 2>/dev/null
  [[ -n "$TIMER_PID" ]] && kill -TERM "$TIMER_PID" 2>/dev/null; TIMER_PID=""
  return 0
}

vault_riga() {   # $1 = line for the daily log, $2 = line for the queue history (empty = none)
  echo "$1"
  if ! python3 - "$VAULT" "$VAULT_LOG_PREFISSO" "$1" "$2" "$LOCK_CODA" <<'EOF'
import fcntl, os, sys, time
from pathlib import Path
v, pref, riga_log, riga_coda, lock = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
if not v.is_dir() or not pref:
    raise SystemExit(f"vault assente o VAULT_LOG_PREFISSO vuoto: {v}")
with open(v / f"{pref}{time.strftime('%Y-%m-%d')}.md", "a") as f:
    f.write(riga_log + "\n")
q = v / "00 Coda simulazioni.md"
if riga_coda and q.exists():
    # read, modify and replace under the same lock (shared by the bench scripts)
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    fl = open(lock, "w"); fcntl.flock(fl, fcntl.LOCK_EX)
    L = q.read_text().splitlines()
    i = next((k for k, l in enumerate(L) if l.strip() == "## Storico"), None)
    if i is None:
        L += ["", "## Storico", riga_coda]
    else:
        j = next((k for k in range(i + 1, len(L)) if L[k].startswith("## ")), len(L))
        while j > i + 1 and not L[j - 1].strip():
            j -= 1
        L.insert(j, riga_coda)
    tmp = q.with_name(q.name + f".tmp_{os.getpid()}")
    tmp.write_text("\n".join(L) + "\n")
    os.replace(tmp, q)
    fcntl.flock(fl, fcntl.LOCK_UN); fl.close()
EOF
  then
    # fallback: the line goes at least to the night log
    mkdir -p "$(dirname "$LOG_NOTTE")"
    echo "$(date +'%F %T')  [prova al banco] scrittura nel vault non riuscita: $1" >> "$LOG_NOTTE"
    nota "IMPREVISTO: scrittura nel vault non riuscita ($VAULT): riga nel log notturno"
  fi
}

prova_fine() {   # $1 = exit code; last action of the shutdown
  [[ "$A_SECCO" == "1" || "$PROVA_MIA" != "1" ]] && return 0
  [[ -n "$TIMER_PID" ]] && kill -TERM "$TIMER_PID" 2>/dev/null; TIMER_PID=""
  local rc="${1:-0}" esito motivo wd fine rel dest
  wd="$(conta_interventi_watchdog)"
  if [[ -e "$OUT/.tempo_massimo_scaduto" ]]; then esito=INTERROTTA; motivo="tempo massimo ($PROVA_TMAX s)"
  elif [[ -n "$MOTIVO_INTERRUZIONE" ]]; then esito=INTERROTTA; motivo="$MOTIVO_INTERRUZIONE"
  elif [[ -n "$SEGNALE" ]]; then esito=INTERROTTA; motivo="segnale $SEGNALE"
  elif [[ "$rc" != "0" ]]; then esito=ERRORE; motivo="codice $rc"
  elif (( ${wd:-0} > PROVA_WD0 )); then esito=INTERROTTA; motivo="Isaac terminato dal watchdog"
  else esito=COMPLETATA; motivo="fine normale"; fi
  [[ "$motivo" != *"watchdog"* ]] && (( ${wd:-0} > PROVA_WD0 )) && motivo="$motivo; Isaac terminato dal watchdog"
  [[ -n "$PROVA_NOTE" ]] && motivo="$motivo; $PROVA_NOTE"
  fine="$(date +%FT%T)"
  rel="${OUT#$ROOT/}"
  if [[ -e "$FILE_PROVA" ]]; then
    python3 - "$FILE_PROVA" "$esito" "$motivo" "$fine" "$rc" <<'EOF'
import json, sys
f, esito, motivo, fine, rc = sys.argv[1:6]
d = json.load(open(f)); d.update({"esito": esito, "motivo": motivo, "fine": fine, "rc": int(rc)})
json.dump(d, open(f + ".tmp", "w"), indent=1)
EOF
    mv "$FILE_PROVA.tmp" "$FILE_PROVA"
    mkdir -p "$OUT"; dest="$OUT/prova_stato_finale.json"
    [[ -e "$dest" ]] && dest="$OUT/prova_stato_finale_$(date +%H%M%S).json"
    mv "$FILE_PROVA" "$dest"
  fi
  nota "== prova $PROVA_NOME: $esito ($motivo)"
  vault_riga "- $(date +%H:%M) prova al banco $PROVA_NOME: $esito ($motivo), cartella \`$rel\`" \
             "- [$( [[ $esito == COMPLETATA ]] && echo x || echo '!')] prova_banco $PROVA_NOME | ${PROVA_INIZIO/T/ } → ${fine/T/ } | $esito: $motivo | \`$rel\`"
  PROVA_NOME=""; PROVA_MIA=0
}

# md5 of the content of the code files (modified and untracked included).
hash_codice() {
  local files
  if git -C "$ROOT" rev-parse HEAD >/dev/null 2>&1; then
    files=$(git -C "$ROOT" ls-files -c -m -o --exclude-standard -- src scripts isaac config tests 2>/dev/null | sort -u)
  else
    files=$(cd "$ROOT" && find src scripts isaac config tests -type f ! -path '*__pycache__*' 2>/dev/null | sort)
  fi
  (cd "$ROOT" && for f in $files; do [[ -f "$f" ]] && md5sum "$f"; done) | md5sum | cut -c1-12
}

controlli_iniziali() {   # $1 = 1 if the adapter is needed; $2 = 1 (default) chain test, 0 offline
  local att pid eta libero pct sulla_catena="${2:-1}"
  att="$(processi_simulazione)"
  if [[ -n "$att" ]]; then fallito "simulazione in corso (una prova alla volta):"; echo "$att" | sed 's/^/    /'
    mola_orfane_di_prove
  else echo "ok: nessuna simulazione in corso"; fi
  if [[ "$sulla_catena" == "1" ]]; then
    att="$(processi_coda)"
    if [[ -n "$att" ]]; then fallito "runner della coda o lancio attivo: fermare la coda prima della prova"; echo "$att" | sed 's/^/    /'
    else echo "ok: nessun runner della coda attivo"; fi
  fi
  libero=$(df --output=avail -BG "$ROOT" | tail -1 | tr -dc 0-9); pct=$(df --output=pcent "$ROOT" | tail -1 | tr -dc 0-9)
  if (( libero < DISCO_MIN_GB || pct >= 90 )); then fallito "disco: ${libero} GB liberi, ${pct}% usato (servono >= ${DISCO_MIN_GB} GB e < 90%)"
  else echo "ok: disco ${libero} GB liberi (${pct}% usato)"; fi
  for b in ros2 mola-yaml-parser tmux; do command -v "$b" >/dev/null || fallito "comando assente: $b"; done
  [[ -n "${ROS_DISTRO:-}" ]] || fallito "ambiente ROS non caricato (source /opt/ros/jazzy/setup.bash)"
  [[ -e "$OUT" ]] && fallito "cartella di uscita gia' esistente: $OUT (le prove vanno in cartelle nuove)"
  echo "uscita: $OUT"
  if [[ "${1:-1}" == "1" ]]; then
    pid=$(pid_adattatore)
    if [[ -z "$pid" ]]; then echo "adattatore non in esecuzione: verra' avviato"
    else eta=$(ps -o etimes= -p "$pid" | tr -dc 0-9)
      echo "adattatore pid $pid acceso da ${eta} s ($(tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep '^ROS_AUTOMATIC_DISCOVERY_RANGE=' || echo 'ROS_AUTOMATIC_DISCOVERY_RANGE non impostata'))"
      (( eta > ADATT_MAX_ETA_S )) && echo "  oltre $((ADATT_MAX_ETA_S/3600)) h: riavvio preventivo prima della prova (D17), con Isaac spento o in pausa"
    fi
  fi
  # an offline test does not use the chain: queue and monitors cannot affect it
  if [[ "$sulla_catena" == "1" ]]; then controlli_coda; controlli_watchdog_fase3
  else echo "fuori linea: coda, sorveglianze e watchdog della fase 3 non controllati"; fi
  controlli_vault
  echo "codice: $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo 'non git'), md5 dei file: $(hash_codice)"
}

isaac_in_pausa_o_spento() {
  pgrep -f "^python3( -u)? .*run_isaac_standalone\.py" >/dev/null || pgrep -x isaacsim >/dev/null || return 0
  python3 -c 'import json,sys,time; d=json.load(open("/tmp/isaac_pose.json")); sys.exit(0 if (not d.get("playing", True)) and time.time()-float(d.get("wall",0))<5 else 1)' 2>/dev/null
}

# Adapter in tmux window tesi:adapt, with the current environment
# (ROS_AUTOMATIC_DISCOVERY_RANGE included). Only with Isaac stopped or paused.
riavvia_adattatore() {   # $1 = log file
  if [[ "$A_SECCO" == "1" ]]; then
    printf '+ tmux kill-window -t tesi:adapt; tmux new-window -d -t tesi -n adapt "ROS_AUTOMATIC_DISCOVERY_RANGE=%s %s%s > %s"\n' \
      "${ROS_AUTOMATIC_DISCOVERY_RANGE:-}" "${DDS_PROFILE:+FASTRTPS_DEFAULT_PROFILES_FILE=$DDS_PROFILE }" "$ADATT_CMD" "$1"; return 0; fi
  isaac_in_pausa_o_spento || { nota "IMPREVISTO: riavvio dell'adattatore chiesto con Isaac in play: rifiutato"; return 1; }
  local vecchi nuovi p
  vecchi="$(pids_adattatore | tr '\n' ' ')"
  tmux has-session -t tesi 2>/dev/null || tmux new-session -d -s tesi
  tmux kill-window -t tesi:adapt 2>/dev/null
  # the old adapter must be dead before starting another one
  for _ in $(seq 1 15); do
    p=""; for v in $vecchi; do kill -0 "$v" 2>/dev/null && p="$p $v"; done
    [[ -z "$p" ]] && break; sleep 1
  done
  if [[ -n "$p" ]]; then
    nota "IMPREVISTO: adattatore ancora vivo dopo la chiusura di tesi:adapt (pid$p, fuori dalla finestra?): nessun riavvio. Chiuderlo a mano: kill -TERM$p"
    return 1
  fi
  local dds=""
  [[ -n "$DDS_PROFILE" ]] && dds="export FASTRTPS_DEFAULT_PROFILES_FILE=$DDS_PROFILE && "
  # the tmux window inherits the tmux server environment: unset ROS_LOCALHOST_ONLY there
  [[ "$DISCOVERY_LOCALE" == "1" ]] && dds="unset ROS_LOCALHOST_ONLY && $dds"
  tmux new-window -d -t tesi -n adapt "source /opt/ros/jazzy/setup.bash && export ROS_AUTOMATIC_DISCOVERY_RANGE=${ROS_AUTOMATIC_DISCOVERY_RANGE:-SUBNET} && ${dds}cd $ROOT && $ADATT_CMD > $1 2>&1"
  sleep 6
  nuovi="$(pids_adattatore | tr '\n' ' ')"
  set -- $nuovi
  if [[ $# -eq 0 ]]; then nota "IMPREVISTO: adattatore non ripartito"; return 1; fi
  if [[ $# -gt 1 ]]; then nota "IMPREVISTO: $# adattatori in esecuzione (pid $nuovi): ne deve restare uno solo"; return 1; fi
  nota "adattatore avviato (pid $1, ROS_AUTOMATIC_DISCOVERY_RANGE=${ROS_AUTOMATIC_DISCOVERY_RANGE:-SUBNET})"
}

# Adapter reset to SUBNET if it runs with another discovery range (end of test or exit
# on error): only after Isaac has been shut down.
adattatore_a_subnet() {   # $1 = log file
  local pid r; pid=$(pid_adattatore); [[ -z "$pid" ]] && return 0
  r=$(tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | sed -n 's/^ROS_AUTOMATIC_DISCOVERY_RANGE=//p')
  [[ "${r:-SUBNET}" == "SUBNET" ]] && return 0
  if [[ "$DISCOVERY_LOCALE" == "1" ]]; then nota "adattatore lasciato a ROS_AUTOMATIC_DISCOVERY_RANGE=$r (DISCOVERY_LOCALE=1)"; return 0; fi
  export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
  nota "adattatore con ROS_AUTOMATIC_DISCOVERY_RANGE=$r: riportato a SUBNET"
  riavvia_adattatore "$1"
}

riavvio_preventivo() {   # $1 = log file
  local pid eta; pid=$(pid_adattatore)
  if [[ -z "$pid" ]]; then riavvia_adattatore "$1"; return; fi
  eta=$(ps -o etimes= -p "$pid" | tr -dc 0-9)
  if (( eta > ADATT_MAX_ETA_S )); then nota "adattatore acceso da $((eta/3600)) h: riavvio preventivo (D17)"; riavvia_adattatore "$1"
  elif [[ -n "$DDS_PROFILE" ]] && ! tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -qxF "FASTRTPS_DEFAULT_PROFILES_FILE=$DDS_PROFILE"; then
    # DDS profile on: the adapter must have it (otherwise run_campaign.sh stops)
    nota "adattatore pid $pid senza FASTRTPS_DEFAULT_PROFILES_FILE=$DDS_PROFILE: riavviato con il profilo"; riavvia_adattatore "$1"
  elif [[ "$DISCOVERY_LOCALE" == "1" ]] && { ! tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -qx "ROS_AUTOMATIC_DISCOVERY_RANGE=$ROS_AUTOMATIC_DISCOVERY_RANGE" \
       || tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -qx "ROS_LOCALHOST_ONLY=1"; }; then
    # local discovery: the adapter must have the same value (otherwise run_campaign.sh stops)
    nota "adattatore pid $pid senza ROS_AUTOMATIC_DISCOVERY_RANGE=$ROS_AUTOMATIC_DISCOVERY_RANGE (o con ROS_LOCALHOST_ONLY=1): riavviato con la scoperta locale"; riavvia_adattatore "$1"
  else nota "adattatore pid $pid acceso da ${eta} s: tenuto"; fi
}

stop_pid() {   # $1 = pid, $2 = first signal, $3 = seconds to wait (as in run_campaign.sh)
  local pid="$1"; [[ -z "$pid" ]] && return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill -"$2" "$pid" 2>/dev/null || true
    for _ in $(seq 1 "$3"); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 15); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -KILL "$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
}

isaac_start() {   # $1 = log file; always headless
  if [[ "$A_SECCO" == "1" ]]; then
    printf '+ python3 -u isaac/run_isaac_standalone.py --headless > %s   # attesa di "[standalone] pronto"\n' "$1"
    printf '+ %s isaac set_pose %s\n' "$CTL" "${START_POSE//,/ }"; return 0; fi
  ( trap - INT; exec python3 -u isaac/run_isaac_standalone.py --headless ) > "$1" 2>&1 &
  ISAAC_PID=$!
  local t0=$SECONDS
  until grep -q "\[standalone\] pronto" "$1" 2>/dev/null; do
    kill -0 "$ISAAC_PID" 2>/dev/null || { nota "IMPREVISTO: Isaac terminato durante l'avvio ($1)"; return 1; }
    (( SECONDS - t0 > ISAAC_READY_SEC )) && { nota "IMPREVISTO: Isaac non pronto dopo $ISAAC_READY_SEC s"; return 1; }
    sleep 2
  done
  sleep 5
  $CTL isaac set_pose ${START_POSE//,/ } >> "$OUT/eventi.txt" 2>&1 || { nota "IMPREVISTO: set_pose fallito"; return 1; }
  nota "Isaac headless pronto (pid $ISAAC_PID), posa iniziale $START_POSE, robot fermo (nessun /cmd_vel)"
}

isaac_stop() { [[ "$A_SECCO" == "1" ]] && { echo "+ chiusura di Isaac (INT, poi TERM, KILL)"; return 0; }; stop_pid "$ISAAC_PID" INT 30; ISAAC_PID=""; }

node_start() {   # $1 = log file, $2 = per-frame log
  local cmd="python3 -u src/nodes/perturbation_node.py --min-latency-ms $MIN_LATENCY_MS --start-disabled --max-drift-lag-s 0.5 --frame-log $2"
  if [[ "$A_SECCO" == "1" ]]; then printf '+ %s > %s\n+ %s status --latency %s\n' "$cmd" "$1" "$CTL" "$MIN_LATENCY_MS"; return 0; fi
  ( trap - INT; exec $cmd ) > "$1" 2>&1 &
  NODE_PID=$!
  $CTL status --latency "$MIN_LATENCY_MS" --timeout 60 >> "$OUT/eventi.txt" 2>&1 \
    || { nota "IMPREVISTO: perturbation_node senza status o senza la latenza minima $MIN_LATENCY_MS ms"; return 1; }
  nota "perturbation_node avviato (pid $NODE_PID): $cmd"
}

node_stop() { [[ "$A_SECCO" == "1" ]] && { echo "+ chiusura del perturbation_node (INT: il registro per frame viene chiuso)"; return 0; }; stop_pid "$NODE_PID" INT 10; NODE_PID=""; }

# MOLA as the orchestrator starts it: bash launch_mola_attack.sh <topic>, with
# ROS_ARGS (namespace) and TRACES; $3... = extra variables (VAR=value ...).
mola_start() {   # $1 = role (victim|attacker), $2 = folder, $3... = variables
  local ruolo="$1" dir="$2"; shift 2
  local topic="/carter/lidar_$ruolo"
  local ra="[\"--ros-args\", \"-r\", \"__ns:=/$ruolo\"]"
  if [[ "$A_SECCO" == "1" ]]; then
    printf "+ env ROS_ARGS='%s' TRACES=%s %s setsid bash src/nodes/launch_mola_attack.sh %s > %s\n" \
      "$ra" "$dir/traces_$ruolo.csv" "$(printf '%q ' "$@")" "$topic" "$dir/mola_$ruolo.log"; return 0; fi
  # without DDS_PROFILE MOLA keeps the factory transport; with it, the chain profile
  local dds=(-u FASTRTPS_DEFAULT_PROFILES_FILE)
  [[ -n "$DDS_PROFILE" ]] && dds=(FASTRTPS_DEFAULT_PROFILES_FILE="$DDS_PROFILE")
  ( trap - INT; exec env "${dds[@]}" ROS_ARGS="$ra" TRACES="$dir/traces_$ruolo.csv" "$@" \
      setsid bash src/nodes/launch_mola_attack.sh "$topic" ) > "$dir/mola_$ruolo.log" 2>&1 &
  MOLA_PID[$ruolo]=$!
}

mola_attendi_pronte() {   # $1 = folder; waits for readiness in both logs (MolaRunner.wait_ready)
  [[ "$A_SECCO" == "1" ]] && { echo "+ attesa di 'Initial re-localization done' nei due log di MOLA (max 90 s)"; return 0; }
  local t0=$SECONDS r
  for r in victim attacker; do
    until grep -q "Initial re-localization done" "$1/mola_$r.log" 2>/dev/null; do
      kill -0 "${MOLA_PID[$r]}" 2>/dev/null || { nota "IMPREVISTO: MOLA $r terminata ($1/mola_$r.log)"; return 1; }
      (( SECONDS - t0 > 90 )) && { nota "IMPREVISTO: MOLA $r non pronta dopo 90 s"; return 1; }
      sleep 0.5
    done
  done
}

mola_stop() {   # SIGTERM to the group, as MolaRunner.stop/kill_orphans
  [[ "$A_SECCO" == "1" ]] && { echo "+ chiusura delle due MOLA (TERM al gruppo, poi KILL)"; return 0; }
  local r pid
  for r in "${!MOLA_PID[@]}"; do
    pid="${MOLA_PID[$r]}"; [[ -z "$pid" ]] && continue
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  done
  for _ in $(seq 1 15); do pgrep -x mola-cli >/dev/null || break; sleep 1; done
  for r in "${!MOLA_PID[@]}"; do pid="${MOLA_PID[$r]}"; kill -KILL -- "-$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; done
  MOLA_PID=()
  pgrep -x mola-cli >/dev/null && nota "IMPREVISTO: mola-cli ancora vivo dopo la chiusura"
  return 0
}

qos_grafo() {   # $1 = folder, $2 = label: ros2 topic info -v on the test topics
  local t f
  for t in /carter/lidar_victim /carter/lidar_attacker /lidar_odometry/pose /victim/lidar_odometry/pose /attacker/lidar_odometry/pose; do
    f="$1/topic_info_$2$(echo "$t" | tr '/' '_').txt"
    esegui "ros2 topic info -v --no-daemon --spin-time 3 $t > $f 2>&1"
  done
}

ambiente_su_file() {   # $1 = file
  [[ "$A_SECCO" == "1" ]] && { echo "+ ambiente RMW/ROS/MOLA, eta' dell'adattatore, hash del codice e versioni dell'interprete del nodo in $1"; return 0; }
  local pid; pid=$(pid_adattatore)
  { echo "date: $(date +%FT%T)"; echo "comando: ${COMANDO_PROVA:-}"
    echo "codice: $(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo 'non git'), md5 dei file: $(hash_codice)"
    echo "adattatore: pid ${pid:-assente} eta_s $( [[ -n "$pid" ]] && ps -o etimes= -p "$pid" | tr -dc 0-9)"
    [[ -n "$pid" ]] && tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep -E '^(ROS_|RMW_|FASTRTPS|FASTDDS)' | sed 's/^/  adattatore /'
    env | grep -E '^(ROS_|RMW_|FASTRTPS|FASTDDS|MOLA_)' | sort | sed 's/\n/\\n/g'
    [[ -n "$DDS_PROFILE" ]] && echo "profilo DDS: $DDS_PROFILE md5 $(md5sum < "$DDS_PROFILE" | cut -d' ' -f1)"
    [[ -n "${CONTROLLO_ETA:-}" || "${SONDA:-0}" == "1" || "${MOLA_BINARIO_G1:-0}" == "1" ]] \
      && echo "controlli del 29/10: CONTROLLO_ETA=${CONTROLLO_ETA:-} SONDA=${SONDA:-0} MOLA_BINARIO_G1=${MOLA_BINARIO_G1:-0}"
    # interpreter versions of the perturbation_node
    echo "== interprete del perturbation_node (python3 dal PATH, come run_campaign.sh)"
    python3 "$ROOT/scripts/ambiente_nodo.py" versioni \
      --riferimento "${AMBIENTE_RIFERIMENTO:-$ROOT/config/ambiente_nodo_serieB.txt}" 2>&1
  } > "$1"
}

# Wait for a process to exit by itself (max $2 s), then TERM and a second wait.
# Needed with run_campaign.sh: a TERM during its shutdown would interrupt it.
attendi_poi_term() {   # $1 = pid, $2 = seconds
  local pid="$1" max="${2:-130}" t0=$SECONDS
  [[ -z "$pid" ]] && return 0
  while kill -0 "$pid" 2>/dev/null && (( SECONDS - t0 < max )); do command sleep 1; done
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null
    t0=$SECONDS
    while kill -0 "$pid" 2>/dev/null && (( SECONDS - t0 < max )); do command sleep 1; done
  fi
  wait "$pid" 2>/dev/null || true
  ! kill -0 "$pid" 2>/dev/null
}

# MOLA (mola-cli) carrying this test's tag in the environment: closed by pid at the end
# of shutdown, after the campaign. A mola-cli without the tag, or with the tag of another
# test, is left alone.
pids_mola_della_prova() {   # $1 = value of PROVA_BANCO_ID
  local p
  for p in $(pgrep -x mola-cli 2>/dev/null); do
    tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qx "PROVA_BANCO_ID=$1" && echo "$p"
  done
}
chiudi_mola_della_prova() {
  [[ "$PROVA_MIA" == "1" && -n "${PROVA_BANCO_ID:-}" ]] || return 0
  local resti p
  resti="$(pids_mola_della_prova "$PROVA_BANCO_ID" | tr '\n' ' ')"
  [[ -z "${resti// }" ]] && return 0
  nota "IMPREVISTO: MOLA della prova ancora vive dopo la chiusura (pid $resti): TERM, poi KILL dopo 15 s"
  for p in $resti; do kill -TERM "$p" 2>/dev/null; done
  for _ in $(seq 1 15); do
    resti="$(pids_mola_della_prova "$PROVA_BANCO_ID" | tr '\n' ' ')"; [[ -z "${resti// }" ]] && break; command sleep 1
  done
  for p in $resti; do kill -KILL "$p" 2>/dev/null; done
  PROVA_NOTE="${PROVA_NOTE:+$PROVA_NOTE; }MOLA chiuse dalla chiusura della prova"
}
# MOLA left by a test that ended without its shutdown (watchdog KILL): list and command
mola_orfane_di_prove() {
  local p id
  for p in $(pgrep -x mola-cli 2>/dev/null); do
    id=$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | sed -n 's/^PROVA_BANCO_ID=//p')
    [[ -n "$id" ]] && pid_vivo_con_avvio "${id%%:*}" "${id#*:}" && continue
    [[ -n "$id" ]] && echo "    mola-cli $p della prova ${id%%:*} (non piu' viva): kill -TERM $p"
  done
}
