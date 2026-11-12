#!/usr/bin/env bash
# Negotiated QoS of the MOLA subscriptions on the victim and attacker streams,
# with the chain running (after the continuous victim has started).
#
# Runs "ros2 topic info -v --no-daemon" on /carter/lidar_victim and
# /carter/lidar_attacker and exits with 1 if a topic has no MOLA subscription
# (node mola_bridge_ros2) or if any of them is not RELIABLE; 0 if all are
# RELIABLE. Run it with the same environment as the chain
# (ROS_AUTOMATIC_DISCOVERY_RANGE included). The full ros2 output is kept
# in $OUT_DIR (default /tmp).
#
# Usage:
#   scripts/controlla_qos_grafo.sh
#   TOPICS="/carter/lidar_victim" SPIN_TIME=5 scripts/controlla_qos_grafo.sh
set -uo pipefail

TOPICS="${TOPICS:-/carter/lidar_victim /carter/lidar_attacker}"
SPIN_TIME="${SPIN_TIME:-3}"
OUT_DIR="${OUT_DIR:-/tmp}"
EXPECT="${EXPECT:-RELIABLE}"

rc=0
for t in $TOPICS; do
  f="$OUT_DIR/qos_grafo_$(echo "$t" | tr '/' '_')_$(date +%Y%m%d_%H%M%S).txt"
  if ! ros2 topic info -v --no-daemon --spin-time "$SPIN_TIME" "$t" > "$f" 2>&1; then
    echo "$t: ros2 topic info fallito (uscita in $f)"
    rc=1
    continue
  fi
  # "Node name: ..." blocks up to the empty line; keeps the subscriptions
  # of the mola_bridge_ros2 nodes with namespace and Reliability.
  res="$(awk -v want="$EXPECT" '
    /^Node name:/      { node=$3 }
    /^Node namespace:/ { ns=$3 }
    /^Endpoint type:/  { ep=$3 }
    /^  Reliability:/  { if (ep == "SUBSCRIPTION" && node ~ /mola_bridge_ros2/) {
                           n++; printf "%s%s/%s=%s", (n > 1 ? " " : ""), ns, node, $2
                           if ($2 != want) bad++ } }
    END { printf "|%d|%d\n", n, bad }' "$f")"
  list="${res%%|*}"; rest="${res#*|}"; n="${rest%%|*}"; bad="${rest#*|}"
  if [[ "$n" -eq 0 ]]; then
    echo "$t: nessuna subscription di MOLA (uscita in $f)"
    rc=1
  elif [[ "$bad" -gt 0 ]]; then
    echo "$t: subscription di MOLA non $EXPECT: $list (uscita in $f)"
    rc=1
  else
    echo "$t: $list"
  fi
done
exit $rc
