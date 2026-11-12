#!/usr/bin/env bash
# Chain status: process count per type (invoke by path, so that no process name appears on the
# command line).
for pat in 'scripts/run_campaign.sh' 'attack_orchestrator.py' 'run_isaac_standalone' 'nodes/perturbation_node' 'mola-cli' 'hop_recorder' 'gate_replay' 'adapter_watch_prog' 'rettilineo.sh' 'dopo_rettilineo'; do
  n=$(pgrep -f "$pat" | grep -v "^$$\$" | wc -l)
  printf "%-28s %s\n" "$pat" "$n"
done
echo "adattatore (python): $(ps -eo pid,cmd | awk '$2=="python3" && $3=="-u" && $4 ~ /add_intensity_node.py$/' | wc -l)"
