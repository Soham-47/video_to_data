#!/usr/bin/env bash
wait_for_rebuild_disk() {
  local label=$1 base=$2 required_kb=$((9*1024*1024)) available_kb
  for ((i=0; i<240; i++)); do
    available_kb=$(df -Pk "$base" | awk 'NR==2 {print $4}')
    if [[ "$available_kb" =~ ^[0-9]+$ ]] && (( available_kb >= required_kb )); then
      echo "${label}_DISK_GATE_PASSED available_kb=$available_kb"
      return 0
    fi
    if (( i == 239 )); then
      echo "${label}_DISK_GATE_TIMEOUT available_kb=$available_kb required_kb=$required_kb"
      return 1
    fi
    echo "${label}_DISK_GATE_WAIT available_kb=$available_kb required_kb=$required_kb"
    sleep 120
  done
}
