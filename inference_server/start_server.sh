#!/usr/bin/env bash
set -euo pipefail

mkdir -p /data/robot_project/logs
cd /data/robot_project/inference_server

pid_file=/data/robot_project/logs/pi_server.pid
log_file=/data/robot_project/logs/pi_server.log

if [[ -f "$pid_file" ]]; then
  old_pid="$(cat "$pid_file" || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Server already running with PID $old_pid"
    exit 0
  fi
fi

nohup bash /data/robot_project/inference_server/run_server.sh \
  > "$log_file" \
  2>&1 &
pid="$!"
echo "$pid" > "$pid_file"
echo "Started SO101 policy server PID $pid"
echo "Log: $log_file"
