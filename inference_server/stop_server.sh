#!/usr/bin/env bash
set -euo pipefail

pid_file=/data/robot_project/logs/pi_server.pid
if [[ ! -f "$pid_file" ]]; then
  echo "No PID file at $pid_file"
  exit 0
fi

pid="$(cat "$pid_file" || true)"
if [[ -z "$pid" ]]; then
  echo "PID file is empty"
  exit 0
fi

if kill -0 "$pid" 2>/dev/null; then
  kill "$pid"
  echo "Stopped SO101 policy server PID $pid"
else
  echo "PID $pid is not running"
fi
