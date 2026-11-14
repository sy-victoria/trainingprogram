#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SIZE="${1:-0.8B}"
MODE="${2:-smoke}"
case "$SIZE" in 0.8B|2B|4B) ;; *) printf 'Invalid size\n' >&2; exit 1;; esac
case "$MODE" in smoke|subset|full) ;; *) printf 'Invalid mode\n' >&2; exit 1;; esac
export TP_DATA="${TP_DATA:-/root/autodl-tmp/trainingprogram-data}"
mkdir -p "$TP_DATA/diagnostics"
tp model prepare --size "$SIZE"
for VARIANT in BF16 Q4_K_M; do
  RUN="qwen35-${SIZE}-${VARIANT}-${MODE}"
  tp model serve --size "$SIZE" --variant "$VARIANT" > "$TP_DATA/diagnostics/server-${SIZE}-${VARIANT}.txt" 2>&1 &
  SERVER_PID=$!
  trap 'kill -TERM "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT
  python scripts/wait_server.py --pid "$SERVER_PID"
  tp eval generate --run "$RUN" --size "$SIZE" --variant "$VARIANT" --mode "$MODE"
  if [[ ! -f "$TP_DATA/runs/$RUN/speed.json" ]]; then
    tp eval speed --run "$RUN"
  fi
  kill -TERM "$SERVER_PID"
  wait "$SERVER_PID" || true
  trap - EXIT
done
printf '\nGeneration complete. Judge separately with tp eval judge --run RUN_ID\n'
