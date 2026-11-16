#!/usr/bin/env bash
set -euo pipefail
# Read-only diagnostics, usable before creating the new environment.
cat /etc/os-release
"$(command -v python3 || command -v python)" --version
nvidia-smi || true
nproc
for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/cpu.max; do
  if [[ -f "$f" ]]; then
    printf '\n%s\n' "$f"
    cat "$f"
  fi
done
DATA_DISK="${TP_DATA:-$HOME}"
while [[ ! -e "$DATA_DISK" ]]; do DATA_DISK=$(dirname "$DATA_DISK"); done
df -h "$HOME" "$DATA_DISK"
command -v conda || true
command -v docker || true
