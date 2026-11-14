#!/usr/bin/env bash
set -euo pipefail
# Read-only diagnostics, usable before creating the new environment.
cat /etc/os-release
python --version
nvidia-smi
nproc
for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/cpu.max; do
  if [[ -f "$f" ]]; then
    printf '\n%s\n' "$f"
    cat "$f"
  fi
done
df -h /root /root/autodl-tmp
command -v conda || true
command -v docker || true
