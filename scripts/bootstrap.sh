#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE="${1:---all}"
case "$STAGE" in
  --agent-only|--all) ;;
  *) printf 'Usage: bash scripts/bootstrap.sh [--agent-only|--all]\n' >&2; exit 2 ;;
esac
TP_DATA="${TP_DATA:-/root/autodl-tmp/trainingprogram-data}"
export TP_DATA
# AutoDL images may configure a lagging mirror in pip.conf. Override it per
# invocation without changing the user's global pip configuration.
TP_PIP_INDEX_URL="${TP_PIP_INDEX_URL:-https://pypi.org/simple}"
CONDA_EXE="${CONDA_EXE:-$(command -v conda || true)}"
CONDA_EXE="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"
export CONDA_EXE
if [[ ! -x "$CONDA_EXE" ]]; then
  printf 'Conda not found. Set CONDA_EXE to its absolute path.\n' >&2
  exit 1
fi
if [[ "$STAGE" == "--all" ]]; then
  if ! nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; then
    printf 'No GPU. Use --agent-only to prepare Python and run offline checks, or start the GPU instance for --all.\n' >&2
    exit 1
  fi
fi
TP_BOOTSTRAP_STAGE="$STAGE" python - <<'PY'
import shutil
import subprocess
import os
from pathlib import Path
full = os.environ['TP_BOOTSTRAP_STAGE'] == '--all'
if full:
    driver = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).splitlines()[0]
    if tuple(int(x) for x in driver.split('.')[:2]) < (570, 26):
        raise SystemExit('This script targets driver >=570.26 / CUDA 12.8. Do not alter the host driver from the container.')
    limit = Path('/sys/fs/cgroup/memory.max')
    if limit.exists() and limit.read_text().strip() != 'max' and int(limit.read_text()) < 8 * 1024**3:
        raise SystemExit('At least 8GiB container memory is required for the GPU build. Use --agent-only in no-GPU mode.')
conda_root = Path(os.environ['CONDA_EXE']).resolve().parent.parent
data_path = Path(os.environ['TP_DATA']).expanduser().resolve()
while not data_path.exists():
    data_path = data_path.parent
for path, gib in ([(conda_root, 15), (data_path, 30)] if full else [(conda_root, 3)]):
    if shutil.disk_usage(path).free < gib * 1024**3:
        raise SystemExit(f'Need {gib} GiB free at {path}; inspect existing files before retrying')
PY
mkdir -p "$TP_DATA/tools" "$TP_DATA/diagnostics"
if ! "$CONDA_EXE" run -n tp-agent python --version >/dev/null 2>&1; then
  "$CONDA_EXE" create -y -n tp-agent python=3.11 pip
fi
"$CONDA_EXE" run --no-capture-output -n tp-agent python -m pip install --index-url "$TP_PIP_INDEX_URL" -r requirements/agent.lock
"$CONDA_EXE" run --no-capture-output -n tp-agent python -m pip install --index-url "$TP_PIP_INDEX_URL" --no-deps --no-build-isolation -e .
"$CONDA_EXE" run --no-capture-output -n tp-agent python -m pip check
"$CONDA_EXE" run --no-capture-output -n tp-agent tp --help
"$CONDA_EXE" run -n tp-agent python -m pip freeze > "$TP_DATA/diagnostics/agent-resolved.txt"
if [[ "$STAGE" == "--agent-only" ]]; then
  printf '\nAgent environment ready (GPU tools not installed). Run: conda activate tp-agent; tp check --stage agent; tp selftest\n'
  exit 0
fi
# nvcc in the original image is 11.3; build llama.cpp with an independent CUDA 12.8 toolchain.
if ! "$CONDA_EXE" run -n tp-tools python --version >/dev/null 2>&1; then
  "$CONDA_EXE" create -y -n tp-tools python=3.11 pip
fi
"$CONDA_EXE" install -y -n tp-tools -c nvidia cuda-toolkit=12.8
"$CONDA_EXE" run --no-capture-output -n tp-tools python -m pip install --index-url "$TP_PIP_INDEX_URL" cmake==3.31.6 ninja==1.11.1.3
LLAMA_DIR="$TP_DATA/tools/llama.cpp"
LLAMA_REV=$(python -c 'import json; print(json.load(open("configs/versions.json"))["llama_cpp"])')
if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  git init "$LLAMA_DIR"
  git -C "$LLAMA_DIR" remote add origin https://github.com/ggml-org/llama.cpp.git
fi
if [[ "$(git -C "$LLAMA_DIR" rev-parse HEAD 2>/dev/null || true)" != "$LLAMA_REV" ]]; then
  for attempt in 1 2 3; do
    if git -c http.version=HTTP/1.1 -c http.lowSpeedLimit=1 -c http.lowSpeedTime=120 \
        -C "$LLAMA_DIR" fetch --depth 1 origin "$LLAMA_REV"; then
      break
    fi
    if [[ "$attempt" == 3 ]]; then
      printf 'Pinned llama.cpp source download failed after 3 attempts. Rerun to resume preparation.\n' >&2
      exit 1
    fi
    sleep 5
  done
fi
git -C "$LLAMA_DIR" checkout --detach "$LLAMA_REV"
# Upstream conversion dependencies are tied to the pinned source. Actual resolved versions are archived.
"$CONDA_EXE" run --no-capture-output -n tp-tools python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
"$CONDA_EXE" run --no-capture-output -n tp-tools python -m pip install --index-url "$TP_PIP_INDEX_URL" -r "$LLAMA_DIR/requirements/requirements-convert_hf_to_gguf.txt"
"$CONDA_EXE" run --no-capture-output -n tp-tools python -m pip install --index-url "$TP_PIP_INDEX_URL" -e "$LLAMA_DIR/gguf-py"
"$CONDA_EXE" run --no-capture-output -n tp-tools bash scripts/build_llama.sh "$LLAMA_DIR"
"$CONDA_EXE" run -n tp-tools python -m pip freeze > "$TP_DATA/diagnostics/converter-resolved.txt"
"$CONDA_EXE" list -n tp-tools --explicit > "$TP_DATA/diagnostics/tools-conda-explicit.txt"
"$CONDA_EXE" run -n tp-agent python -m pip freeze > "$TP_DATA/diagnostics/agent-resolved.txt"
printf '\nEnvironment ready. Run: conda activate tp-agent\n'
