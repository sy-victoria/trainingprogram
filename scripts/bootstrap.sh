#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
TP_DATA="${TP_DATA:-/root/autodl-tmp/trainingprogram-data}"
export TP_DATA
# AutoDL images may configure a lagging mirror in pip.conf. Override it per
# invocation without changing the user's global pip configuration.
TP_PIP_INDEX_URL="${TP_PIP_INDEX_URL:-https://pypi.org/simple}"
CONDA_EXE="${CONDA_EXE:-/root/miniconda3/bin/conda}"
if [[ ! -x "$CONDA_EXE" ]]; then
  printf 'Conda not found. Set CONDA_EXE to its absolute path.\n' >&2
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python - <<'PY'
import shutil
import subprocess
driver = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).splitlines()[0]
if tuple(int(x) for x in driver.split('.')[:2]) < (570, 26):
    raise SystemExit('This script targets driver >=570.26 / CUDA 12.8. Do not alter the host driver from the container.')
for path, gib in [('/root', 15), ('/root/autodl-tmp', 30)]:
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
# nvcc in the original image is 11.3; build llama.cpp with an independent CUDA 12.8 toolchain.
if ! "$CONDA_EXE" run -n tp-tools python --version >/dev/null 2>&1; then
  "$CONDA_EXE" create -y -n tp-tools python=3.11 pip
fi
"$CONDA_EXE" install -y -n tp-tools -c nvidia cuda-toolkit=12.8
"$CONDA_EXE" run --no-capture-output -n tp-tools python -m pip install --index-url "$TP_PIP_INDEX_URL" cmake==3.31.6 ninja==1.11.1.3
LLAMA_DIR="$TP_DATA/tools/llama.cpp"
LLAMA_REV=$(python -c 'import json; print(json.load(open("configs/versions.json"))["llama_cpp"])')
if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi
git -C "$LLAMA_DIR" fetch origin "$LLAMA_REV"
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
