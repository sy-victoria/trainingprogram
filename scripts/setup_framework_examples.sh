#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${TP_DATA:?Set TP_DATA to the experiment data directory}"
TP_AGENT_PYTHON="${TP_AGENT_PYTHON:-python}"
TP_FRAMEWORK_ENVS="${TP_FRAMEWORK_ENVS:-$TP_DATA/tools/framework-envs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$TP_DATA/cache/pip}"
mkdir -p "$TP_FRAMEWORK_ENVS" "$TP_DATA/diagnostics"
for framework in qwen deepagents; do
  prefix="$TP_FRAMEWORK_ENVS/$framework"
  if [[ ! -x "$prefix/bin/python" ]]; then
    "$TP_AGENT_PYTHON" -m venv --copies "$prefix"
  fi
  case "$framework" in
    qwen) requirements=requirements/qwen-example.txt ;;
    deepagents) requirements=requirements/deepagents-example.txt ;;
  esac
  "$prefix/bin/python" -m pip install --index-url "${TP_PIP_INDEX_URL:-https://pypi.org/simple}" -r "$requirements"
  "$prefix/bin/python" -m pip check
  "$prefix/bin/python" -m pip freeze > "$TP_DATA/diagnostics/$framework-example-resolved.txt"
done
echo FRAMEWORK_ENVIRONMENTS_READY
