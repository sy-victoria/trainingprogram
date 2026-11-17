#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${TP_DATA:?Set TP_DATA to the experiment data directory}"
TP_AGENT_PYTHON="${TP_AGENT_PYTHON:-python}"
TP_JUDGE_PREFIX="${TP_JUDGE_PREFIX:-$TP_DATA/tools/judge-env}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$TP_DATA/cache/pip}"
command -v bwrap >/dev/null
command -v systemd-run >/dev/null
REV=$("$TP_AGENT_PYTHON" -c 'import json; print(json.load(open("configs/versions.json"))["evalplus"])')
mkdir -p "$TP_DATA/tools/sources" "$TP_DATA/diagnostics"
if [[ ! -x "$TP_JUDGE_PREFIX/bin/python" ]]; then
  "$TP_AGENT_PYTHON" -m venv --copies "$TP_JUDGE_PREFIX"
fi
"$TP_JUDGE_PREFIX/bin/python" -m pip install --index-url "${TP_PIP_INDEX_URL:-https://pypi.org/simple}" numpy==2.2.6 psutil==7.2.2 setuptools==80.9.0 setuptools-scm==8.3.1 wget==3.2 appdirs==1.4.4 tempdir==0.7.1 datasets==5.2.0
ARCHIVE="$TP_DATA/tools/sources/evalplus-$REV.tar.gz"
curl --http1.1 -fL --retry 3 --connect-timeout 15 --max-time 300 -o "$ARCHIVE.partial" "https://codeload.github.com/evalplus/evalplus/tar.gz/$REV"
mv "$ARCHIVE.partial" "$ARCHIVE"
SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0+g$REV" "$TP_JUDGE_PREFIX/bin/python" -m pip install --no-deps --no-build-isolation "$ARCHIVE"
sha256sum "$ARCHIVE" > "$TP_DATA/diagnostics/evalplus-source.sha256"
"$TP_JUDGE_PREFIX/bin/python" -m pip freeze > "$TP_DATA/diagnostics/bubblewrap-resolved.txt"
# /tmp supports Unix ownership required by pytest; only small test fixtures go there.
TMPDIR=/tmp TP_BWRAP_TESTS=1 TP_BWRAP_PYTHON="$TP_JUDGE_PREFIX/bin/python" \
  "$TP_AGENT_PYTHON" -m pytest -q tests/test_integration_bwrap.py
printf 'Validated. Set TP_SANDBOX=bubblewrap and TP_BWRAP_PYTHON=%s/bin/python\n' "$TP_JUDGE_PREFIX"
