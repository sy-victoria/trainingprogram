#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
docker info >/dev/null
docker build -f docker/Dockerfile -t trainingprogram-eval:0.1 .
docker image inspect --format '{{.Id}}' trainingprogram-eval:0.1
