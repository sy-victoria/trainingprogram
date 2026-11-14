#!/usr/bin/env bash
set -euo pipefail
LLAMA_DIR="$1"
test -x "$CONDA_PREFIX/bin/nvcc"
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CUDA_COMPILER="$CONDA_PREFIX/bin/nvcc" -DCUDAToolkit_ROOT="$CONDA_PREFIX" \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build "$LLAMA_DIR/build" --config Release --target llama-server llama-quantize -j 8
"$LLAMA_DIR/build/bin/llama-server" --version
