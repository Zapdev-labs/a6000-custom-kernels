#!/usr/bin/env bash
# Winning benchmark configuration: CYBER-FROST-3.8 on a single 48GB RTX A6000
# at the model's full 262,144-token context. Reproduces pp512 ~167 / tg128 ~60 t/s.
#
#   MODEL=/path/to/CYBER-FROST-3.8-Q2_K_S.gguf scripts/bench.sh
#
# Knobs:
#   N_CMOE  host-side expert layers (default 8, best joint operating point;
#           4 favors prefill ~300 t/s at ~42 decode, 18 is the conservative end)
#   CTX     context size (default 262144)
#   SEED    seed file override; MOE_CACHE_SLOTS overrides the pool size
set -euo pipefail

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
MODEL="${MODEL:?set MODEL=/path/to/CYBER-FROST-3.8-Q2_K_S.gguf}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
N_CMOE="${N_CMOE:-8}"
CTX="${CTX:-262144}"

. "$REPO_ROOT/scripts/moe-env.sh"

# -lm none is REQUIRED: mmap (-lm auto) gives expert tensors plain CPU_Mapped
# buffers instead of pinned CUDA_Host buffers, which breaks zero-copy GPU reads
# and collapses throughput to ~5 t/s. See README "The -lm none requirement".
exec "$LLAMA_CPP_DIR/build/bin/llama-bench" \
    -m "$MODEL" \
    -p 512 -n 128 \
    -ngl 99 -ncmoe "$N_CMOE" -fa on -c "$CTX" -lm none \
    -r 1
