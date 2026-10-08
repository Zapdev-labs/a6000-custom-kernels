#!/usr/bin/env bash
# Interactive chat with the same winning configuration as bench.sh.
#
#   MODEL=/path/to/CYBER-FROST-3.8-Q2_K_S.gguf scripts/chat.sh
#
set -euo pipefail

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
MODEL="${MODEL:?set MODEL=/path/to/CYBER-FROST-3.8-Q2_K_S.gguf}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
N_CMOE="${N_CMOE:-8}"
CTX="${CTX:-262144}"

. "$REPO_ROOT/scripts/moe-env.sh"

# -lm none is REQUIRED (see scripts/bench.sh / README).
exec "$LLAMA_CPP_DIR/build/bin/llama-cli" \
    -m "$MODEL" \
    -ngl 99 -ncmoe "$N_CMOE" -fa on -c "$CTX" -lm none \
    --color -i \
    "$@"
