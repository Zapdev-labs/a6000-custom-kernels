# Shared MoE-cache environment for the winning configuration (-ncmoe 8 +
# seeded hot-expert cache). Sourced by bench.sh and chat.sh; expects
# REPO_ROOT to be set by the caller.
# shellcheck shell=bash

export GGML_CUDA_MOE_CACHE=1
export GGML_CUDA_MOE_CACHE_SLOTS="${MOE_CACHE_SLOTS:-320}"
export GGML_CUDA_MOE_CACHE_SEED="${SEED:-$REPO_ROOT/assets/moe_seed.txt}"
