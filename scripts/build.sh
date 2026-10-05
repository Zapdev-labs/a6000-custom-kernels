#!/usr/bin/env bash
# Clone llama.cpp at the pinned commit, install the custom kernels, apply the
# scheduler patches, and build llama-cli + llama-bench for SM86 (RTX A6000).
#
# Environment overrides:
#   LLAMA_CPP_DIR  target tree (default ~/llama.cpp)
#   CUDA_ARCH      CMAKE_CUDA_ARCHITECTURES (default 86 = A6000/3090)
#   JOBS           parallel build jobs (default: nproc)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
COMMIT="11fe0215"   # upstream commit every patch anchor is written against
CUDA_ARCH="${CUDA_ARCH:-86}"
JOBS="${JOBS:-$(nproc)}"

# 1. clone + checkout (refuse to clobber a dirty tree)
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    git clone https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR"
fi
cd "$LLAMA_CPP_DIR"
if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: $LLAMA_CPP_DIR has local changes; clean it or point LLAMA_CPP_DIR elsewhere." >&2
    exit 1
fi
git checkout --detach "$COMMIT"

# 2. install the kernels
cp "$REPO_ROOT/kernels/moe-cache.cu" "$REPO_ROOT/kernels/moe-cache.cuh" "$REPO_ROOT/kernels/ssm-conv.cu" \
   ggml/src/ggml-cuda/

# 3. apply the patches (base wire-in, then final-state flips)
python3 "$REPO_ROOT/patches/apply_patches.py"
python3 "$REPO_ROOT/patches/patch_final_state.py"

# 4. build (tests/examples off keeps the build to ~15 min on 60 cores)
command -v nvcc >/dev/null 2>&1 || export PATH="/usr/local/cuda/bin:$PATH"
cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF
cmake --build build -j "$JOBS" --target llama-cli llama-bench

echo
echo "Built: $LLAMA_CPP_DIR/build/bin/llama-bench"
echo "Next: MODEL=/path/to/model.gguf scripts/bench.sh"
