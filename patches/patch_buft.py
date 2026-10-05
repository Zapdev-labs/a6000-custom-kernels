#!/usr/bin/env python3
"""Add GET_ROWS claim to the MoE-cache offload gate (patches the already-flipped gate).

Historical: this naive version stages the 28.8GB PLE table in VRAM and OOMs the
expert pools; patch_ple.py is the refined (pin-on-claim) successor.
Set LLAMA_CPP_DIR (default /root/llama.cpp) to target another checkout.
"""
import os

LLAMA_CPP_DIR = os.environ.get("LLAMA_CPP_DIR", "/root/llama.cpp")
CUDA = os.path.join(LLAMA_CPP_DIR, "ggml/src/ggml-cuda/ggml-cuda.cu")

def patch(path, old, new, count=1):
    with open(path) as f:
        src = f.read()
    n = src.count(old)
    assert n == count, f"{path}: expected {count} occurrence(s), found {n}:\n{old[:200]}"
    with open(path, "w") as f:
        f.write(src.replace(old, new))
    print(f"patched {path}")

patch(CUDA,
'''    if (op->op == GGML_OP_MUL_MAT_ID && ggml_cuda_moe_cache_enabled()) {
        return true;
    }
''',
'''    if (op->op == GGML_OP_MUL_MAT_ID && ggml_cuda_moe_cache_enabled()) {
        return true;
    }

    // Keep PLE n-gram lookups (GET_ROWS over the huge host-side per-layer embedding
    // table) on the GPU as well: they read only a few KB per step over PCIe, while
    // a CPU island forces a CUDA|CPU|CUDA split with sync stalls every decode step.
    if (op->op == GGML_OP_GET_ROWS && ggml_cuda_moe_cache_enabled()) {
        return true;
    }
''')

print("buft patch applied")
