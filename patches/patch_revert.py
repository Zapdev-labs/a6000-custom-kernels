#!/usr/bin/env python3
"""Revert the GET_ROWS claim: the 28.8GB PLE table is not pinned, so claiming it
made the scheduler stage the whole table in VRAM, starving the expert pools.

Historical: superseded by patch_ple.py + patch_ple_revert.py.
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
'''
    // Keep PLE n-gram lookups (GET_ROWS over the huge host-side per-layer embedding
    // table) on the GPU as well: they read only a few KB per step over PCIe, while
    // a CPU island forces a CUDA|CPU|CUDA split with sync stalls every decode step.
    if (op->op == GGML_OP_GET_ROWS && ggml_cuda_moe_cache_enabled()) {
        return true;
    }
''', '')

print("GET_ROWS claim reverted")
