#!/usr/bin/env python3
"""Revert the PLE island experiment (net loss: pp 110->92, tg 50.5->45.2)."""

import os

LLAMA_CPP_DIR = os.environ.get("LLAMA_CPP_DIR") or os.path.expanduser("~/llama.cpp")
CUDA = os.path.join(LLAMA_CPP_DIR, "ggml/src/ggml-cuda/ggml-cuda.cu")

def patch(path, old, new, count=1):
    with open(path) as f:
        src = f.read()
    n = src.count(old)
    assert n == count, f"{path}: expected {count} occurrence(s), found {n}:\n{old[:200]}"
    with open(path, "w") as f:
        f.write(src.replace(old, new))
    print(f"patched {path}")

# 1. drop the GET_ROWS claim
patch(CUDA,
'''
    // Keep PLE n-gram lookups (GET_ROWS over the huge host-side per-layer embedding
    // table) on the GPU: only a few KB is read per step over PCIe, while a CPU island
    // forces a CUDA|CPU|CUDA split with sync stalls on every decode step. The weight
    // tensor is pinned on first claim, which makes it GPU-readable.
    if (op->op == GGML_OP_GET_ROWS && ggml_cuda_moe_cache_enabled()) {
        return ggml_cuda_moe_cache_pin_host(op->src[0]);
    }
''', '')

# 2. restrict supports_buft back to pinned CUDA_Host buffers
patch(CUDA,
'''    // Hot-expert cache: allow direct (zero-copy) reads of host buffers on discrete
    // GPUs. Without this the scheduler materializes full copies of host weights in
    // VRAM for every op claimed by the CUDA backend. Only ops that pass the offload
    // gate run here, and the gate guarantees their host tensors are pinned.
    if (ggml_cuda_moe_cache_enabled() && ggml_backend_buft_is_host(buft)) {
        return true;
    }
''',
'''    // Hot-expert cache: allow direct (zero-copy) reads of pinned CUDA_Host buffers on
    // discrete GPUs. Without this the scheduler materializes full copies of host weights
    // in VRAM for every op claimed by the CUDA backend.
    if (ggml_cuda_moe_cache_enabled() && ggml_backend_buft_is_cuda_host(buft)) {
        return true;
    }
''')

print("PLE patches reverted")
