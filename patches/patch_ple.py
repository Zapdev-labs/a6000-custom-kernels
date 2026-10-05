#!/usr/bin/env python3
"""PLE island fix (EXPERIMENT - measured as a net loss, kept for the record):
  1. offload gate claims GET_ROWS when its weight tensor is (or can be) pinned
  2. supports_buft accepts any host buffer type so the scheduler stops staging
     host weights into VRAM for claimed ops

Result at 262144 ctx: pp512 92.3 / tg128 45.2 vs 102.7 / 48.2 without it.
Revert with patch_ple_revert.py.

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

# 1. claim GET_ROWS (pin the PLE table on first claim)
patch(CUDA,
'''    // Hot-expert cache: claim all MoE GEMVs. Decode batches run through the
    // pointer-table kernel (hot experts in VRAM, cold ones streamed from pinned host
    // RAM over PCIe); prefill batches use the standard MMQ path, which reads the
    // pinned host experts zero-copy over PCIe - both far faster than the CPU path.
    if (op->op == GGML_OP_MUL_MAT_ID && ggml_cuda_moe_cache_enabled()) {
        return true;
    }
''',
'''    // Hot-expert cache: claim all MoE GEMVs. Decode batches run through the
    // pointer-table kernel (hot experts in VRAM, cold ones streamed from pinned host
    // RAM over PCIe); prefill batches use the standard MMQ path, which reads the
    // pinned host experts zero-copy over PCIe - both far faster than the CPU path.
    if (op->op == GGML_OP_MUL_MAT_ID && ggml_cuda_moe_cache_enabled()) {
        return true;
    }

    // Keep PLE n-gram lookups (GET_ROWS over the huge host-side per-layer embedding
    // table) on the GPU: only a few KB is read per step over PCIe, while a CPU island
    // forces a CUDA|CPU|CUDA split with sync stalls on every decode step. The weight
    // tensor is pinned on first claim, which makes it GPU-readable.
    if (op->op == GGML_OP_GET_ROWS && ggml_cuda_moe_cache_enabled()) {
        return ggml_cuda_moe_cache_pin_host(op->src[0]);
    }
''')

# 2. accept all host buffer types (claimed ops only read pinned memory)
patch(CUDA,
'''    // Hot-expert cache: allow direct (zero-copy) reads of pinned CUDA_Host buffers on
    // discrete GPUs. Without this the scheduler materializes full copies of host weights
    // in VRAM for every op claimed by the CUDA backend.
    if (ggml_cuda_moe_cache_enabled() && ggml_backend_buft_is_cuda_host(buft)) {
        return true;
    }
''',
'''    // Hot-expert cache: allow direct (zero-copy) reads of host buffers on discrete
    // GPUs. Without this the scheduler materializes full copies of host weights in
    // VRAM for every op claimed by the CUDA backend. Only ops that pass the offload
    // gate run here, and the gate guarantees their host tensors are pinned.
    if (ggml_cuda_moe_cache_enabled() && ggml_backend_buft_is_host(buft)) {
        return true;
    }
''')

print("PLE island patches applied")
