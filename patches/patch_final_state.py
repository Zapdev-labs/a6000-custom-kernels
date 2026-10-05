#!/usr/bin/env python3
"""Final-state patches applied AFTER apply_patches.py on a fresh clone:
  1. offload gate: claim ALL MoE GEMV batches (not just decode-sized)
  2. supports_buft: accept pinned CUDA_Host buffers on discrete GPUs

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

# 1. flip the offload gate to claim all batches
patch(CUDA,
'''    // Hot-expert cache: claim decode-sized MoE GEMVs (batch <= MMVQ_MAX_BATCH_SIZE) so
    // they run on the GPU through the pointer-table kernel. Larger (prefill) batches stay
    // on the CPU where batched GEMM over host RAM is faster than PCIe streaming.
    if (op->op == GGML_OP_MUL_MAT_ID && ggml_cuda_moe_cache_enabled()) {
        return get_op_batch_size(op) <= MMVQ_MAX_BATCH_SIZE;
    }
''',
'''    // Hot-expert cache: claim all MoE GEMVs. Decode batches run through the
    // pointer-table kernel (hot experts in VRAM, cold ones streamed from pinned host
    // RAM over PCIe); prefill batches use the standard MMQ path, which reads the
    // pinned host experts zero-copy over PCIe - both far faster than the CPU path.
    if (op->op == GGML_OP_MUL_MAT_ID && ggml_cuda_moe_cache_enabled()) {
        return true;
    }
''')

# 2. accept pinned CUDA_Host buffers on discrete GPUs
patch(CUDA,
'''static bool ggml_backend_cuda_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;
    const bool integrated = ggml_cuda_info().devices[dev_ctx->device].integrated;
    return (ggml_backend_buft_is_cuda(buft) && buft->device == dev) || (integrated && ggml_backend_buft_is_cuda_host(buft));
}''',
'''static bool ggml_backend_cuda_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;
    const bool integrated = ggml_cuda_info().devices[dev_ctx->device].integrated;

    // Hot-expert cache: allow direct (zero-copy) reads of pinned CUDA_Host buffers on
    // discrete GPUs. Without this the scheduler materializes full copies of host weights
    // in VRAM for every op claimed by the CUDA backend.
    if (ggml_cuda_moe_cache_enabled() && ggml_backend_buft_is_cuda_host(buft)) {
        return true;
    }

    return (ggml_backend_buft_is_cuda(buft) && buft->device == dev) || (integrated && ggml_backend_buft_is_cuda_host(buft));
}''')

print("final-state patches applied")
