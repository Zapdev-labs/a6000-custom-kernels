#!/usr/bin/env python3
"""Wire the hot-expert MoE cache into llama.cpp: mmvq.cu, ggml-cuda.cu.

Requires kernels/moe-cache.cu/.cuh to already be copied into
<LLAMA_CPP_DIR>/ggml/src/ggml-cuda/. Anchors match llama.cpp commit 11fe0215.
Set LLAMA_CPP_DIR (default /root/llama.cpp) to target another checkout.
"""
import os
import sys

LLAMA_CPP_DIR = os.environ.get("LLAMA_CPP_DIR", "/root/llama.cpp")
MMVQ = os.path.join(LLAMA_CPP_DIR, "ggml/src/ggml-cuda/mmvq.cu")
CUDA = os.path.join(LLAMA_CPP_DIR, "ggml/src/ggml-cuda/ggml-cuda.cu")

def patch(path, old, new, count=1):
    with open(path) as f:
        src = f.read()
    n = src.count(old)
    assert n == count, f"{path}: expected {count} occurrence(s) of anchor, found {n}:\n{old[:200]}"
    src = src.replace(old, new)
    with open(path, "w") as f:
        f.write(src)
    print(f"patched {path}")

# ---------------------------------------------------------------------------
# 1) mmvq.cu: include + intercept inside ggml_cuda_mul_mat_vec_q
# ---------------------------------------------------------------------------
patch(MMVQ,
'''#include "mmvq.cuh"
#include "quantize.cuh"''',
'''#include "mmvq.cuh"
#include "moe-cache.cuh"
#include "quantize.cuh"''')

patch(MMVQ,
'''    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    mul_mat_vec_q_switch_type(''',
'''    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    // Hot-expert cache: route host-resident experts through the device pointer table.
    if (ids && ggml_cuda_moe_cache_wants(src0)) {
        ggml_cuda_moe_cache_mul_mat_vec_q(
            ctx, src0, ids, dst, fusion, fusion_local,
            src1_q8_1.get(), ids_d, dst_d,
            ne00, ne01, ne02, ne03,
            ncols_dst, s01, s02, s03,
            nchannels_y, nchannels_dst,
            stride_col_y, stride_col_dst,
            stride_channel_y, stride_channel_dst,
            ids_stride, stream);
        return;
    }

    mul_mat_vec_q_switch_type(''')

# ---------------------------------------------------------------------------
# 2) ggml-cuda.cu: include + offload gate + step hook
# ---------------------------------------------------------------------------
patch(CUDA,
'''#include "ggml-cuda/common.cuh"''',
'''#include "ggml-cuda/common.cuh"
#include "ggml-cuda/moe-cache.cuh"''')

patch(CUDA,
'''static bool ggml_backend_cuda_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    return get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size;
}''',
'''static bool ggml_backend_cuda_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    // Hot-expert cache: claim decode-sized MoE GEMVs (batch <= MMVQ_MAX_BATCH_SIZE) so
    // they run on the GPU through the pointer-table kernel. Larger (prefill) batches stay
    // on the CPU where batched GEMM over host RAM is faster than PCIe streaming.
    if (op->op == GGML_OP_MUL_MAT_ID && ggml_cuda_moe_cache_enabled()) {
        return get_op_batch_size(op) <= MMVQ_MAX_BATCH_SIZE;
    }

    return get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size;
}''')

patch(CUDA,
'''    ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);

    return GGML_STATUS_SUCCESS;''',
'''    ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);

    // LRU promotion of repeatedly-missed experts into the VRAM pool.
    // Runs on the normal stream (capture has ended inside evaluate_and_capture).
    if (ggml_cuda_moe_cache_active()) {
        ggml_cuda_moe_cache_step(cuda_ctx->stream());
    }

    return GGML_STATUS_SUCCESS;''')

print("all patches applied")
