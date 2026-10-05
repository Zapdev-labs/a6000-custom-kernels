#pragma once

#include "common.cuh"

#include <cstddef>

// Hot-expert VRAM cache for host-resident MoE experts (custom kernels for the A6000).
//
// Enabled via GGML_CUDA_MOE_CACHE=1. All experts live in pinned host RAM; a
// per-tensor pool of GGML_CUDA_MOE_CACHE_SLOTS (default 320) experts is mirrored
// in VRAM. A device pointer table routes each used expert either to its VRAM
// slot (hot) or straight to host RAM over PCIe (cold, parity bit set).
// A miss recorder + LRU promotion runs between decode steps.

bool ggml_cuda_moe_cache_enabled();

// true when this expert weight tensor should go through the table-based MoE GEMV
bool ggml_cuda_moe_cache_wants(const ggml_tensor * src0);

// fork of the MMVQ MoE decode kernel that sources experts through the pointer table
void ggml_cuda_moe_cache_mul_mat_vec_q(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion, const ggml_cuda_mm_fusion_args_device & fusion_dev,
        const char * src1_q8_1, const int32_t * ids_d, float * dst_d,
        int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
        int64_t ncols_dst, int64_t s01, int64_t s02, int64_t s03,
        int64_t nchannels_y, int64_t nchannels_dst,
        int64_t stride_col_y, int64_t stride_col_dst,
        int64_t stride_channel_y, int64_t stride_channel_dst,
        int64_t ids_stride, cudaStream_t stream);

// LRU promotion pass, called between graph computes (not during capture)
void ggml_cuda_moe_cache_step(cudaStream_t stream);

// true when the cache has been activated by at least one tensor
bool ggml_cuda_moe_cache_active();

// pin a host tensor for zero-copy GPU reads; true if GPU-readable after the call
bool ggml_cuda_moe_cache_pin_host(const ggml_tensor * t);
