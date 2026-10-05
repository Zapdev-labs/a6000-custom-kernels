#include "moe-cache.cuh"

#include "ggml-cuda.h"
#include "ggml-backend-impl.h"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"
#include "mmvq.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Hot-expert cache for host-resident MoE experts.
//
// Every expert weight tensor that lives in (pinned) host RAM gets:
//   * a device pointer table d_tbl[n_experts]: LSB=1 -> expert stays in host
//     RAM (read over PCIe), LSB=0 -> expert lives in the VRAM pool
//   * a VRAM pool of n_slots experts (staged copies)
//   * a stamp array d_stamps[n_experts] written by the kernel for LRU eviction
// A fused GEMV kernel (fork of mul_mat_vec_q_moe) resolves each routed expert
// through the table. Cold experts are recorded into a small device-side miss
// log; ggml_cuda_moe_cache_step() promotes repeatedly-missed experts into the
// pool between graph computes (never during CUDA graph capture).
// ---------------------------------------------------------------------------

#define MOE_MISS_CAP 8190
#define MOE_MAX_PROMOTIONS 48

typedef float (*moe_vec_dot_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ moe_vec_dot_t moe_get_vec_dot(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        default:                return nullptr;
    }
}

static constexpr __host__ __device__ int moe_get_vdr(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        default:                return 1;
    }
}

// Fork of mul_mat_vec_q_moe (dedicated MoE MMVQ kernel). Differences:
//   * weights come from a per-expert pointer table instead of one base pointer
//   * kbx_offset drops the channel term (table entries are expert-relative)
//   * cold (host) experts are logged to a miss buffer, hot ones stamped
template<ggml_type type, int c_rows_per_block, bool has_fusion>
__launch_bounds__(MMVQ_MAX_BATCH_SIZE*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void moe_cache_gemv(
        const char * const * tbl,            // [n_experts] up weights (LSB=1 -> host)
        const char * const * tbl_gate,        // [n_experts] gate weights, may be null
        uint32_t * stamps,                    // [n_experts] last-use clock (up tensor)
        uint32_t * stamps_gate,               // [n_experts] last-use clock (gate tensor)
        uint32_t * miss,                      // [1 + MOE_MISS_CAP] packed (id<<12 | expert)
        uint32_t * hits,                     // [1] resident-hit counter (debug/stats)
        const uint32_t * d_clock,             // device copy of the global step counter
        int32_t cache_id,                     // registry id of the up tensor
        int32_t cache_id_gate,                // registry id of the gate tensor (-1 if none)
        const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride) {

    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = moe_get_vdr(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr moe_vec_dot_t vec_dot_q_cuda = moe_get_vec_dot(type);

    const bool shared_expert = has_fusion && fusion.shared_up && blockIdx.y == gridDim.y - 1;
    if (shared_expert) {
        dst = fusion.shared_dst;
        stride_col_dst = fusion.shared_stride_col_dst;
    }

    // fuse gate, bias, scales, and glu_op into the up projection
    bool use_gate = false;
    const void  * vgate      = nullptr;
    const float * x_bias     = nullptr;
    const float * gate_bias  = nullptr;
    const float * x_scale    = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op   active_glu = GGML_GLU_OP_SWIGLU;
    float         glu_limit  = 0.0f;

    if constexpr (has_fusion) {
        use_gate   = fusion.gate != nullptr;
        x_bias     = (const float *) fusion.x_bias;
        gate_bias  = (const float *) fusion.gate_bias;
        active_glu = fusion.glu_op;
        glu_limit  = fusion.glu_limit;
        if constexpr (type == GGML_TYPE_NVFP4) {
            x_scale    = (const float *) fusion.x_scale;
            gate_scale = (const float *) fusion.gate_scale;
        }
    }

    const uint32_t token_idx        = threadIdx.y;
    const int      row0             = c_rows_per_block*blockIdx.x;
    const int      blocks_per_row_x = ncols_x / qk;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;

    const uint32_t channel_dst = shared_expert ? 0 : blockIdx.y;

    if (token_idx >= ncols_dst) {
        return;
    }

    ggml_cuda_pdl_sync();
    const uint32_t channel_x = shared_expert ? 0 : ids[channel_dst + token_idx * ids_stride];
    const uint32_t channel_y = fastmodulo(channel_dst, nchannels_y);

    const void * vx = nullptr;
    const int kbx_offset = row0*stride_row_x; // expert-relative, the table already points at the expert
    if (shared_expert) {
        vx = fusion.shared_up;
        if constexpr (has_fusion) {
            if (use_gate) {
                vgate = fusion.shared_gate;
            }
        }
    } else {
        const char * p = tbl[channel_x];
        const bool is_host = ((uintptr_t) p) & 1u;
        vx = (const void *) (((uintptr_t) p) & ~(uintptr_t) 1u);
        if (stamps) {
            stamps[channel_x] = *d_clock;
        }
        if (is_host && blockIdx.x == 0 && miss) {
            const uint32_t s = atomicAdd(miss, 1u);
            if (s < MOE_MISS_CAP) {
                miss[1 + s] = ((uint32_t) cache_id << 12) | channel_x;
            }
        } else if (!is_host && blockIdx.x == 0 && hits) {
            atomicAdd(hits, 1u);
        }
        if constexpr (has_fusion) {
            if (use_gate) {
                if (tbl_gate) {
                    const char * g = tbl_gate[channel_x];
                    const bool g_host = ((uintptr_t) g) & 1u;
                    vgate = (const void *) (((uintptr_t) g) & ~(uintptr_t) 1u);
                    if (stamps_gate) {
                        stamps_gate[channel_x] = *d_clock;
                    }
                    if (g_host && blockIdx.x == 0 && miss) {
                        const uint32_t s = atomicAdd(miss, 1u);
                        if (s < MOE_MISS_CAP) {
                            miss[1 + s] = ((uint32_t) cache_id_gate << 12) | channel_x;
                        }
                    }
                } else {
                    vgate = fusion.gate;
                }
            }
        }
    }

    const block_q8_1 * y = ((const block_q8_1 *) vy_ptr) + channel_y*stride_channel_y + token_idx*stride_col_y;

    // partial sum for each thread
    float tmp[c_rows_per_block] = {0.0f};
    float tmp_gate[c_rows_per_block] = {0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[i] += vec_dot_q_cuda(vx, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[i] += vec_dot_q_cuda(vgate, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
                }
            }
        }
    }

    ggml_cuda_pdl_lc();

    // Warp-level reduction only - no shared memory needed
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
        if constexpr (has_fusion) {
            if (use_gate) {
                tmp_gate[i] = warp_reduce_sum<warp_size>(tmp_gate[i]);
            }
        }
    }

    // Write results
    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        float result = tmp[threadIdx.x];
        if constexpr (has_fusion) {
            const uint32_t bias_idx = channel_x*stride_channel_dst + row0 + threadIdx.x;

            if constexpr (type == GGML_TYPE_NVFP4) {
                if (x_scale) {
                    result *= x_scale[channel_x];
                }
            }
            if (x_bias) {
                result += x_bias[bias_idx];
            }
            if (use_gate) {
                float gate_value = tmp_gate[threadIdx.x];
                if constexpr (type == GGML_TYPE_NVFP4) {
                    if (gate_scale) {
                        gate_value *= gate_scale[channel_x];
                    }
                }
                if (gate_bias) {
                    gate_value += gate_bias[bias_idx];
                }
                switch (active_glu) {
                    case GGML_GLU_OP_SWIGLU:
                        result *= ggml_cuda_op_silu_single(gate_value);
                        break;
                    case GGML_GLU_OP_GEGLU:
                        result *= ggml_cuda_op_gelu_single(gate_value);
                        break;
                    case GGML_GLU_OP_SWIGLU_OAI:
                        result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                        break;
                    case GGML_GLU_OP_SWIGLU_CLAMP:
                        result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                        break;
                    default:
                        result = result * gate_value;
                        break;
                }
            }
        }
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = result;
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, tmp_gate, vgate, x_bias, gate_bias, active_glu, glu_limit, x_scale, gate_scale);
    } else if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(x_scale, gate_scale);
    }
}

// ---------------------------------------------------------------------------
// host side registry
// ---------------------------------------------------------------------------

struct moe_cache_entry {
    int32_t id = -1;
    ggml_type type = GGML_TYPE_F32;
    const char * data = nullptr;   // host base of the expert tensor
    size_t expert_bytes = 0;
    int n_experts = 0;
    int n_slots = 0;
    char * pool = nullptr;               // device, n_slots*expert_bytes
    char ** d_tbl = nullptr;             // device, n_experts pointers
    uint32_t * d_stamps = nullptr;       // device, n_experts
    std::vector<int32_t> slot_of;        // expert -> slot, -1 = cold
    std::vector<int32_t> expert_of;      // slot -> expert, -1 = free
    std::vector<uint32_t> h_stamps;      // host mirror of d_stamps for eviction
};

static bool g_moe_cache_enabled = false;
static bool g_moe_cache_env_read = false;
static int g_moe_cache_slots = 320;
static size_t g_moe_cache_budget = 30ull << 30;     // total VRAM the pools may use
static size_t g_moe_cache_used = 0;

static std::mutex g_moe_mutex;
static std::map<const char *, moe_cache_entry *> g_moe_by_data;
static std::vector<moe_cache_entry *> g_moe_all;
static uint32_t * g_d_miss = nullptr;      // device [1+MOE_MISS_CAP]
static uint32_t * g_h_miss = nullptr;      // pinned host mirror
static uint32_t * g_d_hits = nullptr;      // device [1]
static uint32_t * g_d_clock = nullptr;     // device clock
static uint32_t g_h_clock = 0;
static std::map<uint32_t, int> g_admission; // (id<<12|expert) -> miss count for admission filter

bool ggml_cuda_moe_cache_enabled() {
    if (!g_moe_cache_env_read) {
        const char * env = getenv("GGML_CUDA_MOE_CACHE");
        g_moe_cache_enabled = env && env[0] && env[0] != '0';
        if (const char * s = getenv("GGML_CUDA_MOE_CACHE_SLOTS")) {
            g_moe_cache_slots = atoi(s);
            if (g_moe_cache_slots < 0) g_moe_cache_slots = 0;
        }
        if (const char * s = getenv("GGML_CUDA_MOE_CACHE_BUDGET_GB")) {
            const double gb = atof(s);
            if (gb > 0) g_moe_cache_budget = (size_t) (gb*1024.0*1024.0*1024.0);
        }
        g_moe_cache_env_read = true;
        if (g_moe_cache_enabled) {
            fprintf(stderr, "moe-cache: enabled, slots=%d budget=%.1f GB\n", g_moe_cache_slots, (double) g_moe_cache_budget / (1024.0*1024.0*1024.0));
        }
    }
    return g_moe_cache_enabled;
}

bool ggml_cuda_moe_cache_active() {
    std::lock_guard<std::mutex> lock(g_moe_mutex);
    return !g_moe_all.empty();
}

// Pin a host tensor for zero-copy GPU reads (page-aligned span, idempotent).
// Returns true when the memory is (or now is) GPU-readable host memory.
bool ggml_cuda_moe_cache_pin_host(const ggml_tensor * t) {
    if (!ggml_cuda_moe_cache_enabled() || !t || !t->buffer || !t->data) {
        return false;
    }

    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, const_cast<void *>(t->data));
    if (err == cudaSuccess) {
        if (attrs.type == cudaMemoryTypeHost || attrs.type == cudaMemoryTypeManaged) {
            return true;  // already pinned / GPU-accessible
        }
        if (attrs.type == cudaMemoryTypeDevice) {
            return false; // VRAM resident - not our business
        }
    } else {
        cudaGetLastError();
        return false;
    }

    // cudaMemoryTypeUnregistered: plain host allocation - pin it now
    const uintptr_t page = 4096;
    const uintptr_t base = (uintptr_t) t->data & ~(page - 1);
    const uintptr_t end  = ((uintptr_t) t->data + ggml_nbytes(t) + page - 1) & ~(page - 1);

    err = cudaHostRegister((void *) base, end - base, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
    if (err == cudaSuccess) {
        fprintf(stderr, "moe-cache: pinned %.2f GB of host memory for %s\n",
                (double) (end - base) / (1024.0*1024.0*1024.0), t->name[0] ? t->name : "<tensor>");
        return true;
    }
    cudaGetLastError();
    fprintf(stderr, "moe-cache: failed to pin %s: %s\n", t->name[0] ? t->name : "<tensor>", cudaGetErrorString(err));
    return false;
}

static bool moe_type_supported(ggml_type t) {
    switch (t) {
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

// cached residency test: the expert tensor must live in host memory that the
// GPU can read directly (cudaHostRegister'ed when GGML_CUDA_REGISTER_HOST=1).
static std::map<const char *, bool> g_ptr_is_host;

static bool moe_ptr_is_host(const void * p) {
    auto it = g_ptr_is_host.find((const char *) p);
    if (it != g_ptr_is_host.end()) {
        return it->second;
    }
    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, const_cast<void *>(p));
    // treat query failures as "not host" and remember the result
    const bool is_host = err == cudaSuccess && attrs.type == cudaMemoryTypeHost;
    g_ptr_is_host[(const char *) p] = is_host;
    return is_host;
}

bool ggml_cuda_moe_cache_wants(const ggml_tensor * src0) {
    if (!ggml_cuda_moe_cache_enabled()) {
        return false;
    }
    if (!src0->buffer || !moe_ptr_is_host(src0->data)) {
        if (getenv("GGML_CUDA_MOE_CACHE_DEBUG") && src0->ne[2] > 1) {
            cudaPointerAttributes attrs;
            cudaError_t err = cudaPointerGetAttributes(&attrs, const_cast<void *>(src0->data));
            fprintf(stderr, "moe-cache: wants? %s type=%s ne2=%lld ptr=%p nbytes=%zu view_src=%p bufname=%s err=%d attr=%d host=%d\n",
                    src0->name[0] ? src0->name : "<tensor>", ggml_type_name(src0->type),
                    (long long) src0->ne[2], src0->data, ggml_nbytes(src0), (void *) src0->view_src,
                    src0->buffer ? ggml_backend_buffer_name(src0->buffer) : "none",
                    (int) err, err == cudaSuccess ? (int) attrs.type : -1,
                    src0->buffer ? (int) ggml_backend_buffer_is_host(src0->buffer) : -1);
        }
        return false;
    }
    if (!moe_type_supported(src0->type)) {
        return false;
    }
    if (src0->ne[2] < 2 || src0->ne[3] != 1 || src0->view_src != nullptr) {
        return false;
    }
    if (!ggml_is_contiguous(src0)) {
        return false;
    }
    return true;
}

static void moe_init_device_globals() {
    if (g_d_miss) {
        return;
    }
    CUDA_CHECK(cudaMalloc(&g_d_miss, sizeof(uint32_t)*(1 + MOE_MISS_CAP)));
    CUDA_CHECK(cudaMallocHost(&g_h_miss, sizeof(uint32_t)*(1 + MOE_MISS_CAP)));
    CUDA_CHECK(cudaMemset(g_d_miss, 0, sizeof(uint32_t)*(1 + MOE_MISS_CAP)));
    CUDA_CHECK(cudaMalloc(&g_d_hits, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(g_d_hits, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&g_d_clock, sizeof(uint32_t)));
    g_h_clock = 0;
    CUDA_CHECK(cudaMemset(g_d_clock, 0, sizeof(uint32_t)));
}

// seed file: lines of "tensorname id1,id2,..." (experts ordered by decode frequency)
static std::map<std::string, std::vector<int>> g_seed;
static bool g_seed_loaded = false;

static void moe_load_seed() {
    if (g_seed_loaded) {
        return;
    }
    g_seed_loaded = true;
    const char * path = getenv("GGML_CUDA_MOE_CACHE_SEED");
    if (!path || !path[0]) {
        return;
    }
    FILE * f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "moe-cache: seed file %s not found, starting cold\n", path);
        return;
    }
    char line[1 << 16];
    while (fgets(line, sizeof(line), f)) {
        char * sp = strchr(line, ' ');
        if (!sp) {
            continue;
        }
        *sp = 0;
        std::vector<int> ids;
        char * p = sp + 1;
        while (*p) {
            char * end = nullptr;
            const long v = strtol(p, &end, 10);
            if (end == p) {
                break;
            }
            ids.push_back((int) v);
            p = (*end == ',') ? end + 1 : end;
        }
        g_seed[line] = std::move(ids);
    }
    fclose(f);
    fprintf(stderr, "moe-cache: loaded seed for %zu tensors from %s\n", g_seed.size(), path);
}

static moe_cache_entry * moe_get_or_create(const ggml_tensor * src0, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(g_moe_mutex);

    moe_init_device_globals();

    auto it = g_moe_by_data.find((const char *) src0->data);
    if (it != g_moe_by_data.end()) {
        return it->second;
    }

    moe_cache_entry * e = new moe_cache_entry();
    e->id = (int32_t) g_moe_all.size();
    e->type = src0->type;
    e->data = (const char *) src0->data;
    e->expert_bytes = src0->nb[2];
    e->n_experts = (int) src0->ne[2];

    // VRAM budget: each tensor takes as many slots as it can afford.
    int slots = g_moe_cache_slots;
    const size_t free_budget = g_moe_cache_budget > g_moe_cache_used ? g_moe_cache_budget - g_moe_cache_used : 0;
    if ((size_t) slots*e->expert_bytes > free_budget) {
        slots = (int) (free_budget / e->expert_bytes);
    }

    // also stay within the GPU's actual free memory (2 GB safety margin for
    // activations, CUDA graphs and allocator growth)
    size_t free_vram = 0, total_vram = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_vram, &total_vram));
    const size_t usable = free_vram > (2ull << 30) ? free_vram - (2ull << 30) : 0;
    if ((size_t) slots*e->expert_bytes > usable) {
        slots = (int) (usable / e->expert_bytes);
    }

    if (slots > e->n_experts) {
        slots = e->n_experts; // more slots than experts is pointless
    }
    e->n_slots = slots;

    std::vector<char *> h_tbl(e->n_experts);
    for (int i = 0; i < e->n_experts; ++i) {
        h_tbl[i] = (char *) (((uintptr_t) (e->data + (size_t) i*e->expert_bytes)) | (uintptr_t) 1);
    }

    CUDA_CHECK(cudaMalloc(&e->d_tbl, sizeof(char *)*e->n_experts));
    CUDA_CHECK(cudaMemcpy(e->d_tbl, h_tbl.data(), sizeof(char *)*e->n_experts, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&e->d_stamps, sizeof(uint32_t)*e->n_experts));
    CUDA_CHECK(cudaMemset(e->d_stamps, 0, sizeof(uint32_t)*e->n_experts));
    e->h_stamps.assign(e->n_experts, 0);

    e->slot_of.assign(e->n_experts, -1);
    e->expert_of.assign(slots, -1);

    if (slots > 0) {
        CUDA_CHECK(cudaMalloc(&e->pool, (size_t) slots*e->expert_bytes));
        g_moe_cache_used += (size_t) slots*e->expert_bytes;
    }

    g_moe_by_data[e->data] = e;
    g_moe_all.push_back(e);

    // seed the pool with the hottest experts from the routing trace (one-time H2D,
    // issued on the compute stream so the first kernel launch is ordered after it)
    moe_load_seed();
    auto sit = g_seed.find(src0->name);
    if (sit != g_seed.end() && e->n_slots > 0) {
        int seeded = 0;
        for (const int expert : sit->second) {
            if (seeded >= e->n_slots) {
                break;
            }
            if (expert < 0 || expert >= e->n_experts || e->slot_of[expert] >= 0) {
                continue;
            }
            int slot = -1;
            for (int s = 0; s < e->n_slots; ++s) {
                if (e->expert_of[s] < 0) {
                    slot = s;
                    break;
                }
            }
            if (slot < 0) {
                break;
            }
            CUDA_CHECK(cudaMemcpyAsync(e->pool + (size_t) slot*e->expert_bytes,
                        e->data + (size_t) expert*e->expert_bytes, e->expert_bytes, cudaMemcpyHostToDevice, stream));
            char * dev_entry = e->pool + (size_t) slot*e->expert_bytes;
            CUDA_CHECK(cudaMemcpyAsync(e->d_tbl + expert, &dev_entry, sizeof(char *), cudaMemcpyHostToDevice, stream));
            e->slot_of[expert] = slot;
            e->expert_of[slot] = expert;
            seeded += 1;
        }
        if (seeded > 0) {
            fprintf(stderr, "moe-cache: seeded %d/%d experts for %s\n", seeded, e->n_slots, src0->name);
        }
    }

    fprintf(stderr, "moe-cache: registered %s type=%s experts=%d slot_bytes=%zu slots=%d (pool %.1f MB, total %.2f GB, free VRAM %.2f GB)\n",
            src0->name[0] ? src0->name : "<tensor>", ggml_type_name(src0->type), e->n_experts, e->expert_bytes,
            e->n_slots, e->n_slots ? (double) e->n_slots*e->expert_bytes / (1024.0*1024.0) : 0.0,
            (double) g_moe_cache_used / (1024.0*1024.0*1024.0),
            (double) free_vram / (1024.0*1024.0*1024.0));

    return e;
}

// ---------------------------------------------------------------------------
// promotion between steps
// ---------------------------------------------------------------------------

static int moe_acquire_slot(moe_cache_entry * e, cudaStream_t stream) {
    if (e->n_slots == 0) {
        return -1;
    }
    for (int s = 0; s < e->n_slots; ++s) {
        if (e->expert_of[s] < 0) {
            return s;
        }
    }
    // evict the least recently used resident expert
    CUDA_CHECK(cudaMemcpyAsync(e->h_stamps.data(), e->d_stamps, sizeof(uint32_t)*e->n_experts, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int victim_slot = -1;
    uint32_t best = UINT32_MAX;
    for (int s = 0; s < e->n_slots; ++s) {
        const int expert = e->expert_of[s];
        const uint32_t stamp = expert >= 0 ? e->h_stamps[expert] : 0;
        if (stamp < best) {
            best = stamp;
            victim_slot = s;
        }
    }
    if (victim_slot < 0) {
        return -1;
    }
    const int victim = e->expert_of[victim_slot];
    if (victim >= 0) {
        e->slot_of[victim] = -1;
        // point the evicted expert back at host RAM
        char * host_entry = (char *) (((uintptr_t) (e->data + (size_t) victim*e->expert_bytes)) | (uintptr_t) 1);
        CUDA_CHECK(cudaMemcpyAsync(e->d_tbl + victim, &host_entry, sizeof(char *), cudaMemcpyHostToDevice, stream));
    }
    e->expert_of[victim_slot] = -1;
    return victim_slot;
}

void ggml_cuda_moe_cache_step(cudaStream_t stream) {
    if (!g_moe_cache_enabled || g_moe_all.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_moe_mutex);

    // pull the miss log
    CUDA_CHECK(cudaMemcpyAsync(g_h_miss, g_d_miss, sizeof(uint32_t)*(1 + MOE_MISS_CAP), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    uint32_t n = g_h_miss[0];
    if (n > MOE_MISS_CAP) {
        n = MOE_MISS_CAP;
    }

    g_h_clock += 1;
    CUDA_CHECK(cudaMemcpyAsync(g_d_clock, &g_h_clock, sizeof(uint32_t), cudaMemcpyHostToDevice, stream));

    static uint32_t stats_calls = 0;
    static uint64_t stats_misses = 0;
    static uint64_t stats_promos = 0;
    static uint64_t stats_hits = 0;
    stats_calls += 1;
    stats_misses += n;

    uint32_t step_hits = 0;
    if (g_d_hits) {
        CUDA_CHECK(cudaMemcpyAsync(&step_hits, g_d_hits, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemsetAsync(g_d_hits, 0, sizeof(uint32_t), stream));
    }
    stats_hits += step_hits;

    if (stats_calls % 64 == 0) {
        size_t resident = 0;
        for (const auto & e : g_moe_all) {
            for (int s = 0; s < e->n_slots; ++s) {
                resident += e->expert_of[s] >= 0 ? 1 : 0;
            }
        }
        fprintf(stderr, "moe-cache: step %u: hits=%u misses=%u (totals: hits %llu, misses %llu, promos %llu), resident %zu\n",
                stats_calls, step_hits, n, (unsigned long long) stats_hits,
                (unsigned long long) stats_misses, (unsigned long long) stats_promos, resident);
    }

    if (n == 0) {
        return;
    }

    // admission filter: promote on the 2nd sighting, keeps one-off experts out
    int promoted = 0;
    for (uint32_t i = 0; i < n && promoted < MOE_MAX_PROMOTIONS; ++i) {
        const uint32_t packed = g_h_miss[1 + i];
        const int32_t cid = (int32_t) (packed >> 12);
        const int32_t expert = (int32_t) (packed & 0xFFF);

        int & count = g_admission[packed];
        count += 1;
        if (count < 2) {
            continue;
        }
        count = 0; // handled

        if (cid < 0 || cid >= (int) g_moe_all.size()) {
            continue;
        }
        moe_cache_entry * e = g_moe_all[cid];
        if (expert < 0 || expert >= e->n_experts || e->slot_of[expert] >= 0) {
            continue;
        }

        const int slot = moe_acquire_slot(e, stream);
        if (slot < 0) {
            continue;
        }

        CUDA_CHECK(cudaMemcpyAsync(e->pool + (size_t) slot*e->expert_bytes,
                    e->data + (size_t) expert*e->expert_bytes, e->expert_bytes, cudaMemcpyHostToDevice, stream));

        char * dev_entry = e->pool + (size_t) slot*e->expert_bytes;
        CUDA_CHECK(cudaMemcpyAsync(e->d_tbl + expert, &dev_entry, sizeof(char *), cudaMemcpyHostToDevice, stream));

        e->slot_of[expert] = slot;
        e->expert_of[slot] = expert;
        promoted += 1;
        stats_promos += 1;
    }

    // keep the admission filter from growing without bound
    if (g_admission.size() > 8192) {
        for (auto it = g_admission.begin(); it != g_admission.end();) {
            if (it->second == 0) {
                it = g_admission.erase(it);
            } else {
                ++it;
            }
        }
    }

    // reset the miss counter for the next step
    uint32_t zero = 0;
    CUDA_CHECK(cudaMemcpyAsync(g_d_miss, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
}

// ---------------------------------------------------------------------------
// launch plumbing
// ---------------------------------------------------------------------------

template<ggml_type type>
static void moe_cache_gemv_launch(
        moe_cache_entry * e, moe_cache_entry * ge,
        const char * src1_q8_1, const int32_t * ids_d, const ggml_cuda_mm_fusion_args_device & fusion, float * dst_d,
        const uint32_t ncols_x, const uint32_t nrows_x, const uint32_t ncols_dst,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t nchannels_y, const uint32_t nchannels_dst, const uint32_t ids_stride,
        const int warp_size, cudaStream_t stream) {

    constexpr int rows_per_block = 2; // same tuning as the stock MoE kernel
    const int64_t nblocks_rows = (nrows_x + rows_per_block - 1) / rows_per_block;
    const dim3 block_nums(nblocks_rows, nchannels_dst + (fusion.shared_up != nullptr));
    const dim3 block_dims(warp_size, ncols_dst);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;

    const uint3 nchannels_y_fd = init_fastdiv_values(nchannels_y);

    if (has_fusion) {
        ggml_cuda_kernel_launch((moe_cache_gemv<type, rows_per_block, true>), launch_params,
                e->d_tbl, ge ? ge->d_tbl : nullptr, e->d_stamps, ge ? ge->d_stamps : nullptr,
                g_d_miss, g_d_hits, g_d_clock, e->id, ge ? ge->id : -1,
                src1_q8_1, ids_d, fusion, dst_d, ncols_x, nchannels_y_fd, nrows_x,
                stride_row_x, stride_col_y, stride_col_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst,
                ncols_dst, ids_stride);
    } else {
        ggml_cuda_kernel_launch((moe_cache_gemv<type, rows_per_block, false>), launch_params,
                e->d_tbl, nullptr, e->d_stamps, nullptr,
                g_d_miss, g_d_hits, g_d_clock, e->id, -1,
                src1_q8_1, ids_d, fusion, dst_d, ncols_x, nchannels_y_fd, nrows_x,
                stride_row_x, stride_col_y, stride_col_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst,
                ncols_dst, ids_stride);
    }
}

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
        int64_t ids_stride, cudaStream_t stream) {

    GGML_UNUSED(ctx); GGML_UNUSED(ids); GGML_UNUSED(dst);
    GGML_UNUSED(ne02); GGML_UNUSED(ne03); GGML_UNUSED(s03);

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;

    moe_cache_entry * e = moe_get_or_create(src0, stream);

    // fused gate tensor (may itself be host-resident -> its own table)
    moe_cache_entry * ge = nullptr;
    if (fusion && fusion->gate && ggml_cuda_moe_cache_wants(fusion->gate)) {
        ge = moe_get_or_create(fusion->gate, stream);
    }

    switch (src0->type) {
        case GGML_TYPE_Q2_K:
            moe_cache_gemv_launch<GGML_TYPE_Q2_K>(e, ge, src1_q8_1, ids_d, fusion_dev, dst_d,
                ne00, ne01, ncols_dst, s01, stride_col_y, stride_col_dst, s02, stride_channel_y, stride_channel_dst,
                nchannels_y, nchannels_dst, ids_stride, warp_size, stream);
            break;
        case GGML_TYPE_Q3_K:
            moe_cache_gemv_launch<GGML_TYPE_Q3_K>(e, ge, src1_q8_1, ids_d, fusion_dev, dst_d,
                ne00, ne01, ncols_dst, s01, stride_col_y, stride_col_dst, s02, stride_channel_y, stride_channel_dst,
                nchannels_y, nchannels_dst, ids_stride, warp_size, stream);
            break;
        case GGML_TYPE_Q4_K:
            moe_cache_gemv_launch<GGML_TYPE_Q4_K>(e, ge, src1_q8_1, ids_d, fusion_dev, dst_d,
                ne00, ne01, ncols_dst, s01, stride_col_y, stride_col_dst, s02, stride_channel_y, stride_channel_dst,
                nchannels_y, nchannels_dst, ids_stride, warp_size, stream);
            break;
        case GGML_TYPE_Q5_K:
            moe_cache_gemv_launch<GGML_TYPE_Q5_K>(e, ge, src1_q8_1, ids_d, fusion_dev, dst_d,
                ne00, ne01, ncols_dst, s01, stride_col_y, stride_col_dst, s02, stride_channel_y, stride_channel_dst,
                nchannels_y, nchannels_dst, ids_stride, warp_size, stream);
            break;
        case GGML_TYPE_Q6_K:
            moe_cache_gemv_launch<GGML_TYPE_Q6_K>(e, ge, src1_q8_1, ids_d, fusion_dev, dst_d,
                ne00, ne01, ncols_dst, s01, stride_col_y, stride_col_dst, s02, stride_channel_y, stride_channel_dst,
                nchannels_y, nchannels_dst, ids_stride, warp_size, stream);
            break;
        case GGML_TYPE_Q4_0:
            moe_cache_gemv_launch<GGML_TYPE_Q4_0>(e, ge, src1_q8_1, ids_d, fusion_dev, dst_d,
                ne00, ne01, ncols_dst, s01, stride_col_y, stride_col_dst, s02, stride_channel_y, stride_channel_dst,
                nchannels_y, nchannels_dst, ids_stride, warp_size, stream);
            break;
        case GGML_TYPE_Q5_0:
            moe_cache_gemv_launch<GGML_TYPE_Q5_0>(e, ge, src1_q8_1, ids_d, fusion_dev, dst_d,
                ne00, ne01, ncols_dst, s01, stride_col_y, stride_col_dst, s02, stride_channel_y, stride_channel_dst,
                nchannels_y, nchannels_dst, ids_stride, warp_size, stream);
            break;
        case GGML_TYPE_Q8_0:
            moe_cache_gemv_launch<GGML_TYPE_Q8_0>(e, ge, src1_q8_1, ids_d, fusion_dev, dst_d,
                ne00, ne01, ncols_dst, s01, stride_col_y, stride_col_dst, s02, stride_channel_y, stride_channel_dst,
                nchannels_y, nchannels_dst, ids_stride, warp_size, stream);
            break;
        default:
            GGML_ABORT("moe-cache: unsupported expert type %s", ggml_type_name(src0->type));
    }
}
