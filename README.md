# CYBER-FROST-3.8 at full 262k context on one RTX A6000

Custom SM86 CUDA kernels for [llama.cpp](https://github.com/ggml-org/llama.cpp) that run a
**180-billion-parameter MoE** (179.55B total, ~3.8B active per token, 512 experts × 48 layers,
77 GiB Q2_K_S) at its **full 262,144-token context** on a single **48 GB RTX A6000**.

The model does not fit — 46 GB of expert weights, a 28.8 GB per-layer embedding (PLE) table,
and a 6.4 GB KV cache against 48 GB of VRAM — so the kernels route around the wall: a
**hot-expert VRAM cache with a device pointer table**, zero-copy PCIe reads of pinned host
experts, and scheduler patches that let the CUDA backend claim the MoE ops in the first place.

## Results

`llama-bench`, model-native context (262,144), single RTX A6000 (RunPod, $0.53/hr):

| Configuration | pp512 (t/s) | tg128 (t/s) |
|---|---:|---:|
| Stock llama.cpp, experts on CPU | ~9 | ~5 |
| Session 1 config (`-ncmoe 18`) | 109.8 | 50.5 |
| **Final (`-ncmoe 8`, seeded cache)** | **166.9** | **60.1** |

That is **~18× faster prefill and ~12× faster decode than the stock baseline**, all at full
context. Prefill clears the 100 t/s target with 67% headroom; decode lands at 60 t/s, which
is the memory-system floor for this model on this GPU (see [Why decode tops out at ~60](#why-decode-tops-out-at-60-ts)).

The host/VRAM expert split is a tunable operating point (`-ncmoe N` = N MoE layers keep
experts in host RAM), measured at 262k context:

| N | pp512 (t/s) | tg128 (t/s) | notes |
|---:|---:|---:|---|
| 20 | 98.5 | 46.1 | conservative; more VRAM headroom |
| 18 | 102.7 | 48.2 | session-1 winner |
| 16 | 115.7 | 51.5 | |
| 14 | 119.7 | 49.3 | |
| 12 | 136.6 | 55.4 | |
| 10 | 150.2 | 54.3 | |
| **8** | **166.9** | **60.1** | **best joint point** |
| 6 | 218.3 | 54.1 | prefill-favoring |
| 4 | 301.1 | 41.9 | prefill-favoring |

Decode holds up at real context depth, not just at benchmark depth: after a 32,768-token
prefill, `pp32768 = 142.2 t/s` and `tg128 = 57.9 t/s` (vs 60.1 at ~640-row attention). The
full sweep and methodology details are in [results/benchmark-results.md](results/benchmark-results.md).

## The three custom pieces

### 1. `kernels/moe-cache.cu` — hot-expert cache + pointer-table MoE GEMV (853 lines)

A fork of the `mul_mat_vec_q_moe` MMVQ kernel that sources each routed expert through a
**device pointer table** instead of one base pointer:

```
                    router (top-8 of 512 experts, per layer)
                                      │
   ┌──────────────────────────────────▼──────────────────────────────────┐
   │ VRAM (48 GB)                                                          │
   │  40 layers' experts (~37 GB)   hot-expert pools (seeded, LRU)         │
   │  KV cache 6.4 GB               dense weights ~2.4 GB                  │
   └───────────────┬──────────────────────────────────┬───────────────────┘
                   │ MUL_MAT_ID claimed by CUDA        │
   ┌───────────────▼──────────────────────────────────▼───────────────────┐
   │ moe_cache_gemv kernel (one launch per expert tensor)                   │
   │    d_tbl[expert]:  LSB=0 → VRAM pool slot  (hot, ~90-95% of reads)    │
   │                    LSB=1 → pinned host RAM   (cold, zero-copy PCIe)   │
   │    cold reads are logged to a miss buffer; d_stamps drive LRU eviction │
   └───────────────┬───────────────────────────────────────────────────────┘
                   │ ggml_cuda_moe_cache_step() between graphs (never captured)
   ┌───────────────▼───────────────────────────────────────────────────────┐
   │ Host RAM (pinned CUDA_Host): 8 layers' experts (~8 GB) + PLE 28.8 GB  │
   └────────────────────────────────────────────────────────────────────────┘
```

Each expert tensor gets its own table, VRAM pool, and stamp array. A miss-recorder kernel
packs `(tensor_id << 12 | expert)` into a device buffer; a promotion pass between decode
steps admits repeatedly-missed experts (2nd sighting) into free pool slots and evicts by
stamp scan. Pools are **seeded at registration** from a measured routing-frequency file, which
is what takes decode from 12 to 60 t/s (a cold cache measured 50.4/12.1 — seeding matters
more than any other single knob). Pool capacity adapts to free VRAM via `cudaMemGetInfo`
with a 2 GB safety margin. Supports Q4_0/Q5_0/Q8_0/Q2_K–Q6_K MMVQ dot kernels and the full
MMVQ fusion argument set (shared-expert block, GLU, bias/scale write-back).

### 2. `kernels/ssm-conv.cu` — F16 conv weights fix (217 lines)

Stock `ssm_conv_f32` hardcodes `sizeof(float)` weights; this model's conv weights arrive as
F16, so stock llama.cpp **crashes on CUDA**. The kernel is templated on the weight type
(`WT = float | ggml_half`). Without this fix nothing else in this repo runs at all.

### 3. Scheduler patches (`patches/`)

llama.cpp's scheduler won't send MoE GEMVs to a GPU that doesn't hold the weights, and
refuses zero-copy reads of host buffers on discrete GPUs. Two patch scripts flip both:
claim all `MUL_MAT_ID` batches for CUDA, and accept pinned `CUDA_Host` buffer types on
discrete GPUs. A hook after `ggml_cuda_graph_evaluate_and_capture` runs the LRU promotion
pass outside of CUDA-graph capture.

## The `-lm none` requirement (the hard-won gotcha)

**You must load the model with `-lm none`.** With the default `-lm auto`, llama.cpp mmaps
the GGUF and `llama-model-loader.cpp` swaps `CUDA_Host` buffer types for plain `CPU_Mapped`
file mappings. Host-resident expert tensors are then *not* pinned, the CUDA backend cannot
read them zero-copy, and the scheduler materializes full VRAM copies of every claimed op —
throughput collapses to ~5 t/s and the copy thrashing looks like a scheduler bug. With
`-lm none`, host tensors land in pinned `CUDA_Host` buffers, registration fires, and the
pointer-table kernel reads cold experts straight over PCIe. Budget host RAM accordingly
(the whole 77 GB file is read into RAM, not mapped).

## Reproduce

Requirements: NVIDIA GPU with ≥ 48 GB VRAM (tuned for SM86; any `CMAKE_CUDA_ARCHITECTURES`
works since the kernels use no SM-specific instructions), CUDA 12.x, ≥ 96 GB host RAM,
~15 min build on 60 cores.

```bash
# 1. build: clone llama.cpp @11fe0215, install kernels, patch, compile
scripts/build.sh                       # LLAMA_CPP_DIR=... CUDA_ARCH=... to override

# 2. benchmark the winning configuration
MODEL=/path/to/CYBER-FROST-3.8-Q2_K_S.gguf scripts/bench.sh

# 3. or chat with it
MODEL=/path/to/CYBER-FROST-3.8-Q2_K_S.gguf scripts/chat.sh
```

### Regenerating the seed for another model/workload

`assets/moe_seed.txt` is measured for CYBER-FROST-3.8 (144 expert tensors, experts sorted by
decode-pick frequency; top-320 of 512 experts cover 99.9% of routing picks, and adjacent-token
recurrence is only ~30%, so a static hot set genuinely pays). To re-measure for a different
model or workload:

```bash
cp patches/trace-hook/ggml-cpu.c  <llama.cpp>/ggml/src/ggml-cpu/ggml-cpu.c   # trace hook
# rebuild, then run your workload with GGML_MOE_TRACE=/root/moe_trace.bin
python3 tools/make_seed.py /root/moe_trace.bin /root/moe_seed.txt
GGML_CUDA_MOE_CACHE_SEED=/root/moe_seed.txt scripts/bench.sh
```

## Tuning knobs

| Knob | Default | Meaning |
|---|---|---|
| `-ncmoe N` | 8 | MoE layers whose experts stay in host RAM behind the cache. 8 = best joint point; 4–6 favor prefill; 18+ for tight VRAM |
| `GGML_CUDA_MOE_CACHE` | 0 | 1 enables the cache and the scheduler claims |
| `GGML_CUDA_MOE_CACHE_SLOTS` | 320 | pool slots per expert tensor |
| `GGML_CUDA_MOE_CACHE_SEED` | — | seed file (frequency-ordered expert ids per tensor) |
| `GGML_CUDA_MOE_CACHE_BUDGET_GB` | 30 | total pool budget cap (free VRAM binds first) |
| `GGML_CUDA_MOE_CACHE_DEBUG` | 0 | per-tensor registration + step stats (every 64 steps) |

## Negative results (measured, don't retry)

Every "obvious" next optimization was benchmarked at 262k context and **lost**:

| Attempt | pp512 / tg128 | Why it lost |
|---|---|---|
| PLE island fix: pin the 28.8 GB PLE table, claim its GET_ROWS for CUDA (kills the CUDA\|CPU\|CUDA split) | 92.3 / 45.2 | scattered GPU-side gathers over PCIe cost more than the island's small sync stalls |
| q8_0 KV cache (`-ctk q8_0 -ctv q8_0`) to free pool VRAM | 105.2 / 47.8 | flash-attention q8_0 kernel cost exceeded the bandwidth saved |
| Speculative decoding: MTP draft / n-gram cache | 2.7 / 11.5 t/s decode | every draft step multiplies expert traffic over PCIe |
| Cache with no seed file (cold pools) | 50.4 / 12.1 | seeding dominates; see above |
| All experts on CPU behind the cache | 49.3 / 20.6 | PCIe is the wall without the hybrid split |

## Why decode tops out at ~60 t/s

Each token activates 8 experts per layer × 48 layers ≈ 0.7 GB of weights. Even if every byte
sat in VRAM — impossible, since experts (46 GB) plus the 262k KV cache (6.4 GB) exceed the
card — that read alone costs ~1 ms at 768 GB/s. The measured 16.6 ms/token additionally
carries per-layer GEMV kernel time, graph-launch overhead, the PLE island sync tax, and the
~5-10% of expert reads served over PCIe. Reaching 100 t/s decode (10 ms/token) needs all
experts resident in VRAM together with the KV cache — a ≥ 64 GB card, or two A6000s. The
prefill target has no such floor, which is why `-ncmoe 4` reaches 301 t/s.

## Repository map

```
kernels/            moe-cache.cu / .cuh  — hot-expert cache + pointer-table GEMV
                    ssm-conv.cu          — F16 conv-weight fix (prerequisite)
patches/            apply_patches.py    — wire-in: include, MMVQ intercept, step hook, gate
                    patch_final_state.py — claim all MUL_MAT_ID + accept CUDA_Host bufts
                    patch_ple*.py       — PLE island experiment + exact revert (net loss)
                    patch_revert.py, patch_buft.py — historical/superseded
                    patch_dbg.py        — scheduler copy-decision instrumentation (SCHEDDBG)
                    trace-hook/ggml-cpu.c — routing-trace hook for seed regeneration
tools/make_seed.py  trace → frequency-ordered seed
assets/moe_seed.txt measured seed for CYBER-FROST-3.8 (144 tensors)
scripts/            build.sh / bench.sh / chat.sh
results/            final.patch (complete diff vs upstream @11fe0215) + benchmark-results.md
```

## Provenance

The kernels are derivatives of llama.cpp (MIT) — `moe-cache.cu` forks the MMVQ MoE kernel and
`ssm-conv.cu` modifies the stock SSM conv kernel, both from commit `11fe0215`, which every
patch anchor in this repo is written against. Model: `freakyskittle/CYBER-FROST-3.8-GGUF`
(Q2_K_S, 77.15 GiB, 179.55B params, `qwen4exp`-family architecture with PLE n-gram embeddings
and an MTP head). All benchmarks ran on RunPod secure US-KS-2, RTX A6000 (48 GB, SM86),
llama-bench `-p 512 -n 128 -fa on -lm none -r 1` at model-native 262,144 context.
