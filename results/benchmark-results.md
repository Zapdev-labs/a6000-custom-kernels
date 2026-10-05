# Benchmark results

All numbers: `llama-bench -p 512 -n 128 -ngl 99 -fa on -lm none -r 1`, model-native context
(262,144 tokens), single RTX A6000 48 GB (SM86, 768 GB/s), RunPod secure US-KS-2, host RAM
pinned via CUDA_Host buffers. Model: CYBER-FROST-3.8 Q2_K_S (77.15 GiB, 179.55B params,
512 experts × 48 layers, PLE n-gram table 28.8 GB, MTP head). llama.cpp @ `11fe0215`,
custom kernels from this repo.

## Headline

| Configuration | pp512 (t/s) | tg128 (t/s) |
|---|---:|---:|
| Stock llama.cpp, experts on CPU (`-ncmoe 48`) | ~9 | ~5 |
| Session 1: cache, seeded, `-ncmoe 18` | 109.76 | 50.53 |
| **Session 2 final: cache, seeded, `-ncmoe 8`** | **166.88** | **60.13** |

Speedup vs stock: prefill ~18.5×, decode ~12×.

## `-ncmoe` sweep (session 2, same binary, same pod)

`-ncmoe N` = N MoE layers keep their experts in host RAM, served by the cache; the other
48−N layers hold experts in VRAM.

| N | pp512 (t/s) | tg128 (t/s) | pool resident (experts) |
|---:|---:|---:|---:|
| 20 | 98.54 | 46.05 | — |
| 18 | 102.71 | 48.22 | 15,801 |
| 16 | 115.73 | 51.47 | 15,447 |
| 14 | 119.67 | 49.28 | 12,635 |
| 12 | 136.62 | 55.44 | 10,922 |
| 10 | 150.18 | 54.25 | 9,291 |
| 8 | 166.88 | 60.13 | 7,680 |
| 6 | 218.26 | 54.05 | 5,057 |
| 4 | 301.10 | 41.86 | 1,975 |

Reading the curve: prefill rises monotonically as N falls (fewer host experts to stream over
PCIe per chunk); decode peaks at N=8 where the seed still fills the pools for all host
layers, and collapses below it as pool capacity gives out. N=8 is the joint optimum.

The N=8 row was confirmed twice: 169.83/59.63 in the sweep and 166.88/60.13 in a fresh
final run — ±3% run-to-run variance is expected (shared cloud host, PCIe contention).

## Context depth

llama-bench's tg128 measures decode with ~640 valid KV rows (the 262,144 cache is fully
allocated but mostly empty). A deeper run at N=8 with a 32,768-token prefill:

| Test | t/s |
|---|---:|
| pp32768 | 142.19 |
| tg128 after 32k prefill | 57.90 |

Decode lost only ~4% moving from 640-row to 32k-row attention because decode is
expert-traffic-bound, not attention-bound. Extrapolating the KV-read cost to a completely
full context puts true 262k-depth decode in the ~45–55 t/s range; the run to measure it
exactly (~35 min of prefill) was killed by the credit limit before completing.

## Cache behavior (N=8, seeded)

- Per decode step: ~2,200 expert reads, 90–95% served from VRAM pools, remainder zero-copy
  over PCIe from pinned host RAM.
- Missing the seed file is catastrophic: identical binary, cold pools → pp512 50.39 /
  tg128 12.10, miss buffer saturating at its 8,190-record cap. Seed on registration is the
  single highest-leverage knob.
- LRU promotion admits on second sighting, max 48 promotions per step, eviction by stamp
  scan; pools size adaptively from free VRAM (`cudaMemGetInfo`, 2 GB margin).

## Negative results

Each was benchmarked at full context and reverted with an exact-inverse patch:

| Attempt | pp512 / tg128 | Root cause |
|---|---|---|
| PLE island fix (pin 28.8 GB PLE table, claim GET_ROWS, accept all host bufts) | 92.29 / 45.22 | The CUDA\|CPU\|CUDA split island (PLE n-gram gather) costs a couple of small syncs per step; moving the gather to GPU zero-copy cost more than the syncs saved. The island is cheaper than its replacement. |
| q8_0 KV cache (`-ctk q8_0 -ctv q8_0`), frees 3.2 GB for pools | 105.15 / 47.77 | Flash-attention q8_0 KV kernels on SM86 cost more than the 50% bandwidth saving returns. |
| Speculative decoding (session 1 pod): MTP draft | 2.7 t/s decode | Draft steps run the same MoE — expert traffic multiplies over PCIe. |
| Speculative decoding: n-gram cache draft | 11.5 t/s decode | Same effect, smaller magnitude. |
| GET_ROWS claim without pinning (session 1) | — | Scheduler staged the entire 28.8 GB PLE table in VRAM and OOM'd the pools; reverted the same day. |

## Historical (session 1, different host, same hardware)

| Configuration | pp512 | tg128 |
|---|---:|---:|
| Stock, experts on CPU | ~9 | ~5 |
| Cache, seeded, all experts on CPU (`-ncmoe 48`) | 49.31 | 20.60 |
| Cache, seeded, hybrid `-ncmoe 18` | 109.76 | 50.53 |

The all-CPU row is the cache working perfectly with zero VRAM experts — still 4× slower than
the N=8 hybrid, which is the argument for the hybrid split: every host-layer expert read is a
~1 ms PCIe latency chain at batch 1, so the winning layout keeps host layers to the minimum
the pools can cover.

## Cost accounting

Two RunPod pods, $0.53/hr secure (community stock was unavailable both days). Total spend
across both optimization sessions fit inside a ~$5 credit balance; the second session ended
when RunPod terminated the pod on credit exhaustion mid-run.
