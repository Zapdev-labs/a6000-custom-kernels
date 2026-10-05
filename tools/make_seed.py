#!/usr/bin/env python3
"""Build a hot-expert seed file from a captured MoE routing trace.

Usage:
    python3 tools/make_seed.py [trace.bin] [seed.txt]
    (defaults: /root/moe_trace.bin -> /root/moe_seed.txt)

The trace is produced by the instrumented CPU backend at
patches/trace-hook/ggml-cpu.c, enabled with GGML_MOE_TRACE=<path>. Binary format
per record:
  u32 namelen, u32 n_used, u32 n_tok, u32 n_exp, char name[namelen], i32 ids[n_used * n_tok]

Output: one line per expert tensor, "tensorname id1,id2,..." with experts sorted
by decode-pick frequency. The cache seeds its VRAM pools from the first
GGML_CUDA_MOE_CACHE_SLOTS entries of each line.
"""
import struct
import sys
from collections import Counter

trace_path = sys.argv[1] if len(sys.argv) > 1 else "/root/moe_trace.bin"
out_path = sys.argv[2] if len(sys.argv) > 2 else "/root/moe_seed.txt"

counts = {}
n_rec = 0
with open(trace_path, "rb") as f:
    while True:
        hdr = f.read(16)
        if len(hdr) < 16:
            break
        namelen, n_used, n_tok, n_exp = struct.unpack("<IIII", hdr)
        name = f.read(namelen).decode("utf-8", "replace")
        ids = struct.unpack(f"<{n_used * n_tok}i", f.read(4 * n_used * n_tok))
        n_rec += 1
        if n_tok != 1 or n_exp < 2:
            continue  # decode records only
        c = counts.setdefault(name, Counter())
        for e in ids:
            if 0 <= e < n_exp:
                c[e] += 1

with open(out_path, "w") as f:
    for name, c in counts.items():
        top = [e for e, _ in c.most_common()]  # all 512, consumer takes what it needs
        f.write(name + " " + ",".join(str(e) for e in top) + "\n")

print(f"records={n_rec} tensors={len(counts)} -> {out_path}")
for name in list(counts)[:3]:
    c = counts[name]
    print(name, "top10:", [e for e, _ in c.most_common(10)], "distinct:", len(c))
