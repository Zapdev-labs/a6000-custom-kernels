#!/usr/bin/env python3
"""Debug: print why split-input copies are created for expert weights.

Enable at runtime with GGML_SCHED_DEBUG=1 (and optionally GGML_CUDA_MOE_CACHE_DEBUG=1).
Set LLAMA_CPP_DIR (default ~/llama.cpp) to target another checkout.
"""
import os
import sys

LLAMA_CPP_DIR = os.environ.get("LLAMA_CPP_DIR") or os.path.expanduser("~/llama.cpp")
BB = os.path.join(LLAMA_CPP_DIR, "ggml/src/ggml-backend.cpp")

def patch(path, old, new, count=1):
    with open(path) as f:
        src = f.read()
    n = src.count(old)
    assert n == count, f"{path}: expected {count}, found {n}"
    with open(path, "w") as f:
        f.write(src.replace(old, new))
    print(f"patched {path}")

patch(BB,
'''                if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {''',
'''                if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    if (getenv("GGML_SCHED_DEBUG") && src->ne[2] > 1) {
                        fprintf(stderr, "SCHEDDBG: copy for %s (src_b=%d cur_b=%d buft=%s supported=%d bufname=%s)\\n",
                                src->name, src_backend_id, cur_backend_id,
                                src->buffer ? ggml_backend_buft_name(src->buffer->buft) : "none",
                                ggml_backend_sched_buffer_supported(sched, src, cur_backend_id),
                                src->buffer ? ggml_backend_buffer_name(src->buffer) : "none");
                    }''',
count=2)
print("dbg patch applied")
