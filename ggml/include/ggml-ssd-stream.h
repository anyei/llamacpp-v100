#pragma once

// SSD expert-streaming buffer type (task 15, phase 2/4 increment 1).
//
// A host buffer type whose tensors are backed by a sparse anonymous arena that
// is filled on demand from the model file (per-expert slices) just before the
// consuming MUL_MAT_ID computes, and evicted (MADV_DONTNEED) under a byte
// budget. Lets an MoE model whose expert weights exceed VRAM+RAM run by holding
// only the hot/in-flight experts resident. See docs/ssd-streaming-plan.md.
//
// Enabled by env LLAMA_SSD_STREAM_BUFFER=1 (budget: LLAMA_SSD_STREAM_BUDGET MiB,
// default 8192). The resident cache is a plain LRU by default; an optional
// segmented LRU (LLAMA_SSD_STREAM_SLRU=1, protected size LLAMA_SSD_STREAM_PROTECTED_PCT,
// default 80) adds scan resistance but measured no win for DeepSeek MoE.
// LLAMA_SSD_STREAM_NO_ODIRECT=1 forces buffered reads (kill-switch/diagnostic).
// This is a distinct mechanism from the advisory mmap director
// in common/ssd-streaming.cpp (LLAMA_SSD_STREAMING).
//
// The fill callback lets non-expert nodes batch and only forces a boundary at
// each streamed expert node; this relies on the expert-selection ids being
// produced in an earlier scheduler split (true when non-experts are offloaded,
// -ngl > 0). For configs where experts and their router share a split (pure-CPU
// -ngl 0), set LLAMA_SSD_STREAM_SERIAL=1 to force correct node-at-a-time fills.

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// true when SSD expert streaming is enabled (env LLAMA_SSD_STREAM_BUFFER)
GGML_API bool ggml_ssd_stream_enabled(void);

// true if a tensor with this name should be streamed (MoE expert weight tensor)
GGML_API bool ggml_ssd_stream_should_stream(const char * name);

// the streaming buffer type (host; backed by an on-demand pread arena)
GGML_API ggml_backend_buffer_type_t ggml_ssd_stream_buft(void);

// true if this tensor lives in the streaming buffer type
GGML_API bool ggml_ssd_stream_is_streamed(const struct ggml_tensor * t);

// register a streamed tensor's file backing; the loader calls this instead of
// reading the tensor's bytes. fd is duplicated internally (kept open for the
// process lifetime); file_offset is the tensor's byte offset in the file.
GGML_API void ggml_ssd_stream_note(const struct ggml_tensor * t, int fd, size_t file_offset);

// eval callback (matches ggml_backend_sched_eval_callback): on the pre-compute
// "ask" for a streamed MUL_MAT_ID, fills the selected experts' slices into the
// arena and returns true (forcing the node to compute alone right after).
GGML_API bool ggml_ssd_stream_eval_cb(struct ggml_tensor * t, bool ask, void * user_data);

// Ensure streamed experts are resident in the arena before the scheduler reads
// them. The backend scheduler's "copy only the used experts" optimization
// materializes a MUL_MAT_ID's expert weights from the source buffer during the
// input-copy phase - which runs BEFORE the consuming node, so the lazy eval_cb
// fill is too late on that path. This hook is called from that block to fill the
// used experts first. w is the expert-weight tensor; used_ids is a ggml_bitset
// over [0, n_expert) (NULL => ensure all experts). No-op if w is not streamed.
GGML_API void ggml_ssd_stream_prefill_experts(const struct ggml_tensor * w, const uint32_t * used_ids, int64_t n_expert);

#ifdef __cplusplus
}
#endif
