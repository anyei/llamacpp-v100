#pragma once

// SSD expert-streaming buffer type (task 15, phase 2/4 increment 1).
//
// A host buffer type whose tensors are backed by a sparse anonymous arena that
// is filled on demand from the model file (per-expert slices) just before the
// consuming MUL_MAT_ID computes, and evicted (MADV_DONTNEED) under an LRU byte
// budget. Lets an MoE model whose expert weights exceed VRAM+RAM run by holding
// only the hot/in-flight experts resident. See docs/ssd-streaming-plan.md.
//
// Enabled by env LLAMA_SSD_STREAM_BUFFER=1 (budget: LLAMA_SSD_STREAM_BUDGET MiB,
// default 8192). This is a distinct mechanism from the advisory mmap director
// in common/ssd-streaming.cpp (LLAMA_SSD_STREAMING).

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

#ifdef __cplusplus
}
#endif
