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

// true when the GPU expert-slot cache (increment 3) is enabled
// (LLAMA_SSD_STREAM_GPU=1; VRAM budget MiB via LLAMA_SSD_STREAM_VRAM_BUDGET).
// 3.2a wires the host-side policy only; the VRAM buffer + fill + id->slot remap
// land in later sub-increments.
GGML_API bool ggml_ssd_stream_gpu_enabled(void);

// (3.2b-1) Lazily allocate a persistent VRAM slot pool for streamed expert tensor
// w on `backend`, sized by the VRAM budget / slice bytes. One pool per slice
// shape (gate/up share; down separate). No-op if GPU landing is off, w is not
// streamed, or the pool already exists. Called from the scheduler's expert-copy
// block. Fill + id->slot remap wire in later sub-increments.
GGML_API void ggml_ssd_stream_gpu_ensure_pool(const struct ggml_tensor * w, ggml_backend_t backend);

// (3.2e) Reclaim the wasted VRAM the graph allocator would otherwise reserve for
// a streamed-expert MUL_MAT_ID input copy. At compute the copy's data pointer is
// redirected to the persistent slot pool (see gpu_bind), so its own full-size
// n_expert-slice buffer is dead weight in the compute arena. This shrinks the copy
// to a single slice BEFORE gallocr sizes the arena, freeing that VRAM for a bigger
// cache. Does NOT allocate the pool (this runs in the measure/reserve pass, before
// the weights load and the streamed-tensor registry is populated - allocating here
// would mis-size the pools by class count); the pool is allocated on the compute
// path and gpu_bind redirects copy->data to it. Returns true if it shrank the copy -
// the caller MUST then let gpu_bind handle the node (the copy is too small for
// copy_experts; the compute fallback aborts loudly if bind ever fails on a shrunk
// copy). No-op (returns false) if GPU landing/reclaim is off or src is not a streamed
// expert tensor (ne[2] <= 1). Kill-switch: LLAMA_SSD_STREAM_GPU_NO_RECLAIM=1.
GGML_API bool ggml_ssd_stream_gpu_shrink_copy(struct ggml_tensor * copy,
        const struct ggml_tensor * src, ggml_backend_t backend);

// (3.2b-2) Capture and return the node's original ids (router selection) tensor.
// The scheduler block reads the router's fresh selection each token from this,
// even after we swap node->src[2] to the remapped slot ids. Captured once per node.
GGML_API const struct ggml_tensor * ggml_ssd_stream_gpu_orig_ids(
        const struct ggml_tensor * node, const struct ggml_tensor * cur_ids);

// (3.2b-2) GPU landing for one streamed-expert MUL_MAT_ID: touch the VRAM slot
// pool for each used expert (hit = resident, zero copy), H2D-fill misses from the
// RAM arena (input->data), remap ids -> slot ids, alias input_cpy to the slot
// buffer, and swap node->src[2] to the slot ids. Returns true if it handled the
// node (caller then skips the full copy_experts). ids = host copy of the router
// selection, n_ids elements; n_expert = input->ne[2].
GGML_API bool ggml_ssd_stream_gpu_bind(struct ggml_tensor * node, const struct ggml_tensor * input,
        struct ggml_tensor * input_cpy, ggml_backend_t backend,
        const int32_t * ids, int64_t n_ids, int64_t n_expert);

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
