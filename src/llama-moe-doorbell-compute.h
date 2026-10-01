#pragma once

// TASKS #154 item 3: the doorbell executor's per-layer math, shared by the executor (a thread team, a
// barrier between phases) and its unit test (tests/test-moe-doorbell.cpp, single thread or any split). It computes
// what the stock CPU chain computes for the miss lanes - up/gate mul_mat_id, the (clamped) SwiGLU, down mul_mat_id -
// with ggml-cpu's own from_float + vec_dot, and returns the gating-weighted sum per row.

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// rows per request the executor handles (the cache's max_batch stays below the scheduler's offload width, 32)
#define LLAMA_MOE_DB_MAX_TOKENS 32

struct llama_moe_db_layer {
    const ggml_tensor * up   = nullptr; // [n_embd, n_ff, n_expert], host memory, plain row layout
    const ggml_tensor * gate = nullptr; // [n_embd, n_ff, n_expert]
    const ggml_tensor * down = nullptr; // [n_ff, n_embd, n_expert]
    float limit            = 0.0f;      // SwiGLU clamp of this layer (0 = none)
    bool  gate_clamp_first = false;     // DeepSeek-V4: clamp the gate BEFORE SiLU; otherwise clamp SiLU(gate)
};

struct llama_moe_db_miss {
    int32_t t;  // row of the request
    int32_t e;  // expert
    float   w;  // gating weight
};

struct llama_moe_db_work {
    int32_t n_embd = 0, n_ff = 0;
    int32_t max_tokens = 0, n_used = 0;

    std::vector<llama_moe_db_miss> misses;
    std::vector<uint8_t>           xq_gate, xq_up, aq; // quantized rows / activations
    size_t                         xq_rs = 0, aq_rs = 0;
    std::vector<float>             g, u;               // [miss][n_ff]

    // size the scratch for requests of up to max_tokens rows x n_used lanes; max_xq_rs / max_aq_rs = the largest
    // quantized row of the layers' gate/up input (n_embd) and down input (n_ff)
    void reserve(int32_t n_embd, int32_t n_ff, int32_t max_tokens, int32_t n_used, size_t max_xq_rs, size_t max_aq_rs);

    // the miss list of one request; false = a corrupt request (row count or expert id out of range)
    bool collect(const llama_moe_db_layer & L, const int32_t * ids, const float * w, int32_t T);

    // the phases, for thread ith of nth (the caller puts a barrier between them)
    void quantize(const llama_moe_db_layer & L, const float * x, int32_t T, int ith, int nth);
    void gate_up (const llama_moe_db_layer & L, int ith, int nth);
    void act     (const llama_moe_db_layer & L, int ith, int nth);
    // out [T][n_embd]: every row of every request row written (zero where a row has no miss)
    void down    (const llama_moe_db_layer & L, int32_t T, float * out, int ith, int nth);
};

// the largest quantized row of a tensor's vec_dot input for rows of n values (0 = no CPU dot product for this type)
size_t llama_moe_db_row_size(ggml_type weight_type, int64_t n);
