#pragma once

// MoE expert cache (TASKS #151, docs/moe-cache-plan.md section 5).
//
// Routed experts that -ncmoe / -ot keep in host memory get a per-layer pool of
// VRAM slots on the device that holds the layer's router. The graph runs two
// chains over the same input: a GPU chain over the slot tensors and the stock
// CPU chain over the host tensors. Both chains take their ids through an I32
// table that maps every expert to either a slot (GPU chain) or its own id (CPU
// chain), with the other side set to the negative skip sentinel, so each lane is
// computed on exactly one chain and the two down outputs simply add.
//
// Residency changes only in step(), called between decodes with no graph in
// flight: completed uploads are published (tables written), victims are evicted
// (table entry cleared before any graph can read the slot again) and their
// replacement copies are handed to a worker thread that uploads on its own
// backend. A miss is never fetched for the current token.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

struct llama_model;

struct llama_moe_cache_params {
    int32_t budget_mib   = 0;   // total VRAM for slots, split across cached layers; 0 = off, -1 = auto
    int32_t reserve_mib  = 3072; // VRAM kept free on the pools' device; a fixed budget is clamped to free - reserve
    int32_t max_batch    = 8;   // nodes wider than this stay on the stock path
    int32_t min_slots    = 8;   // a layer that cannot hold this many experts is not cached
    int32_t inserts      = 4;   // max uploads scheduled per layer per step
    int32_t step_mib     = 384; // max upload bytes scheduled per step, all layers (round-robin over layers)
    int32_t hot_uses     = 4;   // a slot with more resident hits than this is evicted last
    int32_t stats_every  = 0;   // log counters every N steps (0 = never)
    int32_t static_fill  = 0;   // fill experts [0, N) of every layer at create (gate instrument)
    bool    force_cpu    = false; // allocate the pools on the host buffer type (wiring gate)
    bool    debug        = false;
};

struct llama_moe_cache_layer {
    int32_t il       = -1;
    int64_t n_expert = 0;
    int32_t n_slots  = 0;

    // host-resident sources (the authoritative experts)
    ggml_tensor * up_src   = nullptr;
    ggml_tensor * gate_src = nullptr;
    ggml_tensor * down_src = nullptr;

    // device slot pools [ne0, ne1, n_slots + 1]; the last slab is zero padding
    ggml_tensor * up_c   = nullptr;
    ggml_tensor * gate_c = nullptr;
    ggml_tensor * down_c = nullptr;

    // I32 [n_expert]: gpu_table = slot or SKIP; cpu_table = expert id or SKIP (cached)
    ggml_tensor * gpu_table = nullptr;
    ggml_tensor * cpu_table = nullptr;

    std::vector<int32_t>  slot_expert;    // slot -> expert, -1 = free
    std::vector<int32_t>  expert_slot;    // expert -> slot, -1 = not resident
    std::vector<uint64_t> slot_last_use;  // step clock of the last hit
    std::vector<uint16_t> slot_hits;      // resident hits (aged)
    std::vector<uint8_t>  slot_in_flight; // an upload targets this slot
    std::vector<uint16_t> miss_count;     // per expert, reset on admission
    std::vector<uint8_t>  expert_pending; // an upload for this expert is queued

    // routing observations copied out after each graph (flat expert ids)
    std::vector<int32_t> obs;
    size_t               obs_len = 0;

    ggml_backend_t          fill_backend = nullptr; // owned by the cache (backends[])
    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf;
};

class llama_moe_cache {
public:
    // nullptr when the cache is off or no layer qualifies (reason logged)
    static std::unique_ptr<llama_moe_cache> create(const llama_model & model, const llama_moe_cache_params & params);

    ~llama_moe_cache();

    // the cached layer whose up_exps is this tensor, or nullptr
    const llama_moe_cache_layer * lookup(const ggml_tensor * up_exps) const;

    int32_t max_batch() const { return params.max_batch; }
    size_t  n_layers()  const { return layers.size(); }
    const llama_moe_cache_layer * layer_at(size_t i) const { return layers[i].get(); } // TASKS #154 item 3 (doorbell)

    // enqueue the async copy of this graph's routed ids (call right after the graph was computed)
    void observe(ggml_backend_sched_t sched, const std::vector<std::pair<int, ggml_tensor *>> & ids);

    // apply pending residency changes; call only between graph executions, after a synchronize
    void step();

private:
    struct job {
        size_t  layer;
        int32_t expert;
        int32_t slot;
    };

    llama_moe_cache() = default;

    bool alloc_layer(llama_moe_cache_layer & l, ggml_backend_buffer_type_t buft, int32_t n_slots);
    void fill_slot(llama_moe_cache_layer & l, int32_t expert, int32_t slot);
    void copy_slot(ggml_backend_t backend, llama_moe_cache_layer & l, int32_t expert, int32_t slot);
    void flush_tables(llama_moe_cache_layer & l);
    void worker_loop();
    void log_stats();

    llama_moe_cache_params params;
    std::vector<std::unique_ptr<llama_moe_cache_layer>> layers;
    std::unordered_map<const ggml_tensor *, const llama_moe_cache_layer *> by_up;
    std::unordered_map<int, llama_moe_cache_layer *> by_il;

    // upload backends, one per device (the pools' devices), owned here
    std::vector<ggml_backend_ptr> backends;

    // worker
    std::thread             worker;
    std::mutex              mtx;
    std::condition_variable cv;
    std::deque<job>         todo;
    std::vector<job>        done;
    bool                    stop = false;

    // counters
    uint64_t n_steps = 0, n_hits = 0, n_misses = 0, n_fills = 0, n_evict = 0, n_obs_dropped = 0;
};
