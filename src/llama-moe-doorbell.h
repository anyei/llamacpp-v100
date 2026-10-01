#pragma once

// TASKS #154 item 3 (design: docs/strata-port-plan.md section 7): the MoE doorbell.
//
// Decode/verify graphs (n_tokens <= the cache's max_batch) stop computing the cache's misses in a CPU split. Per
// cached layer, GGML_OP_MOE_RING (CUDA) copies the FFN input rows, the miss ids (ids_cpu: expert id, or the skip
// sentinel on cached lanes) and the gating weights into this layer's slot of a pinned host mailbox and publishes the
// graph's step; the GPU goes on with the cache hits while a team of host threads computes the missed experts with
// ggml-cpu's own kernels (llama-moe-doorbell-compute.h) and writes their weighted sum into the slot;
// GGML_OP_MOE_JOIN (CUDA) waits for the slot's DONE word and adds that partial. The decode graph stays one GPU split.
//
// Every graph that rings is one JOB with its own step: begin() queues it (never blocks on or cancels the previous
// one), the step reaches the ring kernels through a graph input, the team serves the jobs in order. Per layer the
// leader alone waits for the ring and publishes the decision (serve / abort) the other threads follow.
//
// LLAMA_MOE_DOORBELL=1 on, =2 timing only (the executor answers zeros: wrong text); _THREADS=N (default -t),
// _SPIN_US=N (how long idle threads spin before they sleep, default 1000), _STATS=N (log every N jobs).

#include "llama-moe-doorbell-compute.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

struct llama_model;
struct llama_moe_cache_layer;
class  llama_moe_cache;

struct llama_moe_doorbell_params {
    int32_t mode        = 0;    // 0 off, 1 on, 2 timing only (zeros)
    int32_t n_threads   = 0;    // executor threads
    int32_t max_tokens  = 8;    // rows per request (the cache's max_batch)
    int32_t spin_us     = 1000; // idle threads spin this long before they sleep (< 0 = never sleep)
    int32_t stats_every = 0;    // log counters every N jobs (0 = never)
};

class llama_moe_doorbell {
public:
    // nullptr when a cached layer's experts cannot be served (reason logged)
    static std::unique_ptr<llama_moe_doorbell> create(const llama_moe_cache & cache, const llama_model & model, const llama_moe_doorbell_params & params);
    ~llama_moe_doorbell();

    // the slot serving this cached layer, -1 = not served
    int32_t slot_of(const llama_moe_cache_layer * l) const;
    int32_t max_tokens() const { return params.max_tokens; }
    int32_t n_used()     const { return n_used_; }

    // step: the graph input carrying this graph's step (I32 [1]), shared by all its layers
    ggml_tensor * build_ring(ggml_context * ctx, int32_t slot, ggml_tensor * x, ggml_tensor * ids_cpu, ggml_tensor * w, ggml_tensor * step) const;
    ggml_tensor * build_join(ggml_context * ctx, int32_t slot, ggml_tensor * a, ggml_tensor * ring) const;

    // queue the job of a graph that rings these slots (graph order); returns the job id. The graph's step input
    // must be set AFTER this call (current_step()).
    int64_t  begin(const std::vector<int32_t> & slots);
    uint32_t current_step() const { return cur_step; }
    // only with no graph in flight (after a synchronize): abandon queued jobs whose rings will never come
    void abort_pending();

private:
    struct slot_info {
        const llama_moe_cache_layer * l = nullptr;
        char *             base = nullptr;
        llama_moe_db_layer layer;
    };
    struct job_t {
        std::vector<int32_t> slots;
        uint32_t             step = 0;
    };
    static constexpr int JOB_CAP = 8;

    llama_moe_doorbell() = default;

    void worker(int ith);
    bool serve(int ith, int32_t slot, uint32_t step, int64_t job, uint64_t & my_go); // false = job abandoned
    bool wait_job(uint64_t k);           // false = stop
    int  wait_go(uint64_t want);         // the leader's verdict for decision #want (1 serve, 0 abort)
    void publish_go(int verdict);
    void barrier();
    void drain();

    llama_moe_doorbell_params params;
    int32_t n_embd_ = 0, n_ff_ = 0, n_used_ = 0;
    int32_t off_x = 0, off_ids = 0, off_w = 0, off_partial = 0;
    size_t  slot_size = 0;

    ggml_backend_buffer_ptr buf;       // the pinned mailbox
    char *                  mailbox = nullptr;
    std::vector<slot_info>  slots;
    std::unordered_map<const llama_moe_cache_layer *, int32_t> by_layer;

    // jobs: a ring of JOB_CAP, begin() waits only if JOB_CAP jobs are outstanding
    job_t                    jobs[JOB_CAP];
    std::atomic<uint64_t>    n_posted{0}, n_done{0};
    std::atomic<int64_t>     abort_upto{-1};    // jobs with id <= this are abandoned when their ring is missing
    uint32_t                 last_step = 0, cur_step = 0;
    std::mutex               mtx;               // job wait + parking
    std::condition_variable  cv_job, cv_go, cv_done;
    std::atomic<int>         n_parked{0};
    std::atomic<bool>        stop{false};
    int32_t                  cur_T_ = 0;       // rows of the layer being served (leader -> team via go_seq)

    // per-layer decision published by the leader
    std::atomic<uint64_t>    go_seq{0};
    std::atomic<int>         go_verdict{0};

    std::vector<std::thread> threads;
    std::atomic<int>         bar_count{0};
    std::atomic<int>         bar_phase{0};

    llama_moe_db_work        work;              // shared scratch of the layer being served

    // counters (leader only)
    uint64_t n_jobs = 0, n_layers_served = 0, n_misses = 0, n_experts = 0, t_wait_us = 0, t_work_us = 0;
    std::vector<int32_t> uniq;
};
