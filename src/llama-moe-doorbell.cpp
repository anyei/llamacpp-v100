// TASKS #154 item 3: the MoE doorbell - see llama-moe-doorbell.h
#include "llama-moe-doorbell.h"

#include "llama-arch.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-moe-cache.h"

#include <algorithm>
#include <chrono>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
static inline void db_pause() { _mm_pause(); }
#else
static inline void db_pause() { std::this_thread::yield(); }
#endif

static inline int64_t db_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static size_t db_align(size_t x, size_t a) { return (x + a - 1) / a * a; }

// mailbox words: plain u32 in pinned host memory the device reads and writes too; acquire / release on the host
static inline uint32_t db_load(const char * p)        { return __atomic_load_n((const uint32_t *) p, __ATOMIC_ACQUIRE); }
static inline void     db_store(char * p, uint32_t v) { __atomic_store_n((uint32_t *) p, v, __ATOMIC_RELEASE); }

// the CPU backend's extra buffer types (repacked layouts, AMX, ...): tensors in them are not in the plain row layout
static std::vector<ggml_backend_buffer_type_t> db_cpu_extra_bufts() {
    std::vector<ggml_backend_buffer_type_t> r;
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu == nullptr) {
        return r;
    }
    auto fn = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(cpu), "ggml_backend_dev_get_extra_bufts");
    if (fn != nullptr) {
        for (ggml_backend_buffer_type_t * p = fn(cpu); p != nullptr && *p != nullptr; ++p) {
            r.push_back(*p);
        }
    }
    return r;
}

static bool db_plain_host(const ggml_tensor * t, const std::vector<ggml_backend_buffer_type_t> & extra) {
    if (t == nullptr || t->buffer == nullptr || t->data == nullptr || !ggml_backend_buffer_is_host(t->buffer)) {
        return false;
    }
    return std::find(extra.begin(), extra.end(), ggml_backend_buffer_get_type(t->buffer)) == extra.end();
}

std::unique_ptr<llama_moe_doorbell> llama_moe_doorbell::create(const llama_moe_cache & cache, const llama_model & model, const llama_moe_doorbell_params & params) {
    if (params.mode <= 0 || cache.n_layers() == 0) {
        return nullptr;
    }
    if (params.max_tokens < 1 || params.max_tokens > LLAMA_MOE_DB_MAX_TOKENS) {
        LLAMA_LOG_WARN("%s: MoE doorbell off: %d rows per request is outside [1, %d]\n", __func__, params.max_tokens, LLAMA_MOE_DB_MAX_TOKENS);
        return nullptr;
    }
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        LLAMA_LOG_WARN("%s: MoE doorbell off: GGML_CUDA_NO_PINNED leaves no device-addressable host memory\n", __func__);
        return nullptr;
    }

    std::unique_ptr<llama_moe_doorbell> db(new llama_moe_doorbell());
    db->params  = params;
    db->n_used_ = (int32_t) model.hparams.n_expert_used;

    // the pools' device gives the pinned host buffer type
    const llama_moe_cache_layer * l0 = cache.layer_at(0);
    ggml_backend_dev_t dev = l0->up_c && l0->up_c->buffer ? ggml_backend_buft_get_device(ggml_backend_buffer_get_type(l0->up_c->buffer)) : nullptr;
    ggml_backend_buffer_type_t host_buft = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
    if (host_buft == nullptr || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU ||
            std::string(ggml_backend_dev_name(dev)).rfind("CUDA", 0) != 0) {
        LLAMA_LOG_WARN("%s: MoE doorbell off: the cache pools are not on a CUDA device with a pinned host buffer type\n", __func__);
        return nullptr;
    }

    const std::vector<ggml_backend_buffer_type_t> extra = db_cpu_extra_bufts();
    size_t max_xq = 0, max_aq = 0;
    for (size_t i = 0; i < cache.n_layers(); ++i) {
        const llama_moe_cache_layer * l = cache.layer_at(i);
        if (!db_plain_host(l->up_src, extra) || !db_plain_host(l->gate_src, extra) || !db_plain_host(l->down_src, extra)) {
            LLAMA_LOG_WARN("%s: MoE doorbell off: layer %d's experts are not plain host tensors\n", __func__, l->il);
            return nullptr;
        }
        const int32_t n_embd = (int32_t) l->up_src->ne[0], n_ff = (int32_t) l->up_src->ne[1];
        if ((db->n_embd_ && (n_embd != db->n_embd_ || n_ff != db->n_ff_)) || l->down_src->ne[0] != n_ff || l->down_src->ne[1] != n_embd ||
                l->gate_src->ne[0] != n_embd || l->gate_src->ne[1] != n_ff) {
            LLAMA_LOG_WARN("%s: MoE doorbell off: layer %d has an unexpected expert shape\n", __func__, l->il);
            return nullptr;
        }
        db->n_embd_ = n_embd;
        db->n_ff_   = n_ff;
        const size_t rs_gate = llama_moe_db_row_size(l->gate_src->type, n_embd);
        const size_t rs_up   = llama_moe_db_row_size(l->up_src->type,   n_embd);
        const size_t rs_down = llama_moe_db_row_size(l->down_src->type, n_ff);
        if (rs_gate == 0 || rs_up == 0 || rs_down == 0) {
            LLAMA_LOG_WARN("%s: MoE doorbell off: layer %d has an expert type without a CPU dot product\n", __func__, l->il);
            return nullptr;
        }
        max_xq = std::max({ max_xq, rs_gate, rs_up });
        max_aq = std::max(max_aq, rs_down);
    }

    const int32_t T = params.max_tokens, U = db->n_used_;
    db->off_x       = GGML_MOE_SLOT_DATA;
    db->off_ids     = db->off_x   + (int32_t) (sizeof(float)   * T * db->n_embd_);
    db->off_w       = db->off_ids + (int32_t) (sizeof(int32_t) * T * U);
    db->off_partial = (int32_t) db_align(db->off_w + sizeof(float) * T * U, 128);
    db->slot_size   = db_align(db->off_partial + sizeof(float) * T * db->n_embd_, 256);

    const size_t size = db->slot_size * cache.n_layers();
    db->buf.reset(ggml_backend_buft_alloc_buffer(host_buft, size));
    if (!db->buf) {
        LLAMA_LOG_WARN("%s: MoE doorbell off: cannot allocate %zu bytes of pinned host memory\n", __func__, size);
        return nullptr;
    }
    db->mailbox = (char *) ggml_backend_buffer_get_base(db->buf.get());
    memset(db->mailbox, 0, size);

    const bool gate_clamp_first = model.arch == LLM_ARCH_DEEPSEEK4;
    for (size_t i = 0; i < cache.n_layers(); ++i) {
        const llama_moe_cache_layer * l = cache.layer_at(i);
        slot_info s;
        s.l    = l;
        s.base = db->mailbox + db->slot_size * i;
        s.layer.up               = l->up_src;
        s.layer.gate             = l->gate_src;
        s.layer.down             = l->down_src;
        s.layer.limit            = l->il >= 0 ? model.hparams.swiglu_clamp_exp[l->il] : 0.0f;
        s.layer.gate_clamp_first = gate_clamp_first;
        db->by_layer[l] = (int32_t) db->slots.size();
        db->slots.push_back(s);
    }
    db->work.reserve(db->n_embd_, db->n_ff_, T, U, max_xq, max_aq);

    const int n_threads = params.mode == 2 ? 1 : std::max(1, params.n_threads);
    db->params.n_threads = n_threads;
    for (int i = 0; i < n_threads; ++i) {
        db->threads.emplace_back([p = db.get(), i] { p->worker(i); });
    }

    LLAMA_LOG_INFO("%s: MoE doorbell %s: %zu layers, %d executor threads (spin %d us), %zu KiB pinned mailbox (%d rows x %d)%s\n",
            __func__, params.mode == 2 ? "TIMING ONLY (zeros)" : "on", db->slots.size(), n_threads, params.spin_us, size/1024,
            T, db->n_embd_, gate_clamp_first ? ", DeepSeek-V4 clamp" : "");
    return db;
}

llama_moe_doorbell::~llama_moe_doorbell() {
    drain();
    {
        std::lock_guard<std::mutex> lock(mtx);
        stop.store(true, std::memory_order_release);
    }
    cv_job.notify_all();
    cv_go.notify_all();
    for (auto & t : threads) {
        t.join();
    }
}

int32_t llama_moe_doorbell::slot_of(const llama_moe_cache_layer * l) const {
    auto it = by_layer.find(l);
    return it == by_layer.end() ? -1 : it->second;
}

ggml_tensor * llama_moe_doorbell::build_ring(ggml_context * ctx, int32_t slot, ggml_tensor * x, ggml_tensor * ids_cpu, ggml_tensor * w, ggml_tensor * step) const {
    return ggml_moe_ring(ctx, x, ids_cpu, w, step, slots[slot].base, off_x, off_ids, off_w);
}

ggml_tensor * llama_moe_doorbell::build_join(ggml_context * ctx, int32_t slot, ggml_tensor * a, ggml_tensor * ring) const {
    return ggml_moe_join(ctx, a, ring, slots[slot].base, off_partial);
}

int64_t llama_moe_doorbell::begin(const std::vector<int32_t> & s) {
    const uint64_t id = n_posted.load(std::memory_order_relaxed);
    if (id - n_done.load(std::memory_order_acquire) >= (uint64_t) JOB_CAP) {
        std::unique_lock<std::mutex> lock(mtx);
        cv_done.wait(lock, [&] { return id - n_done.load(std::memory_order_acquire) < (uint64_t) JOB_CAP; });
    }
    job_t & j = jobs[id % JOB_CAP];
    j.slots = s;
    if (++last_step == 0) {
        ++last_step; // the mailbox words start at 0: a step is never 0
    }
    j.step   = last_step;
    cur_step = last_step;
    {
        std::lock_guard<std::mutex> lock(mtx);
        n_posted.store(id + 1, std::memory_order_release);
    }
    cv_job.notify_all();
    return (int64_t) id;
}

void llama_moe_doorbell::abort_pending() {
    const uint64_t p = n_posted.load(std::memory_order_acquire);
    if (p == 0 || n_done.load(std::memory_order_acquire) >= p) {
        return;
    }
    abort_upto.store((int64_t) p - 1, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(mtx);
    }
    cv_job.notify_all();
    cv_go.notify_all();
    std::unique_lock<std::mutex> lock(mtx);
    cv_done.wait(lock, [&] { return n_done.load(std::memory_order_acquire) >= p; });
}

void llama_moe_doorbell::drain() {
    // the owner synchronized the device first: every ring that will come has come, the rest are abandoned
    abort_pending();
}

bool llama_moe_doorbell::wait_job(uint64_t k) {
    const int64_t t0 = db_us();
    while (n_posted.load(std::memory_order_acquire) <= k) {
        if (stop.load(std::memory_order_acquire)) {
            return false;
        }
        if (params.spin_us >= 0 && db_us() - t0 > params.spin_us) {
            std::unique_lock<std::mutex> lock(mtx);
            cv_job.wait(lock, [&] { return stop || n_posted.load(std::memory_order_acquire) > k; });
            return n_posted.load(std::memory_order_acquire) > k;
        }
        db_pause();
    }
    return true;
}

int llama_moe_doorbell::wait_go(uint64_t want) {
    const int64_t t0 = db_us();
    while (go_seq.load(std::memory_order_seq_cst) < want) {
        if (stop.load(std::memory_order_acquire)) {
            return 0;
        }
        if (params.spin_us >= 0 && db_us() - t0 > params.spin_us) {
            std::unique_lock<std::mutex> lock(mtx);
            n_parked.fetch_add(1, std::memory_order_seq_cst);
            cv_go.wait(lock, [&] { return stop || go_seq.load(std::memory_order_seq_cst) >= want; });
            n_parked.fetch_sub(1, std::memory_order_seq_cst);
            if (go_seq.load(std::memory_order_seq_cst) < want) {
                return 0; // stop
            }
            break;
        }
        db_pause();
    }
    // stable: the next decision is published only after every thread passed this layer's barriers or the job's end
    return go_verdict.load(std::memory_order_acquire);
}

void llama_moe_doorbell::publish_go(int verdict) {
    go_verdict.store(verdict, std::memory_order_release);
    go_seq.fetch_add(1, std::memory_order_seq_cst);
    if (n_parked.load(std::memory_order_seq_cst) > 0) {
        std::lock_guard<std::mutex> lock(mtx);
        cv_go.notify_all();
    }
}

void llama_moe_doorbell::barrier() {
    const int n = (int) threads.size();
    if (n == 1) {
        return;
    }
    const int phase = bar_phase.load(std::memory_order_relaxed);
    if (bar_count.fetch_add(1, std::memory_order_acq_rel) == n - 1) {
        bar_count.store(0, std::memory_order_relaxed);
        bar_phase.fetch_add(1, std::memory_order_release);
        return;
    }
    while (bar_phase.load(std::memory_order_acquire) == phase) {
        db_pause();
    }
}

void llama_moe_doorbell::worker(int ith) {
    uint64_t my_job = 0, my_go = 0;
    while (wait_job(my_job)) {
        const job_t & j = jobs[my_job % JOB_CAP];
        for (int32_t slot : j.slots) {
            if (!serve(ith, slot, j.step, (int64_t) my_job, my_go)) {
                break;
            }
        }
        barrier(); // the job's end: every thread is out of it before its entry can be reused
        if (ith == 0) {
            n_done.store(my_job + 1, std::memory_order_release);
            {
                std::lock_guard<std::mutex> lock(mtx);
            }
            cv_done.notify_all();
            if (params.stats_every > 0 && ++n_jobs % params.stats_every == 0) {
                LLAMA_LOG_INFO("moe-doorbell: jobs %llu layers %llu misses %llu (%.2f/layer, %.2f experts/layer) wait %.1f ms/job work %.2f ms/job\n",
                        (unsigned long long) n_jobs, (unsigned long long) n_layers_served, (unsigned long long) n_misses,
                        n_layers_served ? (double) n_misses / n_layers_served : 0.0,
                        n_layers_served ? (double) n_experts / n_layers_served : 0.0,
                        t_wait_us / 1000.0 / n_jobs, t_work_us / 1000.0 / n_jobs);
            }
        }
        my_job++;
    }
}

bool llama_moe_doorbell::serve(int ith, int32_t slot, uint32_t step, int64_t job, uint64_t & my_go) {
    slot_info & s = slots[slot];
    const int nth = (int) threads.size();
    my_go++;

    // the leader alone waits for this layer's request and decides for everyone (serve / abandon the job)
    int     verdict = 1;
    int64_t t_w0 = 0, t_w1 = 0;
    if (ith == 0) {
        t_w0 = db_us();
        while (db_load(s.base + GGML_MOE_SLOT_RING) != step) {
            if (abort_upto.load(std::memory_order_acquire) >= job) {
                verdict = 0;
                break;
            }
            db_pause();
        }
        t_w1 = db_us();
        if (verdict) {
            const int32_t T = (int32_t) db_load(s.base + GGML_MOE_SLOT_NTOK);
            if (params.mode == 2) { // timing only (one thread): the partial stays zero
                db_store(s.base + GGML_MOE_SLOT_DONE, step);
                n_layers_served++;
                t_wait_us += t_w1 - t_w0;
                return true;
            }
            if (!work.collect(s.layer, (const int32_t *) (s.base + off_ids), (const float *) (s.base + off_w), T)) {
                GGML_ABORT("moe doorbell: corrupt request in slot %d (rows %d)", slot, T);
            }
            n_misses += work.misses.size();
            uniq.clear();
            for (const auto & ms : work.misses) {
                uniq.push_back(ms.e);
            }
            std::sort(uniq.begin(), uniq.end());
            n_experts += std::unique(uniq.begin(), uniq.end()) - uniq.begin();
            cur_T_ = T;
        }
        publish_go(verdict);
    } else {
        verdict = wait_go(my_go);
    }
    if (!verdict) {
        return false;
    }

    const int32_t T = cur_T_;
    float * partial = (float *) (s.base + off_partial);
    if (work.misses.empty()) {
        if (ith == 0) {
            memset(partial, 0, sizeof(float) * (size_t) T * n_embd_);
        }
    } else {
        work.quantize(s.layer, (const float *) (s.base + off_x), T, ith, nth);
        barrier();
        work.gate_up(s.layer, ith, nth);
        barrier();
        work.act(s.layer, ith, nth);
        barrier();
        work.down(s.layer, T, partial, ith, nth);
    }
    barrier();

    if (ith == 0) {
        db_store(s.base + GGML_MOE_SLOT_DONE, step);
        n_layers_served++;
        t_wait_us += t_w1 - t_w0;
        t_work_us += db_us() - t_w1;
    }
    return true;
}
