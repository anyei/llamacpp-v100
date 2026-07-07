#include "ssd-streaming.h"

#include "common.h"
#include "log.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#define COMMON_SSD_STREAMING_SUPPORTED 1
#endif

// mmap streaming director (task 15 phase 1, docs/ssd-streaming-plan.md).
//
// With mmap loading, CPU-resident weights are file-backed pages: correctness is
// the kernel's problem (pages fault in on first touch), but an over-committed
// model thrashes under default LRU reclaim (measured 0.21 t/s vs 1.51 t/s
// resident for the 27B on this box). This director rides the scheduler's eval
// callback as a progress signal and steers residency explicitly:
//
//   - WILLNEED  a budgeted window of upcoming weights (kernel readahead pulls
//     them from disk while the current layer computes)
//   - DONTNEED  weights far behind the cursor (bounds the page-cache footprint
//     instead of letting reclaim evict the wrong pages)
//
// The weight order is discovered on the first pass (graph execution order) and
// treated as a ring after that - decode repeats the same order every token.
// Computation is NEVER altered: madvise is advisory, so output is byte-identical
// with the director on or off.

namespace {

struct ssd_stream_entry {
    const void * addr;
    size_t       nbytes;
};

struct ssd_stream_state {
    std::mutex mutex;

    std::vector<ssd_stream_entry>            order;    // weights in first-use order
    std::unordered_map<const void *, size_t> index_of; // tensor data -> position in order
    bool   ring_complete = false; // first full pass done, order is now stable
    size_t cursor        = 0;     // position of the weight most recently used
    size_t ahead_target  = 0;     // how far WILLNEED has been issued (ring position)

    size_t budget_bytes  = 0;
    size_t min_tensor    = 1024 * 1024; // ignore small tensors (norms, biases)
    long   page_size     = 4096;
};

ssd_stream_state g_state;

void ssd_madvise(const void * addr, size_t nbytes, int advice) {
#ifdef COMMON_SSD_STREAMING_SUPPORTED
    const uintptr_t a     = (uintptr_t) addr;
    const uintptr_t start = a & ~(uintptr_t)(g_state.page_size - 1);
    const uintptr_t stop  = a + nbytes;
    if (stop <= start) {
        return;
    }
    posix_madvise((void *) start, stop - start, advice);
#else
    GGML_UNUSED(addr); GGML_UNUSED(nbytes); GGML_UNUSED(advice);
#endif
}

// issue WILLNEED forward of the cursor and DONTNEED behind it, keeping the
// resident window within budget. Called with the state mutex held.
void ssd_stream_advance(size_t pos) {
    auto & st = g_state;
    st.cursor = pos;

    const size_t n = st.order.size();
    if (n == 0) {
        return;
    }

    // walk forward from the cursor until the look-ahead budget is spent
    size_t ahead_bytes = 0;
    for (size_t k = 1; k < n; k++) {
        const size_t i = (pos + k) % n;
        if (!st.ring_complete && pos + k >= n) {
            break; // order beyond this point is not known yet
        }
        ahead_bytes += st.order[i].nbytes;
        if (ahead_bytes > st.budget_bytes / 2) {
            break;
        }
        ssd_madvise(st.order[i].addr, st.order[i].nbytes, POSIX_MADV_WILLNEED);
    }

    // drop pages far behind: keep roughly half the budget of recently-used
    // weights resident (they are also the next token's tail of the ring)
    if (st.ring_complete) {
        size_t behind_bytes = 0;
        for (size_t k = 0; k < n; k++) {
            const size_t i = (pos + n - k) % n;
            behind_bytes += st.order[i].nbytes;
            if (behind_bytes > st.budget_bytes / 2) {
                ssd_madvise(st.order[i].addr, st.order[i].nbytes, POSIX_MADV_DONTNEED);
            }
        }
    }
}

} // namespace

// MoE expert steering (task 15 phase 4): expert weights are one 3D tensor per
// layer with ne[2] = n_expert, but each token touches only the selected experts'
// slices. Under memory pressure the kernel serves those as scattered small
// faults (measured: 10.9 -> 1.1 t/s for a 14 GB MoE at a 10 GB cap). On every
// MUL_MAT_ID we read the (tiny, host-resident) selection ids and WILLNEED the
// selected experts' contiguous slices - one batched readahead per expert
// instead of hundreds of page faults. Expert tensors are excluded from the
// generic ring: prefetching ALL experts would flood the budget with unused
// weights, and eviction is left to kernel LRU, which approximates hot-expert
// retention at page granularity.
static void ssd_stream_experts(const struct ggml_tensor * t) {
    const struct ggml_tensor * w   = t->src[0]; // experts weight [.., .., n_expert]
    const struct ggml_tensor * ids = t->src[2]; // selected experts [n_expert_used, n_tokens]
    if (w == nullptr || ids == nullptr || ids->type != GGML_TYPE_I32) {
        return;
    }
    if (w->buffer == nullptr || !ggml_backend_buffer_is_host(w->buffer) || w->op != GGML_OP_NONE) {
        return;
    }
    if (ids->data == nullptr || ids->buffer == nullptr || !ggml_backend_buffer_is_host(ids->buffer)) {
        return; // ids must be directly readable (they are, for CPU inference)
    }
    const int64_t n_expert  = w->ne[2];
    const size_t  slice     = w->nb[2];
    const int64_t n_ids     = ggml_nelements(ids);
    const int32_t * id_data = (const int32_t *) ids->data;
    uint64_t seen = 0; // n_expert <= 64 covers current models; extras just re-advise
    for (int64_t k = 0; k < n_ids; k++) {
        const int32_t e = id_data[k];
        if (e < 0 || e >= n_expert) {
            continue;
        }
        if (e < 64) {
            if (seen & (1ull << e)) {
                continue;
            }
            seen |= 1ull << e;
        }
        ssd_madvise((const char *) w->data + (size_t) e * slice, slice, POSIX_MADV_WILLNEED);
    }
}

bool common_ssd_streaming_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    GGML_UNUSED(user_data);
    if (!ask) {
        return true; // we never request tensor data; only the progress ticks matter
    }

    auto & st = g_state;
    std::lock_guard<std::mutex> lock(st.mutex);

    if (t->op == GGML_OP_MUL_MAT_ID) {
        ssd_stream_experts(t);
    }

    for (int i = 0; i < GGML_MAX_SRC; i++) {
        const struct ggml_tensor * src = t->src[i];
        if (src == nullptr || src->op != GGML_OP_NONE || src->data == nullptr) {
            continue;
        }
        if (src->buffer == nullptr || !ggml_backend_buffer_is_host(src->buffer)) {
            continue; // only file-backed host weights are streamable
        }
        if (src->ne[2] > 1 && t->op == GGML_OP_MUL_MAT_ID) {
            continue; // expert tensors are steered per-selection, never as a whole
        }
        const size_t nbytes = ggml_nbytes(src);
        if (nbytes < st.min_tensor) {
            continue;
        }
        auto it = st.index_of.find(src->data);
        if (it == st.index_of.end()) {
            if (st.ring_complete) {
                continue; // stable order should not grow; ignore stragglers
            }
            st.index_of[src->data] = st.order.size();
            st.order.push_back({ src->data, nbytes });
            ssd_stream_advance(st.order.size() - 1);
        } else {
            if (!st.ring_complete && it->second == 0 && st.cursor + 1 >= st.order.size()) {
                st.ring_complete = true; // wrapped back to the first weight
                LOG_INF("%s: weight ring complete: %zu tensors\n", __func__, st.order.size());
            }
            ssd_stream_advance(it->second);
        }
    }
    return false; // do not fetch any data, do not force extra syncs
}

bool common_ssd_streaming_init(common_params & params) {
    const char * env = getenv("LLAMA_SSD_STREAMING");
    if (env == nullptr || atoi(env) == 0) {
        return false;
    }
#ifndef COMMON_SSD_STREAMING_SUPPORTED
    LOG_WRN("%s: LLAMA_SSD_STREAMING is not supported on this platform\n", __func__);
    return false;
#else
    if (params.cb_eval != nullptr) {
        LOG_WRN("%s: eval callback already in use - SSD streaming disabled\n", __func__);
        return false;
    }
    auto & st = g_state;
    st.page_size = sysconf(_SC_PAGESIZE);
    const char * budget = getenv("LLAMA_SSD_STREAM_BUDGET");
    st.budget_bytes = budget ? (size_t) atoll(budget) * 1024ull * 1024ull : 4096ull * 1024ull * 1024ull;

    params.cb_eval           = common_ssd_streaming_cb;
    params.cb_eval_user_data = nullptr;
    // the director manages residency itself: do not prefault the whole file
    params.use_mmap = true;
    params.use_mlock = false;

    LOG_INF("%s: SSD streaming director enabled (budget %zu MiB)\n", __func__, st.budget_bytes / (1024*1024));
    return true;
#endif
}
