#include "ggml-ssd-stream.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <list>
#include <map>
#include <mutex>
#include <unordered_map>
#include <utility>

// SSD expert-streaming buffer type — see ggml-ssd-stream.h / docs/ssd-streaming-plan.md.
//
// Increment 1 (CPU-landing): experts live in a sparse anonymous arena at their
// natural offsets; a per-expert slice is pread() from the model file into the
// arena just before the MUL_MAT_ID that consumes it, cached under a byte budget,
// and MADV_DONTNEED-evicted when the budget is exceeded. Correct and RSS-bounded.
//
// Increment 2 (this): the resident cache defaults to plain LRU; an optional
// segmented LRU (SLRU) is available via LLAMA_SSD_STREAM_SLRU=1 - slices seen
// once sit in a probation segment, a second hit promotes to a protected segment,
// and eviction always takes the probation tail first, so a burst of one-time
// experts churns only probation and never displaces a hot expert.
// LLAMA_SSD_STREAM_PROTECTED_PCT sizes the protected segment (default 80%).
// Measured neutral-to-slightly-worse than LRU for DeepSeek-V4 (expert reuse not
// skewed enough); kept opt-in for more skewed workloads. See docs/ssd-streaming-plan.md.

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#define GGML_SSD_STREAM_SUPPORTED 1
#endif

namespace {

struct slice_key {
    const ggml_tensor * t;
    int                 e;
    bool operator<(const slice_key & o) const {
        return t != o.t ? t < o.t : e < o.e;
    }
};

struct slice_node {
    slice_key key;
    void *    addr;
    size_t    len;
};

// index entry: which segment the slice lives in, and its node iterator.
struct slice_ref {
    bool                            prot; // true = protected segment
    std::list<slice_node>::iterator it;
};

struct tensor_backing {
    int    fd;   // dup'd, kept open for process lifetime
    size_t offs; // byte offset of the tensor's data in the file
};

struct ssd_stream_state {
    std::mutex mutex;

    bool   enabled_checked = false;
    bool   enabled         = false;
    bool   debug           = false;
    bool   odirect         = false;   // reads bypass the page cache (set when the file fd takes O_DIRECT)
    bool   no_odirect      = false;   // force buffered reads (kill-switch / A-B diagnostic)
    bool   slru            = true;     // segmented LRU (probationary+protected); false = plain LRU
    size_t budget_bytes    = 0;
    size_t protected_cap   = 0;        // byte cap of the protected segment (slru only)
    long   page_size       = 4096;

    // aligned bounce buffer for O_DIRECT reads (reused; guarded by mutex during fill)
    void * bounce      = nullptr;
    size_t bounce_cap  = 0;

    std::unordered_map<const ggml_tensor *, tensor_backing> registry;
    std::unordered_map<int, int>                            fd_dup;      // orig fd -> dup fd

    // resident expert slices as a segmented LRU, bounded by budget_bytes.
    // probation holds slices seen once; a second hit promotes to protected
    // (bounded by protected_cap, overflow demotes back). Eviction takes the
    // probation LRU tail first, so a scan of one-time experts never displaces a
    // hot expert. Plain LRU (slru=false) skips promotion: everything stays in
    // probation, which then behaves as a single global LRU. Both lists MRU-front.
    std::list<slice_node>                probation;
    std::list<slice_node>                protectd;
    size_t                               protected_bytes = 0;
    std::map<slice_key, slice_ref>       index;
    size_t                               resident_bytes  = 0;

    // stats
    uint64_t n_hit = 0, n_miss = 0, n_evict = 0, n_promote = 0, bytes_read = 0;
};

ssd_stream_state g_state;

#ifdef GGML_SSD_STREAM_SUPPORTED
// plain buffered pread of [off, off+len) into dst (loops over short reads)
bool ssd_pread_buffered(int fd, char * dst, size_t len, size_t off) {
    size_t done = 0;
    while (done < len) {
        ssize_t r = pread(fd, dst + done, len - done, (off_t) (off + done));
        if (r < 0) {
            GGML_LOG_ERROR("%s: pread failed (fd=%d off=%zu len=%zu): %s\n",
                    __func__, fd, off + done, len - done, strerror(errno));
            return false;
        }
        if (r == 0) {
            break; // EOF
        }
        done += (size_t) r;
    }
    return done >= len;
}
#endif

// read [off, off+len) from fd into dst. With O_DIRECT the read is issued to a
// page-aligned bounce buffer over the aligned superset and the exact range is
// copied out - so streaming never populates the page cache (keeps RSS/cgroup
// charge bounded to the arena, unlike buffered reads). Caller holds the mutex.
void ssd_pread_full(int fd, void * dst, size_t len, size_t off) {
#ifdef GGML_SSD_STREAM_SUPPORTED
    auto & st = g_state;
    if (!st.odirect) {
        if (ssd_pread_buffered(fd, (char *) dst, len, off)) {
            st.bytes_read += len;
        }
        return;
    }
    const size_t A    = (size_t) st.page_size;
    const size_t aoff = off & ~(A - 1);
    const size_t aend = (off + len + A - 1) & ~(A - 1);
    const size_t asz  = aend - aoff;
    if (st.bounce_cap < asz) {
        free(st.bounce);
        st.bounce = nullptr;
        if (posix_memalign(&st.bounce, A, asz) != 0) {
            st.bounce = nullptr; st.bounce_cap = 0;
            GGML_LOG_ERROR("%s: posix_memalign(%zu) failed\n", __func__, asz);
            return;
        }
        st.bounce_cap = asz;
    }
    // O_DIRECT requires aligned offset/buffer; the kernel returns a short (but
    // still page-multiple) count only at EOF, which is fine as long as it covers
    // the requested tail.
    const size_t need = off + len - aoff;
    size_t done = 0;
    while (done < need) {
        ssize_t r = pread(fd, (char *) st.bounce + done, asz - done, (off_t) (aoff + done));
        if (r < 0) {
            GGML_LOG_ERROR("%s: O_DIRECT pread failed (fd=%d aoff=%zu asz=%zu): %s\n",
                    __func__, fd, aoff, asz - done, strerror(errno));
            return;
        }
        if (r == 0) {
            break; // EOF
        }
        done += (size_t) r;
    }
    memcpy(dst, (const char *) st.bounce + (off - aoff), len);
    st.bytes_read += len;
#else
    GGML_UNUSED(fd); GGML_UNUSED(dst); GGML_UNUSED(len); GGML_UNUSED(off);
#endif
}

// drop only whole pages fully inside [addr, addr+len) so a shared boundary page
// belonging to a still-resident neighbour slice is never zeroed.
void ssd_evict(const slice_node & s) {
#ifdef GGML_SSD_STREAM_SUPPORTED
    const long   pg    = g_state.page_size;
    const uintptr_t a  = (uintptr_t) s.addr;
    const uintptr_t st = (a + pg - 1) & ~(uintptr_t) (pg - 1); // round up
    const uintptr_t en = (a + s.len)  & ~(uintptr_t) (pg - 1); // round down
    if (en > st) {
        madvise((void *) st, en - st, MADV_DONTNEED);
    }
    g_state.n_evict++;
#else
    GGML_UNUSED(s);
#endif
}

// demote protected LRU tails back to the probation MRU until protected fits its
// cap; caller holds g_state.mutex. Splices preserve node iterators.
void ssd_demote_protected(void) {
    auto & st = g_state;
    while (st.protected_bytes > st.protected_cap && !st.protectd.empty()) {
        auto tail = std::prev(st.protectd.end());
        st.protected_bytes -= tail->len;
        st.probation.splice(st.probation.begin(), st.protectd, tail);
        st.index[tail->key] = slice_ref{ false, st.probation.begin() };
    }
}

// ensure expert slice (t,e) at [addr,addr+len) backed by file bytes at file_off
// is resident; caller holds g_state.mutex.
void ssd_touch(const ggml_tensor * t, int e, void * addr, size_t len, int fd, size_t file_off) {
    auto &  st  = g_state;
    slice_key key{ t, e };
    auto it = st.index.find(key);
    if (it != st.index.end()) {
        slice_ref & ref = it->second;
        if (ref.prot) {
            st.protectd.splice(st.protectd.begin(), st.protectd, ref.it); // -> protected MRU
        } else if (st.slru) {
            // second hit: promote probation -> protected MRU (node iterator survives splice)
            st.protectd.splice(st.protectd.begin(), st.probation, ref.it);
            ref.prot = true;
            st.protected_bytes += len;
            st.n_promote++;
            ssd_demote_protected();
        } else {
            st.probation.splice(st.probation.begin(), st.probation, ref.it); // plain LRU: -> MRU
        }
        st.n_hit++;
        return;
    }
    // miss: evict the probation LRU tail (then protected tail as a last resort)
    // until the new slice fits the budget.
    while (st.resident_bytes + len > st.budget_bytes &&
           (!st.probation.empty() || !st.protectd.empty())) {
        const bool from_prot = st.probation.empty();
        std::list<slice_node> & seg = from_prot ? st.protectd : st.probation;
        auto tail = std::prev(seg.end());
        ssd_evict(*tail);
        st.resident_bytes -= tail->len;
        if (from_prot) {
            st.protected_bytes -= tail->len;
        }
        st.index.erase(tail->key);
        seg.pop_back();
    }
    ssd_pread_full(fd, addr, len, file_off);
    st.probation.push_front(slice_node{ key, addr, len });
    st.index[key] = slice_ref{ false, st.probation.begin() };
    st.resident_bytes += len;
    st.n_miss++;
}

// ---- buffer type / buffer iface -------------------------------------------

struct ssd_buffer_context {
    void * base;
    size_t size;
};

const char * ssd_buft_get_name(ggml_backend_buffer_type_t) {
    return "SSD_Stream";
}

void ssd_buffer_free(ggml_backend_buffer_t buffer) {
    auto * ctx = (ssd_buffer_context *) buffer->context;
#ifdef GGML_SSD_STREAM_SUPPORTED
    if (ctx->base) {
        munmap(ctx->base, ctx->size);
    }
#endif
    delete ctx;
}

void * ssd_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ((ssd_buffer_context *) buffer->context)->base;
}

void ssd_buffer_set_tensor(ggml_backend_buffer_t, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    memcpy((char *) tensor->data + offset, data, size); // rare path; keeps generic tensor_set correct
}

void ssd_buffer_get_tensor(ggml_backend_buffer_t, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    memcpy(data, (const char *) tensor->data + offset, size);
}

void ssd_buffer_clear(ggml_backend_buffer_t, uint8_t) {
    // no-op: the arena is zero-filled by MAP_ANONYMOUS and holds read-only
    // weights; a real clear would fault the entire (multi-GB) arena.
}

ggml_backend_buffer_t ssd_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);

size_t ssd_buft_get_alignment(ggml_backend_buffer_type_t) {
    return 4096; // page-aligned tensors so eviction can drop whole pages
}

bool ssd_buft_is_host(ggml_backend_buffer_type_t) {
    return true; // CPU kernels read tensor->data directly
}

const ggml_backend_buffer_i ssd_buffer_iface = {
    /* .free_buffer   = */ ssd_buffer_free,
    /* .get_base      = */ ssd_buffer_get_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ nullptr,
    /* .set_tensor    = */ ssd_buffer_set_tensor,
    /* .get_tensor    = */ ssd_buffer_get_tensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ nullptr,
    /* .clear         = */ ssd_buffer_clear,
    /* .reset         = */ nullptr,
    /* .set_usage     = */ nullptr,
};

ggml_backend_buffer_type ssd_buft = {
    /* .iface = */ {
        /* .get_name       = */ ssd_buft_get_name,
        /* .alloc_buffer   = */ ssd_buft_alloc_buffer,
        /* .get_alignment  = */ ssd_buft_get_alignment,
        /* .get_max_size   = */ nullptr,
        /* .get_alloc_size = */ nullptr,
        /* .is_host        = */ ssd_buft_is_host,
    },
    /* .device  = */ nullptr,
    /* .context = */ nullptr,
};

ggml_backend_buffer_t ssd_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto * ctx = new ssd_buffer_context{ nullptr, size };
#ifdef GGML_SSD_STREAM_SUPPORTED
    // sparse: MAP_NORESERVE so the full (multi-GB) virtual span costs no commit;
    // physical pages appear only where slices are pread in, and are released on
    // MADV_DONTNEED eviction.
    void * base = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (base == MAP_FAILED) {
        GGML_LOG_ERROR("%s: mmap of %zu bytes failed: %s\n", __func__, size, strerror(errno));
        delete ctx;
        return nullptr;
    }
    ctx->base = base;
    GGML_LOG_INFO("ggml_ssd_stream: arena reserved %.2f GB (virtual, MAP_NORESERVE)\n", size / 1e9);
#endif
    return ggml_backend_buffer_init(buft, ssd_buffer_iface, ctx, size);
}

} // namespace

// ---- public API ------------------------------------------------------------

bool ggml_ssd_stream_enabled(void) {
#ifdef GGML_SSD_STREAM_SUPPORTED
    auto & st = g_state;
    if (!st.enabled_checked) {
        const char * env = getenv("LLAMA_SSD_STREAM_BUFFER");
        st.enabled = env != nullptr && atoi(env) != 0;
        const char * b = getenv("LLAMA_SSD_STREAM_BUDGET");
        st.budget_bytes = (b ? (size_t) atoll(b) : 8192ull) * 1024ull * 1024ull;
        st.no_odirect = getenv("LLAMA_SSD_STREAM_NO_ODIRECT") != nullptr;
        // Plain LRU is the default: SLRU was measured neutral-to-slightly-worse
        // than LRU for DeepSeek-V4 MoE streaming (30GB budget: 73.4% vs 73.6%
        // hit; 8GB: 54.0% vs 52.9%) - expert reuse is not skewed enough for the
        // protected segment to earn its complexity. Kept opt-in for skewed
        // workloads. LLAMA_SSD_STREAM_SLRU=1 enables it.
        const char * s = getenv("LLAMA_SSD_STREAM_SLRU");
        st.slru = s != nullptr && atoi(s) != 0;
        const char * p = getenv("LLAMA_SSD_STREAM_PROTECTED_PCT");
        long pct = p ? atol(p) : 80;
        if (pct < 0)   pct = 0;
        if (pct > 100) pct = 100;
        st.protected_cap = st.slru ? (size_t) ((double) st.budget_bytes * pct / 100.0) : 0;
        st.page_size = sysconf(_SC_PAGESIZE);
        st.enabled_checked = true;
        if (st.enabled) {
            st.debug = getenv("GGML_SSD_STREAM_DEBUG") != nullptr;
            GGML_LOG_INFO("ggml_ssd_stream: enabled, expert-cache budget %zu MiB, policy %s\n",
                    st.budget_bytes / (1024 * 1024),
                    st.slru ? "SLRU" : "LRU");
        }
    }
    return st.enabled;
#else
    return false;
#endif
}

bool ggml_ssd_stream_should_stream(const char * name) {
    // MoE expert weight tensors: blk.N.ffn_(gate|up|down)_exps.weight.
    // "_exps" excludes the small always-active shared experts ("_shexp").
    return name != nullptr && strstr(name, "_exps") != nullptr;
}

ggml_backend_buffer_type_t ggml_ssd_stream_buft(void) {
    if (ssd_buft.device == nullptr) {
        // schedule streamed-expert ops on the CPU backend (host weights)
        ssd_buft.device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    return &ssd_buft;
}

bool ggml_ssd_stream_is_streamed(const ggml_tensor * t) {
    return t != nullptr && t->buffer != nullptr &&
           ggml_backend_buffer_get_type(t->buffer) == &ssd_buft;
}

void ggml_ssd_stream_note(const ggml_tensor * t, int fd, size_t file_offset) {
#ifdef GGML_SSD_STREAM_SUPPORTED
    auto & st = g_state;
    std::lock_guard<std::mutex> lock(st.mutex);
    int dfd;
    auto f = st.fd_dup.find(fd);
    if (f == st.fd_dup.end()) {
        dfd = dup(fd); // survives loader teardown
        // switch the dup to O_DIRECT so streamed reads bypass the page cache
        // (keeps the cgroup/page-cache charge bounded to the arena). Falls back
        // to buffered reads if the fs/file rejects it or LLAMA_SSD_STREAM_NO_ODIRECT.
        const int fl = st.no_odirect ? -1 : fcntl(dfd, F_GETFL);
        if (fl != -1 && fcntl(dfd, F_SETFL, fl | O_DIRECT) == 0) {
            st.odirect = true;
        } else if (st.fd_dup.empty()) {
            GGML_LOG_INFO("ggml_ssd_stream: %s, using buffered reads\n",
                    st.no_odirect ? "O_DIRECT disabled (LLAMA_SSD_STREAM_NO_ODIRECT)" : "O_DIRECT unavailable");
        }
        st.fd_dup[fd] = dfd;
    } else {
        dfd = f->second;
    }
    st.registry[t] = tensor_backing{ dfd, file_offset };
#else
    GGML_UNUSED(t); GGML_UNUSED(fd); GGML_UNUSED(file_offset);
#endif
}

void ggml_ssd_stream_prefill_experts(const ggml_tensor * w, const uint32_t * used_ids, int64_t n_expert) {
#ifdef GGML_SSD_STREAM_SUPPORTED
    if (!ggml_ssd_stream_enabled() || w == nullptr || n_expert <= 0) {
        return;
    }
    auto & st = g_state;
    std::lock_guard<std::mutex> lock(st.mutex);
    auto reg = st.registry.find(w);
    if (reg == st.registry.end()) {
        return; // not a streamed expert tensor
    }
    const size_t slice = w->nb[2];
    char *       base  = (char *) w->data;
    const int    fd    = reg->second.fd;
    const size_t offs  = reg->second.offs;
    for (int64_t e = 0; e < n_expert; e++) {
        if (used_ids && !ggml_bitset_get(used_ids, e)) {
            continue;
        }
        ssd_touch(w, (int) e, base + (size_t) e * slice, slice, fd, offs + (size_t) e * slice);
    }
#else
    GGML_UNUSED(w); GGML_UNUSED(used_ids); GGML_UNUSED(n_expert);
#endif
}

bool ggml_ssd_stream_eval_cb(ggml_tensor * t, bool ask, void * user_data) {
    GGML_UNUSED(user_data);
    if (!ask) {
        return true;
    }
    // A streamed MUL_MAT_ID reads its expert-selection ids (src[2]) at fill time,
    // so those ids must already be computed. When non-experts are offloaded
    // (-ngl > 0), the router that produces the ids runs in an earlier scheduler
    // split (different backend) and is done by the time the CPU expert node's
    // callback fires — so we can let every other node batch normally and only
    // force a boundary at the expert node. When experts and their router share a
    // split (e.g. pure-CPU -ngl 0), that guarantee doesn't hold; set
    // LLAMA_SSD_STREAM_SERIAL=1 to fall back to node-at-a-time execution.
    static const bool serial = getenv("LLAMA_SSD_STREAM_SERIAL") != nullptr;
    if (t->op != GGML_OP_MUL_MAT_ID) {
        return serial;
    }
    ggml_tensor * w   = t->src[0];
    ggml_tensor * ids = t->src[2];
    if (w == nullptr || ids == nullptr || ids->type != GGML_TYPE_I32 || ids->data == nullptr) {
        return serial;
    }

    auto & st = g_state;
    std::lock_guard<std::mutex> lock(st.mutex);
    auto reg = st.registry.find(w);
    if (reg == st.registry.end()) {
        return serial; // not a streamed expert tensor
    }

    const int64_t n_expert = w->ne[2];
    const size_t  slice    = w->nb[2];
    char *        base     = (char *) w->data;
    const int     fd       = reg->second.fd;
    const size_t  offs     = reg->second.offs;

    const int64_t   n_ids   = ggml_nelements(ids);
    const int32_t * id_data = (const int32_t *) ids->data;
    for (int64_t k = 0; k < n_ids; k++) {
        const int32_t e = id_data[k];
        if (e < 0 || e >= n_expert) {
            continue;
        }
        ssd_touch(w, (int) e, base + (size_t) e * slice, slice, fd, offs + (size_t) e * slice);
    }
    if (st.debug && (st.n_hit + st.n_miss) % 4000 < (uint64_t) n_ids) {
        const uint64_t total = st.n_hit + st.n_miss;
        GGML_LOG_INFO("ggml_ssd_stream: hits=%llu miss=%llu (hit %.1f%%) promote=%llu evict=%llu resident=%zuMiB (prot %zuMiB) read=%.2fGB\n",
                (unsigned long long) st.n_hit, (unsigned long long) st.n_miss,
                total ? 100.0 * st.n_hit / total : 0.0,
                (unsigned long long) st.n_promote, (unsigned long long) st.n_evict,
                st.resident_bytes / (1024*1024), st.protected_bytes / (1024*1024),
                st.bytes_read / 1e9);
    }
    return true; // node-at-a-time: guarantees ids and all deps are computed before the fill
}
