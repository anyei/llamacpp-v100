#include "ggml-ssd-stream.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <list>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

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
    bool   gpu             = false;    // increment 3: GPU expert-slot cache (LLAMA_SSD_STREAM_GPU)
    size_t gpu_vram_budget = 0;        // VRAM budget for the slot cache, bytes (LLAMA_SSD_STREAM_VRAM_BUDGET MiB)
    size_t budget_bytes    = 0;
    size_t protected_cap   = 0;        // byte cap of the protected segment (slru only)
    long   page_size       = 4096;

    // aligned bounce buffer for O_DIRECT reads (reused; guarded by mutex during fill)
    void * bounce      = nullptr;
    size_t bounce_cap  = 0;

    // (3.3) parallel miss-path reads: LLAMA_SSD_STREAM_READ_THREADS workers, each
    // with its own aligned bounce buffer (the shared `bounce` above is not safe for
    // concurrent O_DIRECT reads). Default 1 => serial path, no behaviour change.
    int                 read_threads = 1;
    std::vector<void *>  rbufs;      // per-worker O_DIRECT bounce buffers
    std::vector<size_t>  rbuf_caps;

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
    uint64_t gpu_hit = 0, gpu_miss = 0; // VRAM slot-cache hits/misses (GPU landing)
    // (3.3 profiling, GGML_SSD_STREAM_DEBUG) miss-path time split: wall-clock ns
    // blocked in ssd_touch (RAM lookup / SSD pread) vs issuing the H2D copy.
    uint64_t gpu_read_ns = 0, gpu_h2d_ns = 0;
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

// O_DIRECT read of [off, off+len) into dst via a caller-owned page-aligned bounce
// (grown as needed). Touches no shared state, so concurrent callers with distinct
// dst + bounce are safe. Returns true on success. (O_DIRECT reads an aligned
// superset into the bounce, then copies the exact range out - so streaming never
// populates the page cache, keeping RSS/cgroup charge bounded to the arena.)
static bool ssd_pread_odirect(int fd, void * dst, size_t len, size_t off,
                              void ** bounce, size_t * bounce_cap, long page_size) {
#ifdef GGML_SSD_STREAM_SUPPORTED
    const size_t A    = (size_t) page_size;
    const size_t aoff = off & ~(A - 1);
    const size_t aend = (off + len + A - 1) & ~(A - 1);
    const size_t asz  = aend - aoff;
    if (*bounce_cap < asz) {
        free(*bounce);
        *bounce = nullptr;
        if (posix_memalign(bounce, A, asz) != 0) {
            *bounce = nullptr; *bounce_cap = 0;
            GGML_LOG_ERROR("%s: posix_memalign(%zu) failed\n", __func__, asz);
            return false;
        }
        *bounce_cap = asz;
    }
    // O_DIRECT requires aligned offset/buffer; the kernel returns a short (but
    // still page-multiple) count only at EOF, which is fine as long as it covers
    // the requested tail.
    const size_t need = off + len - aoff;
    size_t done = 0;
    while (done < need) {
        ssize_t r = pread(fd, (char *) *bounce + done, asz - done, (off_t) (aoff + done));
        if (r < 0) {
            GGML_LOG_ERROR("%s: O_DIRECT pread failed (fd=%d aoff=%zu asz=%zu): %s\n",
                    __func__, fd, aoff, asz - done, strerror(errno));
            return false;
        }
        if (r == 0) {
            break; // EOF
        }
        done += (size_t) r;
    }
    memcpy(dst, (const char *) *bounce + (off - aoff), len);
    return true;
#else
    GGML_UNUSED(fd); GGML_UNUSED(dst); GGML_UNUSED(len); GGML_UNUSED(off);
    GGML_UNUSED(bounce); GGML_UNUSED(bounce_cap); GGML_UNUSED(page_size);
    return false;
#endif
}

// read [off, off+len) from fd into dst (serial path; uses the shared bounce).
// Caller holds the mutex.
void ssd_pread_full(int fd, void * dst, size_t len, size_t off) {
#ifdef GGML_SSD_STREAM_SUPPORTED
    auto & st = g_state;
    if (!st.odirect) {
        if (ssd_pread_buffered(fd, (char *) dst, len, off)) {
            st.bytes_read += len;
        }
        return;
    }
    if (ssd_pread_odirect(fd, dst, len, off, &st.bounce, &st.bounce_cap, st.page_size)) {
        st.bytes_read += len;
    }
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
// is resident; caller holds g_state.mutex. With defer_read=true the SLRU
// bookkeeping/eviction/insertion happen now (serial) but the pread is SKIPPED and
// the function returns true - the caller must then read [file_off,+len) into addr
// itself (used to batch+parallelize the miss reads). Returns false when no read is
// needed (RAM hit, or defer_read=false where the read was done inline).
bool ssd_touch(const ggml_tensor * t, int e, void * addr, size_t len, int fd, size_t file_off,
        bool defer_read = false) {
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
        return false; // resident: no read needed
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
    if (!defer_read) {
        ssd_pread_full(fd, addr, len, file_off);
    }
    st.probation.push_front(slice_node{ key, addr, len });
    st.index[key] = slice_ref{ false, st.probation.begin() };
    st.resident_bytes += len;
    st.n_miss++;
    return defer_read; // caller reads the bytes when deferred
}

// ---- GPU expert-slot cache policy (increment 3.2a) -------------------------
//
// A persistent pool of K VRAM slots holding hot expert slices across decode
// passes: a cache hit is a zero-copy GPU read (the antipattern to avoid is
// re-copying the used experts H2D every token - measured 2.5 -> 1.2 t/s on
// DeepSeek). Keyed by (expert-weight tensor, expert index); one pool per expert
// tensor "class" (e.g. gate/up/down share a pool across layers - same slice
// shape). This 3.2a step is the host-side policy only (slot assignment + LRU
// eviction); the VRAM buffer alloc, H2D fill, and id->slot remap in the
// scheduler's expert-copy block land in 3.2b. Plain LRU here; SLRU + 2nd-miss
// admission (issue #20757: +8-15pp hit) is the 3.2e refinement.

struct gpu_slot_pool {
    // Segmented LRU (SLRU) over K slots for scan resistance: a slice seen once sits
    // in probation; a second hit promotes it to protected (bounded by protected_cap,
    // overflow demotes back). Eviction takes the probation LRU tail first, so a burst
    // of one-time experts never displaces a hot (protected) expert. This raises the
    // steady-state hit rate on skewed MoE routing (issue #20757: +8-15pp vs LRU).
    // slru=false degenerates to a single global LRU (everything stays in probation).
    struct ref { bool prot; int slot; std::list<slice_key>::iterator it; };

    int  k             = 0;
    int  protected_cap = 0;
    bool slru          = true;
    std::list<slice_key>       probation;   // MRU front
    std::list<slice_key>       protectd;     // MRU front
    std::map<slice_key, ref>   idx;
    std::vector<int>           freelist;
    uint64_t n_hit = 0, n_miss = 0, n_evict = 0, n_promote = 0;

    void init(int k_, bool slru_, int protected_pct) {
        k    = k_ > 0 ? k_ : 0;
        slru = slru_;
        protected_cap = slru ? (int) ((double) k * protected_pct / 100.0) : 0;
        probation.clear();
        protectd.clear();
        idx.clear();
        freelist.clear();
        freelist.reserve(k);
        for (int i = k - 1; i >= 0; --i) {
            freelist.push_back(i);
        }
        n_hit = n_miss = n_evict = n_promote = 0;
    }

    void demote() {
        while ((int) protectd.size() > protected_cap && !protectd.empty()) {
            auto tail = std::prev(protectd.end());
            ref & r = idx[*tail];
            probation.splice(probation.begin(), protectd, tail);
            r.prot = false;
            r.it   = probation.begin();
        }
    }

    // Return the slot holding (key); on a miss assign a slot (free, else evict the
    // probation LRU tail, else the protected tail) and set *miss=true. -1 if k==0.
    int touch(const slice_key & key, bool * miss) {
        auto it = idx.find(key);
        if (it != idx.end()) {
            ref & r = it->second;
            if (r.prot) {
                protectd.splice(protectd.begin(), protectd, r.it);
            } else if (slru) {
                protectd.splice(protectd.begin(), probation, r.it); // promote
                r.prot = true;
                r.it   = protectd.begin();
                n_promote++;
                demote();
            } else {
                probation.splice(probation.begin(), probation, r.it);
            }
            *miss = false;
            n_hit++;
            return r.slot;
        }
        if (k == 0) {
            *miss = true;
            return -1;
        }
        int slot;
        if (!freelist.empty()) {
            slot = freelist.back();
            freelist.pop_back();
        } else {
            std::list<slice_key> & seg = !probation.empty() ? probation : protectd;
            const slice_key victim = seg.back();
            slot = idx[victim].slot;
            idx.erase(victim);
            seg.pop_back();
            n_evict++;
        }
        probation.push_front(key);
        idx[key] = ref{ false, slot, probation.begin() };
        *miss = true;
        n_miss++;
        return slot;
    }
};

// One VRAM slot pool + its backing tensor, shared by all streamed expert tensors
// of the same slice shape (gate/up share; down is a different shape -> own pool).
// Slots are keyed by (source tensor, expert) so distinct tensors never alias.
struct gpu_pool_entry {
    ggml_context *        ctx   = nullptr;
    ggml_backend_buffer_t buf   = nullptr;
    ggml_tensor *         slots = nullptr; // [ne0, ne1, K] on the GPU backend
    gpu_slot_pool         pool;
    int                   k     = 0;
    bool                  logged = false;
};

// keyed by (ne0, ne1, type) of one expert slice
using gpu_pool_key = std::tuple<int64_t, int64_t, int>;
std::map<gpu_pool_key, gpu_pool_entry> g_gpu_pools;

// per-MUL_MAT_ID-node state: the captured router-selection tensor (so we keep
// reading fresh ids after swapping src[2]) and the remapped slot-ids GPU scratch.
struct gpu_node_state {
    const ggml_tensor *   orig_ids = nullptr;
    ggml_context *        ids_ctx  = nullptr;
    ggml_backend_buffer_t ids_buf  = nullptr;
    ggml_tensor *         slot_ids = nullptr; // GPU I32, same shape as orig_ids
    int64_t               cap      = 0;
};
std::map<const ggml_tensor *, gpu_node_state> g_gpu_nodes;

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
        // increment 3: GPU expert-slot cache (host policy only in 3.2a)
        st.gpu = getenv("LLAMA_SSD_STREAM_GPU") != nullptr && atoi(getenv("LLAMA_SSD_STREAM_GPU")) != 0;
        const char * vb = getenv("LLAMA_SSD_STREAM_VRAM_BUDGET");
        st.gpu_vram_budget = (vb ? (size_t) atoll(vb) : 4096ull) * 1024ull * 1024ull;
        st.page_size = sysconf(_SC_PAGESIZE);
        // (3.3) parallel miss-path reads; clamp to a sane range.
        const char * rt = getenv("LLAMA_SSD_STREAM_READ_THREADS");
        st.read_threads = rt ? atoi(rt) : 1;
        if (st.read_threads < 1)  st.read_threads = 1;
        if (st.read_threads > 16) st.read_threads = 16;
        st.enabled_checked = true;
        if (st.enabled) {
            st.debug = getenv("GGML_SSD_STREAM_DEBUG") != nullptr;
            GGML_LOG_INFO("ggml_ssd_stream: enabled, expert-cache budget %zu MiB, policy %s%s, read-threads %d\n",
                    st.budget_bytes / (1024 * 1024),
                    st.slru ? "SLRU" : "LRU",
                    st.gpu ? " [GPU slot cache ON]" : "",
                    st.read_threads);
        }
    }
    return st.enabled;
#else
    return false;
#endif
}

bool ggml_ssd_stream_gpu_enabled(void) {
    return ggml_ssd_stream_enabled() && g_state.gpu;
}

void ggml_ssd_stream_gpu_ensure_pool(const ggml_tensor * w, ggml_backend_t backend) {
    if (!ggml_ssd_stream_gpu_enabled() || w == nullptr || backend == nullptr) {
        return;
    }
    if (!ggml_ssd_stream_is_streamed(w)) {
        return;
    }
    auto & st = g_state;
    std::lock_guard<std::mutex> lock(st.mutex);
    const gpu_pool_key key{ w->ne[0], w->ne[1], (int) w->type };
    gpu_pool_entry & e = g_gpu_pools[key];
    if (e.slots != nullptr) {
        return; // already allocated for this slice shape
    }
    // The VRAM budget is TOTAL across all pools; give each pool an equal share.
    // The number of pools = distinct expert-slice classes (ne0,ne1,type), auto-
    // detected from the registered streamed tensors (all registered at load, before
    // this runs at first compute). LLAMA_SSD_STREAM_VRAM_POOLS overrides the count.
    int n_pools;
    const char * ph = getenv("LLAMA_SSD_STREAM_VRAM_POOLS");
    if (ph && atoi(ph) > 0) {
        n_pools = atoi(ph);
    } else {
        std::set<gpu_pool_key> classes;
        for (const auto & kv : st.registry) {
            const ggml_tensor * t = kv.first;
            classes.insert(gpu_pool_key{ t->ne[0], t->ne[1], (int) t->type });
        }
        n_pools = classes.empty() ? 1 : (int) classes.size();
    }
    const size_t slice          = w->nb[2];
    const size_t per_pool_bytes = st.gpu_vram_budget / (size_t) n_pools;
    int k = slice ? (int) (per_pool_bytes / slice) : 0;
    if (k < 1) {
        k = 1;
    }
    struct ggml_init_params ip = { 2 * ggml_tensor_overhead(), nullptr, /*no_alloc=*/ true };
    e.ctx = ggml_init(ip);
    if (e.ctx == nullptr) {
        GGML_LOG_ERROR("ggml_ssd_stream: GPU pool ggml_init failed\n");
        return;
    }
    // +1 pad slot: the CUDA MMQ path may over-read ~512 B past the last expert
    // (the reason copy_experts pads); the extra slot keeps that read in-bounds.
    e.slots = ggml_new_tensor_3d(e.ctx, w->type, w->ne[0], w->ne[1], k + 1);
    e.buf   = e.slots ? ggml_backend_alloc_ctx_tensors(e.ctx, backend) : nullptr;
    if (e.buf == nullptr || e.slots == nullptr) {
        GGML_LOG_ERROR("ggml_ssd_stream: GPU slot pool alloc failed (K=%d, %.2f GB)\n",
                k, (double) k * slice / 1e9);
        return;
    }
    // GPU slot pool defaults to SLRU (scan resistance for skewed MoE routing);
    // LLAMA_SSD_STREAM_GPU_SLRU=0 forces plain LRU, LLAMA_SSD_STREAM_GPU_PROTECTED_PCT
    // sizes the protected segment (default 80).
    static const bool gpu_slru = []{ const char * e = getenv("LLAMA_SSD_STREAM_GPU_SLRU"); return e == nullptr || atoi(e) != 0; }();
    static const int  gpu_pct  = []{ const char * e = getenv("LLAMA_SSD_STREAM_GPU_PROTECTED_PCT"); int v = e ? atoi(e) : 80; return v < 0 ? 0 : (v > 100 ? 100 : v); }();
    e.k = k;
    e.pool.init(k, gpu_slru, gpu_pct);
    e.logged = true;
    GGML_LOG_INFO("ggml_ssd_stream: GPU slot pool [%lldx%lld %s] K=%d slots (%.2f GB VRAM, 1/%d classes)\n",
            (long long) w->ne[0], (long long) w->ne[1], ggml_type_name(w->type), k,
            (double) k * slice / 1e9, n_pools);
}

bool ggml_ssd_stream_gpu_shrink_copy(ggml_tensor * copy, const ggml_tensor * src, ggml_backend_t backend) {
#ifdef GGML_SSD_STREAM_SUPPORTED
    if (!ggml_ssd_stream_gpu_enabled() || copy == nullptr || src == nullptr || backend == nullptr) {
        return false;
    }
    if (!ggml_ssd_stream_is_streamed(src) || src->ne[2] <= 1) {
        return false; // only the 3D expert weight (ne[2] = n_expert) is worth shrinking
    }
    static const bool no_reclaim = []{
        const char * e = getenv("LLAMA_SSD_STREAM_GPU_NO_RECLAIM");
        return e != nullptr && atoi(e) != 0;
    }();
    if (no_reclaim) {
        return false;
    }
    GGML_UNUSED(backend);
    // NOTE: do NOT allocate the pool here. This runs during the scheduler's
    // measure/reserve pass, which happens BEFORE the model weights finish loading -
    // so the streamed-tensor registry is still empty and pool auto-sizing (by class
    // count) would see 0 classes and give each pool the whole budget. The pool is
    // allocated later on the compute path (ggml_ssd_stream_gpu_ensure_pool, called
    // right before gpu_bind) when the registry is complete. The shrink is safe
    // without a pool now: gpu_bind is guaranteed to redirect copy->data to the pool
    // at compute (same streamed-expert MUL_MAT_ID condition drives both), and the
    // compute-path fallback aborts loudly if bind ever fails on a shrunk copy.
    if (copy->ne[2] <= 1) {
        return true; // already minimal
    }
    // Shrink to a single expert slice: at compute gpu_bind sets ne[2]=pool.k and
    // data=slots, so this buffer is never read - it just needs to be tiny for the
    // arena. Keep the tensor contiguous (recompute the row stride nb[3]).
    copy->ne[2] = 1;
    copy->nb[3] = copy->nb[2];
    return true;
#else
    GGML_UNUSED(copy); GGML_UNUSED(src); GGML_UNUSED(backend);
    return false;
#endif
}

const ggml_tensor * ggml_ssd_stream_gpu_orig_ids(const ggml_tensor * node, const ggml_tensor * cur_ids) {
    std::lock_guard<std::mutex> lock(g_state.mutex);
    gpu_node_state & ns = g_gpu_nodes[node];
    // cur_ids (node->src[2]) is either the router selection (first sight, or a fresh
    // graph node) or our slot_ids (swapped on a previous token for THIS node). If it
    // is neither the captured orig_ids nor our slot_ids, the node pointer was recycled
    // by a graph rebuild onto a stale entry whose orig_ids now dangles - reset and
    // re-capture, dropping the stale scratch. (Without this, a recycled node reads a
    // freed ids tensor -> garbage expert ids -> the MUL_MAT_ID id-range assert.)
    if (ns.orig_ids == nullptr || (cur_ids != ns.orig_ids && cur_ids != ns.slot_ids)) {
        if (ns.ids_buf) { ggml_backend_buffer_free(ns.ids_buf); ns.ids_buf = nullptr; }
        if (ns.ids_ctx) { ggml_free(ns.ids_ctx); ns.ids_ctx = nullptr; }
        ns.slot_ids = nullptr;
        ns.cap      = 0;
        ns.orig_ids = cur_ids;
    }
    return ns.orig_ids;
}

bool ggml_ssd_stream_gpu_bind(ggml_tensor * node, const ggml_tensor * input, ggml_tensor * input_cpy,
        ggml_backend_t backend, const int32_t * ids, int64_t n_ids, int64_t n_expert) {
#ifdef GGML_SSD_STREAM_SUPPORTED
    if (!ggml_ssd_stream_gpu_enabled() || node == nullptr || input == nullptr ||
        input_cpy == nullptr || backend == nullptr || ids == nullptr || n_ids <= 0 || n_expert <= 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_state.mutex);
    const gpu_pool_key key{ input->ne[0], input->ne[1], (int) input->type };
    auto pit = g_gpu_pools.find(key);
    if (pit == g_gpu_pools.end() || pit->second.slots == nullptr || pit->second.k <= 0) {
        return false; // pool not allocated for this slice class
    }
    gpu_pool_entry & pe    = pit->second;
    const size_t     slice = input->nb[2];

    gpu_node_state & ns = g_gpu_nodes[node];
    if (ns.orig_ids == nullptr) {
        return false; // orig_ids must be captured (gpu_orig_ids) before binding
    }
    // logical id shape + strides come from orig_ids (the router selection); the
    // `ids` buffer is a raw host copy of it, so we index with the same strides.
    const ggml_tensor * oi = ns.orig_ids;
    const int64_t ne0 = oi->ne[0], ne1 = oi->ne[1];
    const int64_t s0  = (int64_t) (oi->nb[0] / sizeof(int32_t));
    const int64_t s1  = (int64_t) (oi->nb[1] / sizeof(int32_t));
    const int64_t n_logical = ne0 * ne1;
    if (n_logical <= 0 || (s1 * (ne1 - 1) + s0 * (ne0 - 1)) >= n_ids) {
        return false; // ids buffer smaller than the strided extent - bail safely
    }

    // touch each used expert once; VRAM misses are filled H2D from the RAM arena
    // (input->data) into their slot. Build expert -> slot. Three phases so the SSD
    // miss reads can be parallelized (LLAMA_SSD_STREAM_READ_THREADS): (1) touch the
    // pool + reserve the RAM slot serially (SLRU state is not thread-safe), (2) pread
    // the RAM-missed slices - parallel when read_threads>1, (3) H2D every VRAM miss.
    const bool prof = g_state.debug;
    const bool par  = g_state.read_threads > 1;
    auto reg = g_state.registry.find(input);
    const bool have_reg = reg != g_state.registry.end();
    struct read_task { int fd; char * dst; size_t len; size_t off; };
    std::vector<std::pair<int32_t,int>> misses; // (expert, slot) needing an H2D
    std::vector<read_task>              reads;  // VRAM misses that also missed RAM

    std::vector<int32_t> expert_to_slot((size_t) n_expert, -1);
    auto tp0 = std::chrono::steady_clock::now();
    for (int64_t i1 = 0; i1 < ne1; i1++) {
        for (int64_t i0 = 0; i0 < ne0; i0++) {
            const int32_t e = ids[i1 * s1 + i0 * s0];
            if (e < 0 || e >= n_expert || expert_to_slot[e] != -1) {
                continue;
            }
            bool miss = false;
            const int slot = pe.pool.touch(slice_key{ input, (int) e }, &miss);
            if (slot < 0) {
                return false;
            }
            if (miss) { g_state.gpu_miss++; } else { g_state.gpu_hit++; }
            expert_to_slot[e] = slot;
            if (miss) {
                misses.emplace_back(e, slot);
                if (have_reg) {
                    // RAM tier: bookkeeping now (serial); on a RAM miss, read the
                    // slice - inline when serial, or deferred to the parallel phase.
                    char * dst = (char *) input->data + (size_t) e * slice;
                    const size_t foff = reg->second.offs + (size_t) e * slice;
                    if (ssd_touch(input, (int) e, dst, slice, reg->second.fd, foff, /*defer_read=*/par)) {
                        reads.push_back(read_task{ reg->second.fd, dst, slice, foff });
                    }
                }
            }
        }
    }
    auto tp1 = std::chrono::steady_clock::now();

    // (2) read the RAM-missed slices. Parallel across read_threads workers, each with
    // its own O_DIRECT bounce buffer (distinct dst + bounce -> no shared state).
    if (par && !reads.empty()) {
        const int nt = std::min<int>(g_state.read_threads, (int) reads.size());
        if ((int) g_state.rbufs.size() < nt) {
            g_state.rbufs.resize(nt, nullptr);
            g_state.rbuf_caps.resize(nt, 0);
        }
        auto worker = [&](int wi) {
            for (size_t j = (size_t) wi; j < reads.size(); j += (size_t) nt) {
                const read_task & t = reads[j];
                if (g_state.odirect) {
                    ssd_pread_odirect(t.fd, t.dst, t.len, t.off,
                            &g_state.rbufs[wi], &g_state.rbuf_caps[wi], g_state.page_size);
                } else {
                    ssd_pread_buffered(t.fd, t.dst, t.len, t.off);
                }
            }
        };
        std::vector<std::thread> pool;
        pool.reserve((size_t) (nt - 1));
        for (int wi = 1; wi < nt; ++wi) pool.emplace_back(worker, wi);
        worker(0);
        for (auto & th : pool) th.join();
        for (const read_task & t : reads) g_state.bytes_read += t.len;
    }
    auto tp2 = std::chrono::steady_clock::now();

    // (3) H2D every VRAM miss (arena -> slot); the arena slice is now resident.
    for (const auto & m : misses) {
        ggml_backend_tensor_set_async(backend, pe.slots,
                (const char *) input->data + (size_t) m.first * slice,
                (size_t) m.second * slice, slice);
    }
    if (prof) {
        auto tp3 = std::chrono::steady_clock::now();
        // phase-1 ssd_touch inline reads (serial path) + phase-2 parallel reads.
        g_state.gpu_read_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(tp1 - tp0).count()
                             + std::chrono::duration_cast<std::chrono::nanoseconds>(tp2 - tp1).count();
        g_state.gpu_h2d_ns  += std::chrono::duration_cast<std::chrono::nanoseconds>(tp3 - tp2).count();
    }

    // remap into a CONTIGUOUS GPU scratch shaped like orig_ids (values in [0,K)).
    if (ns.slot_ids == nullptr || ns.cap < n_logical) {
        if (ns.ids_buf) { ggml_backend_buffer_free(ns.ids_buf); ns.ids_buf = nullptr; }
        if (ns.ids_ctx) { ggml_free(ns.ids_ctx); ns.ids_ctx = nullptr; }
        struct ggml_init_params ip = { 2 * ggml_tensor_overhead(), nullptr, /*no_alloc=*/ true };
        ns.ids_ctx  = ggml_init(ip);
        ns.slot_ids = ns.ids_ctx ? ggml_new_tensor_2d(ns.ids_ctx, GGML_TYPE_I32, ne0, ne1) : nullptr;
        ns.ids_buf  = ns.slot_ids ? ggml_backend_alloc_ctx_tensors(ns.ids_ctx, backend) : nullptr;
        if (ns.ids_buf == nullptr) {
            ns.slot_ids = nullptr;
            return false;
        }
        ns.cap = n_logical;
    }
    static thread_local std::vector<int32_t> slot_ids_host;
    slot_ids_host.resize((size_t) n_logical);
    for (int64_t i1 = 0; i1 < ne1; i1++) {
        for (int64_t i0 = 0; i0 < ne0; i0++) {
            const int32_t e = ids[i1 * s1 + i0 * s0];
            slot_ids_host[i1 * ne0 + i0] = (e >= 0 && e < n_expert && expert_to_slot[e] >= 0) ? expert_to_slot[e] : 0;
        }
    }
    ggml_backend_tensor_set_async(backend, ns.slot_ids, slot_ids_host.data(), 0, (size_t) n_logical * sizeof(int32_t));

    // alias input_cpy to the persistent slot buffer and swap the ids -> slot ids.
    input_cpy->data  = pe.slots->data;
    input_cpy->ne[2] = pe.k;
    input_cpy->nb[2] = slice;
    node->src[2]     = ns.slot_ids;

    if (g_state.debug) {
        const uint64_t tot = g_state.gpu_hit + g_state.gpu_miss;
        if (tot % 8000 < (uint64_t) n_logical) {
            const uint64_t ram_tot = g_state.n_hit + g_state.n_miss;
            GGML_LOG_INFO("ggml_ssd_stream: GPU cache hit=%llu miss=%llu (%.1f%%) [pool %lldx%lld %s K=%d]\n",
                    (unsigned long long) g_state.gpu_hit, (unsigned long long) g_state.gpu_miss,
                    tot ? 100.0 * g_state.gpu_hit / tot : 0.0,
                    (long long) input->ne[0], (long long) input->ne[1], ggml_type_name(input->type), pe.k);
            // (3.3 profiling) On a VRAM miss, ssd_touch either finds the slice in the
            // RAM arena (fast) or preads it from SSD (slow); then we H2D it. This shows
            // the RAM-tier hit rate and the cumulative time split read(RAM/SSD) vs H2D
            // issue - deciding whether prefetch should target SSD reads or the H2D.
            GGML_LOG_INFO("ggml_ssd_stream: miss-path RAM hit=%llu miss=%llu (%.1f%%) | read %.2fs H2D-issue %.3fs\n",
                    (unsigned long long) g_state.n_hit, (unsigned long long) g_state.n_miss,
                    ram_tot ? 100.0 * g_state.n_hit / ram_tot : 0.0,
                    g_state.gpu_read_ns / 1e9, g_state.gpu_h2d_ns / 1e9);
        }
    }
    return true;
#else
    GGML_UNUSED(node); GGML_UNUSED(input); GGML_UNUSED(input_cpy); GGML_UNUSED(backend);
    GGML_UNUSED(ids); GGML_UNUSED(n_ids); GGML_UNUSED(n_expert);
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
