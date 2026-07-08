#include "ggml-ssd-stream.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <map>
#include <mutex>
#include <unordered_map>
#include <utility>

// SSD expert-streaming buffer type — see ggml-ssd-stream.h / docs/ssd-streaming-plan.md.
//
// Increment 1 (CPU-landing): experts live in a sparse anonymous arena at their
// natural offsets; a per-expert slice is pread() from the model file into the
// arena just before the MUL_MAT_ID that consumes it, cached under an LRU byte
// budget, and MADV_DONTNEED-evicted when the budget is exceeded. Correct and
// RSS-bounded; O_DIRECT / prefetch / GPU-landing are later increments.

#if defined(__linux__)
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

struct tensor_backing {
    int    fd;   // dup'd, kept open for process lifetime
    size_t offs; // byte offset of the tensor's data in the file
};

struct ssd_stream_state {
    std::mutex mutex;

    bool   enabled_checked = false;
    bool   enabled         = false;
    bool   debug           = false;
    size_t budget_bytes    = 0;
    long   page_size       = 4096;

    std::unordered_map<const ggml_tensor *, tensor_backing> registry;
    std::unordered_map<int, int>                            fd_dup;      // orig fd -> dup fd

    // global LRU of resident expert slices, bounded by budget_bytes
    std::list<slice_node>                                   lru;
    std::map<slice_key, std::list<slice_node>::iterator>    index;
    size_t                                                  resident_bytes = 0;

    // stats
    uint64_t n_hit = 0, n_miss = 0, n_evict = 0, bytes_read = 0;
};

ssd_stream_state g_state;

void ssd_pread_full(int fd, void * dst, size_t len, size_t off) {
#ifdef GGML_SSD_STREAM_SUPPORTED
    char * p = (char *) dst;
    size_t done = 0;
    while (done < len) {
        ssize_t r = pread(fd, p + done, len - done, (off_t) (off + done));
        if (r < 0) {
            GGML_LOG_ERROR("%s: pread failed (fd=%d off=%zu len=%zu): %s\n",
                    __func__, fd, off + done, len - done, strerror(errno));
            return;
        }
        if (r == 0) {
            break; // EOF (should not happen for valid offsets)
        }
        done += (size_t) r;
    }
    g_state.bytes_read += done;
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

// ensure expert slice (t,e) at [addr,addr+len) backed by file bytes at file_off
// is resident; caller holds g_state.mutex.
void ssd_touch(const ggml_tensor * t, int e, void * addr, size_t len, int fd, size_t file_off) {
    slice_key key{ t, e };
    auto it = g_state.index.find(key);
    if (it != g_state.index.end()) {
        g_state.lru.splice(g_state.lru.begin(), g_state.lru, it->second); // -> MRU
        g_state.n_hit++;
        return;
    }
    // miss: evict LRU until the new slice fits
    while (g_state.resident_bytes + len > g_state.budget_bytes && !g_state.lru.empty()) {
        auto & back = g_state.lru.back();
        ssd_evict(back);
        g_state.resident_bytes -= back.len;
        g_state.index.erase(back.key);
        g_state.lru.pop_back();
    }
    ssd_pread_full(fd, addr, len, file_off);
    g_state.lru.push_front(slice_node{ key, addr, len });
    g_state.index[key] = g_state.lru.begin();
    g_state.resident_bytes += len;
    g_state.n_miss++;
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
        st.page_size = sysconf(_SC_PAGESIZE);
        st.enabled_checked = true;
        if (st.enabled) {
            st.debug = getenv("GGML_SSD_STREAM_DEBUG") != nullptr;
            GGML_LOG_INFO("ggml_ssd_stream: enabled, expert-cache budget %zu MiB\n",
                    st.budget_bytes / (1024 * 1024));
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
        st.fd_dup[fd] = dfd;
    } else {
        dfd = f->second;
    }
    st.registry[t] = tensor_backing{ dfd, file_offset };
#else
    GGML_UNUSED(t); GGML_UNUSED(fd); GGML_UNUSED(file_offset);
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
        GGML_LOG_INFO("ggml_ssd_stream: hits=%llu miss=%llu evict=%llu resident=%zuMiB read=%.2fGB\n",
                (unsigned long long) st.n_hit, (unsigned long long) st.n_miss,
                (unsigned long long) st.n_evict, st.resident_bytes / (1024*1024), st.bytes_read / 1e9);
    }
    return true; // node-at-a-time: guarantees ids and all deps are computed before the fill
}
