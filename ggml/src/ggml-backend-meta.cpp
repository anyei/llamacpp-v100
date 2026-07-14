#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

struct ggml_backend_meta_device;
struct ggml_backend_meta_buffer_type;
struct ggml_backend_meta_buffer;
struct ggml_backend_meta;

const char * ggml_backend_meta_split_axis_name(enum ggml_backend_meta_split_axis split_axis) {
    switch (split_axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
            return "0";
        case GGML_BACKEND_SPLIT_AXIS_1:
            return "1";
        case GGML_BACKEND_SPLIT_AXIS_2:
            return "2";
        case GGML_BACKEND_SPLIT_AXIS_3:
            return "3";
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            return "MIRRORED";
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL:
            return "PARTIAL";
        case GGML_BACKEND_SPLIT_AXIS_NONE:
            return "NONE";
        case GGML_BACKEND_SPLIT_AXIS_UNKNOWN:
            return "UNKNOWN";
        default:
            GGML_ABORT("fatal error");
    }
}

//
// meta backend device
//

struct ggml_backend_meta_device_context {
    std::vector<ggml_backend_dev_t>     simple_devs;
    ggml_backend_meta_get_split_state_t get_split_state;
    void *                              get_split_state_ud;

    std::string name;
    std::string description;

    ggml_backend_meta_device_context(
            std::vector<ggml_backend_dev_t> simple_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) :
            simple_devs(std::move(simple_devs)), get_split_state(get_split_state), get_split_state_ud(get_split_state_ud) {
        // the [N] device count lets an RPC client detect a remote TP island and its size
        name        = std::string("Meta(");
        description = std::string("Meta[") + std::to_string(this->simple_devs.size()) + "](";
        for (size_t i = 0; i < this->simple_devs.size(); i++) {
            if (i > 0) {
                name        += ",";
                description += ",";
            }
            name        += ggml_backend_dev_name       (this->simple_devs[i]);
            description += ggml_backend_dev_description(this->simple_devs[i]);
        }
        name        += ")";
        description += ")";
    }

    bool operator<(const ggml_backend_meta_device_context & other) const {
        return std::tie(simple_devs, get_split_state, get_split_state_ud)
            < std::tie(other.simple_devs, other.get_split_state, other.get_split_state_ud);
    }
};

static bool ggml_backend_dev_is_meta(ggml_backend_dev_t dev);

static const char * ggml_backend_meta_device_get_name(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->name.c_str();
}

static const char * ggml_backend_meta_device_get_description(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->description.c_str();
}

static void ggml_backend_meta_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    *free  = 0;
    *total = 0;
    for (ggml_backend_dev_t dev : meta_dev_ctx->simple_devs) {
        size_t tmp_free, tmp_total;
        ggml_backend_dev_memory(dev, &tmp_free, &tmp_total);
        *free  += tmp_free;
        *total += tmp_total;
    }
}

static enum ggml_backend_dev_type ggml_backend_meta_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_META;

    GGML_UNUSED(dev);
}

static void ggml_backend_meta_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    // TODO replace placeholders
    props->name        = ggml_backend_meta_device_get_name(dev);
    props->description = ggml_backend_meta_device_get_description(dev);
    props->type        = ggml_backend_meta_device_get_type(dev);
    props->device_id   = 0;

    ggml_backend_meta_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false, // Not implemented.
        /* .buffer_from_host_ptr  = */ false, // Not implemented.
        /* .events                = */ false, // Not implemented.
    };
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_dev_props tmp_props;
        ggml_backend_dev_get_props(simple_dev, &tmp_props);
        props->caps.async                = props->caps.async                && tmp_props.caps.async;
        props->caps.host_buffer          = props->caps.host_buffer          && tmp_props.caps.host_buffer;
        props->caps.buffer_from_host_ptr = props->caps.buffer_from_host_ptr && tmp_props.caps.buffer_from_host_ptr;
        props->caps.events               = props->caps.events               && tmp_props.caps.events;
    }
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params);

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev);

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev);

static bool ggml_backend_meta_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return std::all_of(meta_dev_ctx->simple_devs.begin(), meta_dev_ctx->simple_devs.end(),
        [op](ggml_backend_dev_t simple_dev) { return ggml_backend_dev_supports_op(simple_dev, op); });
}

static bool ggml_backend_meta_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ggml_backend_dev_t dev_buft = ggml_backend_buft_get_device(buft);
    if (!ggml_backend_dev_is_meta(dev_buft)) {
        return false;
    }
    const ggml_backend_meta_device_context * meta_dev_ctx      = (const ggml_backend_meta_device_context *) dev->context;
    const ggml_backend_meta_device_context * meta_buft_dev_ctx = (const ggml_backend_meta_device_context *) dev_buft->context;
    if (meta_dev_ctx->simple_devs.size() != meta_buft_dev_ctx->simple_devs.size()) {
        return false;
    }
    for (size_t i = 0; i < meta_dev_ctx->simple_devs.size(); i++) {
        if (meta_dev_ctx->simple_devs[i] != meta_buft_dev_ctx->simple_devs[i]) {
            return false;
        }
    }
    return true;
}

static const ggml_backend_device_i ggml_backend_meta_device_iface = {
    /* .get_name             = */ ggml_backend_meta_device_get_name,
    /* .get_description      = */ ggml_backend_meta_device_get_description,
    /* .get_memory           = */ ggml_backend_meta_device_get_memory,
    /* .get_type             = */ ggml_backend_meta_device_get_type,
    /* .get_props            = */ ggml_backend_meta_device_get_props,
    /* .init_backend         = */ ggml_backend_meta_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_meta_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_meta_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_meta_device_supports_op,
    /* .supports_buft        = */ ggml_backend_meta_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static bool ggml_backend_dev_is_meta(ggml_backend_dev_t dev) {
    return dev != nullptr && dev->iface.get_name == ggml_backend_meta_device_iface.get_name;
}

static size_t ggml_backend_meta_dev_n_devs(ggml_backend_dev_t meta_dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    return meta_dev_ctx->simple_devs.size();
}

static ggml_backend_dev_t ggml_backend_meta_dev_simple_dev(ggml_backend_dev_t meta_dev, size_t index) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    GGML_ASSERT(index < meta_dev_ctx->simple_devs.size());
    return meta_dev_ctx->simple_devs[index];
}

ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) {
    GGML_ASSERT(n_devs <= GGML_BACKEND_META_MAX_DEVICES);
    // TODO: this is not thread-safe - needs to be fixed
    static std::vector<std::unique_ptr<ggml_backend_meta_device_context>>         ctxs;
    static std::map<ggml_backend_meta_device_context, struct ggml_backend_device> meta_devs;

    std::vector<ggml_backend_dev_t> simple_devs;
    simple_devs.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_devs.push_back(devs[i]);
    }
    ggml_backend_meta_device_context ctx(simple_devs, get_split_state, get_split_state_ud);

    {
        auto it = meta_devs.find(ctx);
        if (it != meta_devs.end()) {
            return &it->second;
        }
    }
    ctxs.push_back(std::make_unique<ggml_backend_meta_device_context>(ctx));

    struct ggml_backend_device meta_dev = {
        /*iface  =*/ ggml_backend_meta_device_iface,
        /*reg    =*/ nullptr,
        /*ctx    =*/ ctxs.back().get(),
    };

    auto result = meta_devs.emplace(*ctxs.back(), meta_dev);
    return &result.first->second;
}

//
// meta backend buffer type
//

struct ggml_backend_meta_buffer_type_context {
    std::vector<ggml_backend_buffer_type_t> simple_bufts;

    std::string name;

    ggml_backend_meta_buffer_type_context(std::vector<ggml_backend_buffer_type_t> simple_bufts) : simple_bufts(std::move(simple_bufts)) {
        name = "Meta(";
        for (size_t i = 0; i < this->simple_bufts.size(); i++) {
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_buft_name(this->simple_bufts[i]);
        }
        name += ")";
    }

    bool operator<(const ggml_backend_meta_buffer_type_context & other) const {
        return simple_bufts < other.simple_bufts;
    }
};

static size_t ggml_backend_meta_buft_n_bufts(ggml_backend_buffer_type_t meta_buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    return meta_buft_ctx->simple_bufts.size();
}

static const char * ggml_backend_meta_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) buft->context;
    return meta_buft_ctx->name.c_str();
}

static ggml_backend_buffer_type_t ggml_backend_meta_buft_simple_buft(ggml_backend_buffer_type_t meta_buft, size_t index) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    GGML_ASSERT(index < meta_buft_ctx->simple_bufts.size());
    return meta_buft_ctx->simple_bufts[index];
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);

static size_t ggml_backend_meta_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alignment = 1;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_meta_buft_simple_buft(buft, i));
        max_alignment = std::max(max_alignment, alignment);
        GGML_ASSERT(max_alignment % alignment == 0);
    }
    return max_alignment;
}

static size_t ggml_backend_meta_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_size = SIZE_MAX;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        max_size = std::min(max_size, ggml_backend_buft_get_max_size(ggml_backend_meta_buft_simple_buft(buft, i)));
    }
    return max_size;
}

static size_t ggml_backend_meta_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alloc_size = 0;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alloc_size = ggml_backend_buft_get_alloc_size(ggml_backend_meta_buft_simple_buft(buft, i), tensor);
        max_alloc_size = std::max(max_alloc_size, alloc_size);
    }

    // for statically resolvable tensors (weights, KV/recurrent caches) only the per-device
    // slice is stored, so reserve the largest slice rather than the full tensor - otherwise
    // every device's buffer spans the whole allocation space and split tensors waste ~half
    // the memory per device (offset-preserving placement keeps working: slice offsets are
    // assigned in this compacted space on every device alike)
    if (tensor->op == GGML_OP_NONE && tensor->view_src == nullptr && buft->device != nullptr) {
        const ggml_backend_meta_device_context * dev_ctx = (const ggml_backend_meta_device_context *) buft->device->context;
        const ggml_backend_meta_split_state state = dev_ctx->get_split_state(tensor, dev_ctx->get_split_state_ud);
        if (state.axis >= 0 && state.axis < GGML_MAX_DIMS && tensor->ne[state.axis] > 0) {
            int64_t ne_max = 0;
            for (size_t j = 0; j < n_simple_bufts; j++) {
                int64_t ne_j = 0;
                for (size_t s = 0; s < state.n_segments; s++) {
                    ne_j += state.ne[s*n_simple_bufts + j] * state.nr[s];
                }
                ne_max = std::max(ne_max, ne_j);
            }
            if (ne_max > 0 && ne_max < tensor->ne[state.axis]) {
                // multiply first: divide-first truncates for quantized block sizes
                max_alloc_size = (max_alloc_size * (size_t) ne_max + (size_t) tensor->ne[state.axis] - 1)
                    / (size_t) tensor->ne[state.axis];
            }
        }
    }
    return max_alloc_size;
}

static bool ggml_backend_meta_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        if (!ggml_backend_buft_is_host(ggml_backend_meta_buft_simple_buft(buft, i))) {
            return false;
        }
    }
    return true;
}

static const struct ggml_backend_buffer_type_i ggml_backend_meta_buffer_type_iface = {
    /* .get_name         = */ ggml_backend_meta_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_meta_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_meta_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_meta_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_meta_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_meta_buffer_type_is_host,
};

bool ggml_backend_buft_is_meta(ggml_backend_buffer_type_t buft) {
    return buft != nullptr && buft->iface.get_name == ggml_backend_meta_buffer_type_iface.get_name;
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev) {
    static std::map<ggml_backend_dev_t, struct ggml_backend_buffer_type> meta_bufts;
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    {
        auto it = meta_bufts.find(dev);
        if (it != meta_bufts.end()) {
            return &it->second;
        }
    }

    const size_t n_devs = ggml_backend_meta_dev_n_devs(dev);
    std::vector<ggml_backend_buffer_type_t> simple_bufts;
    simple_bufts.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_bufts.push_back(ggml_backend_dev_buffer_type(ggml_backend_meta_dev_simple_dev(dev, i)));
    }
    ggml_backend_meta_buffer_type_context * buft_ctx = new ggml_backend_meta_buffer_type_context(simple_bufts);

    struct ggml_backend_buffer_type meta_buft = {
        /*iface  =*/ ggml_backend_meta_buffer_type_iface,
        /*device =*/ dev,
        /*ctx    =*/ buft_ctx,
    };
    auto result = meta_bufts.emplace(dev, meta_buft);
    return &result.first->second;
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    ggml_backend_buffer_type_t host_buft = nullptr;
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_buffer_type_t simple_host_buft = ggml_backend_dev_host_buffer_type(simple_dev);
        if (simple_host_buft == nullptr) {
            return nullptr;
        }
        if (host_buft == nullptr) {
            host_buft = simple_host_buft;
        } else if (host_buft != simple_host_buft) {
            // if different simple devices have different host buffer types,
            // we cannot provide a single host buffer type for the meta device
            return nullptr;
        }
    }
    return host_buft;
}

//
// meta backend buffer
//

// Container to hold the tensor slices per simple ggml backend buffer.
// shadow tensors are keyed by the metadata address of the meta tensor; a fingerprint of the
// identity-defining fields guards against two ways that address can go stale:
//   - graph arenas reuse addresses for different tensors across graphs (in-process)
//   - RPC deserializes a fresh struct per command, often at a recycled arena address,
//     so the same weight arrives at different addresses and different weights can
//     arrive at the same address
// a mismatched fingerprint is treated as a miss and the shadow is recreated on demand,
// which is always possible because slice placement derives from tensor->data alone.
// mutable or context-dependent fields (buffer, flags, extra, src/view pointers) are
// deliberately excluded: they legitimately differ between registration and use
// (e.g. the buffer field is assigned after init) and across RPC deserializations.
struct ggml_backend_meta_tensor_fingerprint {
    enum ggml_type type;
    enum ggml_op   op;
    int64_t ne[GGML_MAX_DIMS];
    size_t  nb[GGML_MAX_DIMS];
    int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    void *  data;
    size_t  view_offs;
    char    name[GGML_MAX_NAME];

    ggml_backend_meta_tensor_fingerprint() = default;
    explicit ggml_backend_meta_tensor_fingerprint(const ggml_tensor * t) {
        memset(this, 0, sizeof(*this));
        type      = t->type;
        op        = t->op;
        memcpy(ne, t->ne, sizeof(ne));
        memcpy(nb, t->nb, sizeof(nb));
        memcpy(op_params, t->op_params, sizeof(op_params));
        data      = t->data;
        view_offs = t->view_offs;
        memcpy(name, t->name, sizeof(name));
    }

    bool matches(const ggml_tensor * t) const {
        const ggml_backend_meta_tensor_fingerprint other(t);
        return memcmp(this, &other, sizeof(*this)) == 0;
    }
};

struct ggml_backend_meta_simple_tensor_entry {
    std::vector<ggml_tensor *> shadows;
    ggml_backend_meta_tensor_fingerprint fingerprint;
    // captured at registration so that set/get_tensor on an RPC-deserialized struct
    // (whose src pointers are meaningless) never needs to re-derive it
    ggml_backend_meta_split_state split_state;
    // graph-link identity at registration time, for the init_tensor memo (the
    // fingerprint deliberately excludes these): the scheduler re-allocates the
    // same graph at the same addresses every token, and re-registering it would
    // grow the shadow arenas forever once the build cache stops ring rotation
    const ggml_tensor * reg_srcs[GGML_MAX_SRC] = {};
    const ggml_tensor * reg_view_src = nullptr;
    int32_t             reg_flags    = 0;
};

struct ggml_backend_meta_simple_tensor_container {
    std::vector<ggml_context_ptr> ctxs;
    // arenas that filled up and were replaced: their shadow tensors are still
    // referenced by entries in simple_tensors, so they stay alive until the
    // container is reset (ring rotation / free)
    std::vector<ggml_context_ptr> ctxs_retired;
    std::map<const ggml_tensor *, ggml_backend_meta_simple_tensor_entry> simple_tensors;
    // secondary index for identity-based resolution when the struct address differs
    // (every RPC command deserializes a fresh struct)
    std::map<const void *, const ggml_backend_meta_simple_tensor_entry *> by_data;

    ggml_backend_meta_simple_tensor_container(const ggml_init_params & params, const int n_simple) {
        ctxs.reserve(n_simple);
        for (int i = 0; i < n_simple; i++) {
            ctxs.emplace_back(ggml_init(params));
        }
    }
    ggml_backend_meta_simple_tensor_container() {}

    // shadow arena with guaranteed room for at least one more tensor. The initial
    // sizing is a heuristic (e.g. 16 views per static tensor) that real graphs
    // can exceed - DeltaNet layers create many views of the recurrent state per
    // graph - so a full arena chains a fresh one instead of aborting in
    // ggml_new_tensor (this crashed production: "not enough space in the
    // context's memory pool", TASKS.md).
    ggml_context * shadow_ctx(size_t j) {
        ggml_context * c = ctxs[j].get();
        if (ggml_get_mem_size(c) - ggml_used_mem(c) < 2*ggml_tensor_overhead()) {
            const ggml_init_params params = {
                /*.mem_size   =*/ ggml_get_mem_size(c),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ctxs_retired.push_back(std::move(ctxs[j]));
            ctxs[j].reset(ggml_init(params));
            c = ctxs[j].get();
        }
        return c;
    }
};

struct ggml_backend_meta_buffer_context {
    // FIXME
    // Most tensors can simply be stored statically in their own buffer.
    // Externally created views however also need a mapping to simple tensors but they use the buffer of the view source.
    // If external views are simply using that buffer they will slowly deplete its memory.
    // Current solution: rotating set of "compute" containers to hold external views, works correctly for llama.cpp.
    // The ring must hold at least 2 (current + previous graph); more slots allow multiple llama-side cached
    // graphs to keep their shadow tensors alive across shape switches. Shadows evicted by the rotation are
    // recreated on demand during subgraph rebuild (see ggml_backend_meta_buffer_ensure_simple_tensor).
    // Long-term: tie the lifetime of external views to the meta backend executing the graph instead,
    //     currently not possible due to graph-external operations in the backend scheduler.
    ggml_backend_meta_simple_tensor_container stc_static;
    std::vector<ggml_backend_meta_simple_tensor_container> stc_compute;
    int stc_compute_index      = 0;
    int stc_compute_index_next = 0;
    std::vector<ggml_backend_buffer_ptr> bufs;

    // FIXME
    // The size of the split state cache is unbounded and can theoretically grow infinitely large.
    // However, it is also expensive to build and clearing it on every rebuild in ggml_backend_meta_graph_compute is too expensive.
    static constexpr size_t nbtc = GGML_TENSOR_SIZE - sizeof(ggml_tensor::padding);
    std::map<std::pair<const ggml_tensor *, bool>, std::pair<ggml_backend_meta_split_state, char[nbtc]>> split_state_cache;

    int debug;

    // live-context registry so a surgical re-provision can enumerate every meta
    // buffer's member buffers and shadow registrations (TASKS.md #29c refinement)
    static std::mutex             & registry_mutex() { static std::mutex m; return m; }
    static std::set<ggml_backend_meta_buffer_context *> & registry() {
        static std::set<ggml_backend_meta_buffer_context *> r;
        return r;
    }

    ggml_backend_meta_buffer_context(
            ggml_backend_meta_simple_tensor_container & stc_static,
            const ggml_init_params & ctx_params,
            const int n_simple,
            const std::vector<ggml_backend_buffer_t> & bufs)
            : stc_static(std::move(stc_static)) {
        {
            std::lock_guard<std::mutex> lock(registry_mutex());
            registry().insert(this);
        }
        // note: should stay above LLAMA_DECODE_GRAPH_CACHE (llama-context.cpp) plus the
        // in-flight graphs, otherwise cached graphs churn through shadow recreation
        const char * GGML_META_MAX_GRAPHS = getenv("GGML_META_MAX_GRAPHS");
        const int n_graphs = std::max(2, GGML_META_MAX_GRAPHS ? atoi(GGML_META_MAX_GRAPHS) : 8);
        stc_compute.reserve(n_graphs);
        for (int i = 0; i < n_graphs; i++) {
            stc_compute.emplace_back(ctx_params, n_simple);
        }
        this->bufs.reserve(bufs.size());
        for (ggml_backend_buffer_t buf : bufs) {
            this->bufs.emplace_back(buf);
        }
        const char * GGML_META_DEBUG = getenv("GGML_META_DEBUG");
        debug = GGML_META_DEBUG ? atoi(GGML_META_DEBUG) : 0;
    }

    ~ggml_backend_meta_buffer_context() {
        std::lock_guard<std::mutex> lock(registry_mutex());
        registry().erase(this);
    }

    ggml_backend_meta_simple_tensor_container & get_simple_tensor_container(const ggml_tensor * tensor) {
        if (stc_static.simple_tensors.find(tensor) != stc_static.simple_tensors.end()) {
            return stc_static;
        }
        return stc_compute[stc_compute_index];
    }
};

static void ggml_backend_meta_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    delete buf_ctx;
}

// surgical re-provision (TASKS.md #29c refinement): a worker at `endpoint` died and
// came back at the same address. Re-create every member buffer that lived there,
// replay the weight journal (worker-cache hash replay / journaled small writes) and
// rebase this side's shadow registrations onto the fresh allocations. KV and
// compute member buffers are re-allocated empty - safe in the EP topology, where a
// worker member holds no cross-token state; a non-weight member buffer big enough
// to plausibly hold real state fails the attempt (caller falls back to a reload).
bool ggml_backend_meta_reprovision_endpoint(const char * endpoint) {
    if (endpoint == nullptr) {
        return false;
    }
    ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
    if (rpc_reg == nullptr) {
        return false;
    }
    typedef const char * (*buf_endpoint_t)(ggml_backend_buffer_t);
    typedef bool (*buf_reprovision_t)(ggml_backend_buffer_t, void **, void **);
    typedef void (*clear_failed_t)(const char *);
    auto buf_endpoint_fn = (buf_endpoint_t) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_buffer_endpoint");
    auto buf_reprov_fn   = (buf_reprovision_t) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_buffer_reprovision");
    auto clear_failed_fn = (clear_failed_t) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_clear_failed_endpoint");
    if (buf_endpoint_fn == nullptr || buf_reprov_fn == nullptr || clear_failed_fn == nullptr) {
        return false;
    }
    static const long long max_state_mib = []() {
        const char * env = getenv("GGML_META_SURGICAL_MAX_STATE_MIB");
        return env != nullptr ? atoll(env) : 64LL;
    }();

    std::lock_guard<std::mutex> lock(ggml_backend_meta_buffer_context::registry_mutex());
    struct target {
        ggml_backend_meta_buffer_context * ctx;
        size_t                             j;
        ggml_backend_buffer_t              buf;
    };
    std::vector<target> targets;
    for (ggml_backend_meta_buffer_context * ctx : ggml_backend_meta_buffer_context::registry()) {
        for (size_t j = 0; j < ctx->bufs.size(); j++) {
            ggml_backend_buffer_t b = ctx->bufs[j].get();
            const char * ep = buf_endpoint_fn(b);
            if (ep == nullptr || strcmp(ep, endpoint) != 0) {
                continue;
            }
            // compute members are exempt: galloc rebuilds their contents from the
            // meta logical addresses every graph, nothing survives across tokens
            if (b->usage == GGML_BACKEND_BUFFER_USAGE_ANY &&
                ggml_backend_buffer_get_size(b) > (size_t) max_state_mib * 1024 * 1024) {
                GGML_LOG_WARN("meta: %s holds a %zu MiB stateful member buffer - its contents died "
                              "with the worker, only a reload can rebuild it\n",
                              endpoint, ggml_backend_buffer_get_size(b) / (1024*1024));
                return false;
            }
            targets.push_back({ctx, j, b});
        }
    }
    if (targets.empty()) {
        return false;
    }
    for (target & t : targets) {
        void * old_base = nullptr;
        void * new_base = nullptr;
        if (!buf_reprov_fn(t.buf, &old_base, &new_base)) {
            return false; // endpoint stays marked failed; the caller reloads
        }
        if (old_base == new_base) {
            continue;
        }
        const char * ob = (const char *) old_base;
        const size_t sz = ggml_backend_buffer_get_size(t.buf);
        auto rebase = [&](ggml_backend_meta_simple_tensor_container & stc) {
            for (auto & kv : stc.simple_tensors) {
                auto & shadows = kv.second.shadows;
                if (t.j >= shadows.size() || shadows[t.j] == nullptr) {
                    continue;
                }
                char * d = (char *) shadows[t.j]->data;
                if (d >= ob && d < ob + sz) {
                    shadows[t.j]->data = (char *) new_base + (d - ob);
                }
            }
        };
        rebase(t.ctx->stc_static);
        for (auto & stc : t.ctx->stc_compute) {
            rebase(stc);
        }
        // rebase completeness check: every shadow registered on this member buffer
        // must now point into the fresh allocation - a survivor outside it was
        // derived from a base the range rebase does not know about
        {
            size_t n_stale = 0;
            auto scan = [&](ggml_backend_meta_simple_tensor_container & stc) {
                for (auto & kv : stc.simple_tensors) {
                    auto & shadows = kv.second.shadows;
                    if (t.j >= shadows.size() || shadows[t.j] == nullptr || shadows[t.j]->buffer != t.buf) {
                        continue;
                    }
                    const char * d = (const char *) shadows[t.j]->data;
                    if (ggml_nbytes(shadows[t.j]) > 0 &&
                        (d < (const char *) new_base || d >= (const char *) new_base + sz)) {
                        n_stale++;
                    }
                }
            };
            scan(t.ctx->stc_static);
            for (auto & stc : t.ctx->stc_compute) {
                scan(stc);
            }
            if (n_stale > 0) {
                GGML_LOG_WARN("meta: %s: %zu shadow tensors on a re-provisioned member buffer "
                              "still point outside the fresh allocation\n", endpoint, n_stale);
            }
        }
        // graph-computed static state (e.g. deepseek4's hash-layer tables) never
        // passes through the client, so the journal cannot restore it - copy
        // MIRRORED entries of non-weight static buffers from a healthy member
        // (mirrored bytes are identical on every member by definition)
        if (t.buf->usage != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            size_t n_sibling  = 0;
            size_t nb_sibling = 0;
            std::vector<uint8_t> tmp;
            for (auto & kv : t.ctx->stc_static.simple_tensors) {
                const ggml_backend_meta_simple_tensor_entry & entry = kv.second;
                if (entry.split_state.axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED ||
                    kv.first->view_src != nullptr) {
                    continue;
                }
                if (t.j >= entry.shadows.size() || entry.shadows[t.j] == nullptr ||
                    entry.shadows[t.j]->buffer != t.buf) {
                    continue;
                }
                ggml_tensor * src = nullptr;
                for (size_t m = 0; m < entry.shadows.size(); m++) {
                    if (m != t.j && entry.shadows[m] != nullptr) {
                        const char * mep = buf_endpoint_fn(entry.shadows[m]->buffer);
                        if (mep == nullptr || strcmp(mep, endpoint) != 0) {
                            src = entry.shadows[m];
                            break;
                        }
                    }
                }
                if (src == nullptr) {
                    continue;
                }
                const size_t nbytes = ggml_nbytes(src);
                tmp.resize(nbytes);
                ggml_backend_tensor_get(src, tmp.data(), 0, nbytes);
                ggml_backend_tensor_set(entry.shadows[t.j], tmp.data(), 0, nbytes);
                n_sibling++;
                nb_sibling += nbytes;
            }
            GGML_LOG_INFO("meta: %s: sibling-restored %zu mirrored statics (%zu bytes) on a "
                          "%zu MiB member buffer\n", endpoint, n_sibling, nb_sibling,
                          ggml_backend_buffer_get_size(t.buf) / (1024*1024));
        }
    }
    clear_failed_fn(endpoint);
    GGML_LOG_INFO("meta: endpoint %s surgically re-provisioned (%zu member buffers)\n", endpoint, targets.size());
    return true;
}

static size_t ggml_backend_meta_buffer_n_bufs(ggml_backend_buffer_t meta_buf) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    return buf_ctx->bufs.size();
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_simple_buffer(ggml_backend_buffer_t meta_buf, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    GGML_ASSERT(index < buf_ctx->bufs.size());
    return buf_ctx->bufs[index].get();
}

static struct ggml_tensor * ggml_backend_meta_buffer_simple_tensor(const struct ggml_tensor * tensor, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    GGML_ASSERT(index < buf_ctx->bufs.size());

    {
        // static entries are only registered by the in-process alloc_ctx_tensors path where
        // the struct address is stable for the buffer's lifetime - no fingerprint needed,
        // and per-device placement there does not follow the offset formula that on-demand
        // recreation uses, so these entries must never be invalidated
        auto it = buf_ctx->stc_static.simple_tensors.find(tensor);
        if (it != buf_ctx->stc_static.simple_tensors.end()) {
            return it->second.shadows[index];
        }
    }

    // search the compute containers newest to oldest: a reused metadata address
    // must resolve to the shadow of the most recently registered tensor
    const int n_stc = (int) buf_ctx->stc_compute.size();
    for (int k = 0; k < n_stc; k++) {
        ggml_backend_meta_simple_tensor_container & stc =
            buf_ctx->stc_compute[(buf_ctx->stc_compute_index - k + n_stc) % n_stc];
        auto it = stc.simple_tensors.find(tensor);
        if (it != stc.simple_tensors.end() && it->second.fingerprint.matches(tensor)) {
            return it->second.shadows[index];
        }
    }
    return nullptr;
}

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const struct ggml_tensor * tensor, bool assume_sync);

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(
        ggml_backend_meta_simple_tensor_container & stc, const struct ggml_tensor * tensor, bool assume_sync) {
    // FIXME Currently this function preserves/erases the information in n_segments and nr in an inconsistent way.
    // Since the operations in question are developed specifically for llama.cpp this currently does not manifest as a bug there.
    // However, in a broader ggml context with arbitrary ggml graphs this can lead to unexpected results.
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;

    auto split_states_equal = [&](const ggml_backend_meta_split_state & a, const ggml_backend_meta_split_state & b) -> bool {
        if (a.axis != b.axis) {
            return false;
        }
        for (size_t j = 0; j < n_bufs; j++) {
            int64_t sum_a = 0;
            for (size_t s = 0; s < a.n_segments; s++) {
                sum_a += a.ne[s*n_bufs + j] * a.nr[s];
            }
            int64_t sum_b = 0;
            for (size_t s = 0; s < b.n_segments; s++) {
                sum_b += b.ne[s*n_bufs + j] * b.nr[s];
            }
            if (sum_a != sum_b) {
                return false;
            }
        }
        return true;
    };

    // a DEGENERATE axis split holds every slice on ONE member (dedicated
    // attention, TASKS #28 inc 3): rows and dot dimensions stay complete on the
    // owner, so combinations that must be rejected for REAL feature-dim splits
    // are exact for it - the owner computes the full op, everyone else follows
    // the zero-slice path (compute disabled). Returns the owner index, or -1.
    auto split_state_owner = [&](const ggml_backend_meta_split_state & ss) -> int {
        if (ss.axis < 0 || ss.axis >= GGML_MAX_DIMS) {
            return -1;
        }
        int owner = -1;
        for (size_t j = 0; j < n_bufs; j++) {
            int64_t sum = 0;
            for (size_t s = 0; s < ss.n_segments; s++) {
                sum += ss.ne[s*n_bufs + j] * ss.nr[s];
            }
            if (sum != 0) {
                if (owner >= 0) {
                    return -1;
                }
                owner = (int) j;
            }
        }
        return owner;
    };
    auto degenerate_state = [&](const int axis, const int owner, const int64_t ne_axis) -> ggml_backend_meta_split_state {
        ggml_backend_meta_split_state ret = {ggml_backend_meta_split_axis(axis), {0}, {1}, 1};
        ret.ne[owner] = ne_axis;
        return ret;
    };

    auto handle_generic = [&](const std::vector<ggml_backend_meta_split_state> & src_ss, bool scalar_only) -> ggml_backend_meta_split_state {
        ggml_backend_meta_split_state ret = {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1};
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                continue;
            }
            if (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
                ret = src_ss[i];
                continue;
            }
            if (split_states_equal(src_ss[i], ret)) {
                continue;
            }
            // mirrored operands combine with a same-owner degenerate one: the
            // owner has everything, the rest are zero-size
            const int owner_ret = split_state_owner(ret);
            const int owner_i   = split_state_owner(src_ss[i]);
            if (owner_i >= 0 && ret.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                ret = src_ss[i];
                continue;
            }
            if (owner_ret >= 0 && (src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || owner_i == owner_ret)) {
                continue;
            }
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            break;
        }
        if (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        if (scalar_only && ret.axis >= 0 && ret.axis < GGML_MAX_DIMS && split_state_owner(ret) < 0) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        if (ret.axis == GGML_BACKEND_SPLIT_AXIS_UNKNOWN) {
            GGML_ABORT("unsupported generic split combination: %s[%s] srcs [%s,%s,%s]",
                    tensor->name, ggml_op_name(tensor->op),
                    tensor->src[0] ? ggml_backend_meta_split_axis_name(src_ss[0].axis) : "-",
                    tensor->src[1] ? ggml_backend_meta_split_axis_name(src_ss[1].axis) : "-",
                    tensor->src[2] ? ggml_backend_meta_split_axis_name(src_ss[2].axis) : "-");
        }
        return ret;
    };

    // Some ops process data on a per-row bases:
    auto handle_per_row = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        // a feature-dim split breaks row integrity - except degenerate, where the
        // owner's rows are complete
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0 || split_state_owner(src_ss[0]) >= 0);
        return src_ss[0];
    };

    // Some ops broadcast the src1 data across src0:
    auto handle_bin_bcast = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS &&
                tensor->src[1]->ne[src_ss[0].axis] == 1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        if (src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[0].axis == src_ss[1].axis ||
           (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL)))) {
            return src_ss[0]; // GGML_OP_ADD_ID
        }
        GGML_ASSERT(tensor->src[2] == nullptr || src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        {
            const int owner0 = split_state_owner(src_ss[0]);
            const int owner1 = split_state_owner(src_ss[1]);
            if (owner0 >= 0 && (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || owner1 == owner0)) {
                return src_ss[0];
            }
            if (owner1 >= 0 && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                // only the owner has the second operand; everyone else takes the
                // zero-slice path
                return degenerate_state(src_ss[1].axis, owner1, tensor->ne[src_ss[1].axis]);
            }
        }
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_concat = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        const ggml_backend_meta_split_axis concat_axis = ggml_backend_meta_split_axis(ggml_get_op_params_i32(tensor, 0));
        {
            // dedicated attention: a degenerate operand makes the whole concat
            // owner-only (mirrored operands are available there too)
            int owner = split_state_owner(src_ss[0]);
            if (owner < 0) {
                owner = split_state_owner(src_ss[1]);
            }
            if (owner >= 0 &&
                    (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || split_state_owner(src_ss[0]) == owner) &&
                    (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || split_state_owner(src_ss[1]) == owner)) {
                return degenerate_state(GGML_BACKEND_SPLIT_AXIS_0, owner, tensor->ne[0]);
            }
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis >= 0 && src_ss[1].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[1].axis);
            return src_ss[1];
        }
        if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[0].axis);
            return src_ss[0];
        }
        if (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis != concat_axis) {
            return src_ss[0];
        }
        return handle_generic(src_ss, /*scalar_only =*/ true);
    };

    auto handle_mul_mat = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ggml_backend_meta_split_state ret = src_ss[0];
            ret.axis = GGML_BACKEND_SPLIT_AXIS_0;
            ret.nr[0] = 1;
            ret.n_segments = 1;
            return ret;
        }
        if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[1];
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            {
                // same-owner degenerate operands: PARTIAL is only wanted at the
                // designated island EXIT (the dedicated attention out projection,
                // dot-dim split by the policy - names must stay in sync with
                // pattern_dsa_attn_exit in llama-model.cpp). Everything else
                // (indexer scores, cache reads) stays on the owner - a broadcast
                // there is pure boundary overhead.
                const int owner0 = split_state_owner(src_ss[0]);
                if (owner0 >= 0 && owner0 == split_state_owner(src_ss[1]) &&
                        !(tensor->src[0]->op == GGML_OP_NONE && strstr(tensor->src[0]->name, "attn_output_b") != nullptr)) {
                    return degenerate_state(GGML_BACKEND_SPLIT_AXIS_0, owner0, tensor->ne[0]);
                }
            }
            GGML_ASSERT(split_states_equal(src_ss[0], src_ss[1]));
            return {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
        }
        {
            // degenerate operands: the owner holds the full matrices, so mixing
            // with mirrored or same-owner operands is exact on the owner
            const int owner0 = split_state_owner(src_ss[0]);
            const int owner1 = split_state_owner(src_ss[1]);
            if (owner1 >= 0 && (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || owner0 == owner1)) {
                return degenerate_state(GGML_BACKEND_SPLIT_AXIS_0, owner1, tensor->ne[0]);
            }
            if (owner0 >= 0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                return degenerate_state(GGML_BACKEND_SPLIT_AXIS_0, owner0, tensor->ne[0]);
            }
        }
        // note: do not print src names here - over RPC the src pointers of a lone
        // deserialized struct can dangle and turn this abort into a silent segfault
        GGML_ABORT("unsupported mul_mat split combination: %s = src0[%s] x src1[%s]",
                tensor->name,
                ggml_backend_meta_split_axis_name(src_ss[0].axis),
                ggml_backend_meta_split_axis_name(src_ss[1].axis));
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    auto handle_reshape = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1:
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3: {
                GGML_ASSERT(src_ss[0].n_segments == 1);
                if (src_ss[0].axis == ggml_n_dims(tensor->src[0]) - 1 && src_ss[0].nr[0] == 1) {
                    return {ggml_backend_meta_split_axis(ggml_n_dims(tensor) - 1), {0}, {1}, 1};
                }
                int64_t base_ne_in = tensor->src[0]->ne[0];
                for (int dim = 1; dim <= src_ss[0].axis; dim++) {
                    base_ne_in *= tensor->src[0]->ne[dim];
                }
                base_ne_in /= src_ss[0].nr[0];
                int64_t base_ne_out = 1;
                for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                    const int64_t base_ne_out_next = base_ne_out *= tensor->ne[dim];
                    if (base_ne_out_next % base_ne_in == 0) {
                        return {ggml_backend_meta_split_axis(dim), {0}, {uint32_t(base_ne_out_next/base_ne_in)}, 1};
                    }
                    if (base_ne_out_next > base_ne_in) {
                        GGML_ASSERT(src_ss[0].n_segments == 1);
                        GGML_ASSERT(src_ss[0].nr[0]      == 1);
                        return {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                    }
                    base_ne_out = base_ne_out_next;
                }
                GGML_ABORT("shape mismatch for %s", ggml_op_name(tensor->op));
            }
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_cpy = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            return handle_reshape(src_ss);
        }
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_view = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (ggml_is_contiguous(tensor) && ggml_is_contiguous(tensor->src[0])) {
            return handle_reshape(src_ss);
        }
        const int axis = src_ss[0].axis;
        {
            bool all_strides_the_same = true;
            for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                if (tensor->ne[dim] == 1 && tensor->src[0]->ne[dim] == 1) {
                    continue;
                }
                if (tensor->nb[dim] != tensor->src[0]->nb[dim]) {
                    all_strides_the_same = false;
                    break;
                }
            }
            if (all_strides_the_same) {
                return src_ss[0];
            }
        }
        if (!ggml_is_permuted(tensor) && !ggml_is_permuted(tensor->src[0]) && axis >= 0 && axis < GGML_MAX_DIMS-1) {
            for (int dim = 0; dim < GGML_MAX_DIMS-1; dim++) {
                if (tensor->nb[dim+1] == tensor->src[0]->nb[axis+1]) {
                    return {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                }
            }
            GGML_ABORT("fatal error");
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
            return src_ss[0];
        }
        GGML_ABORT("view of permuted tensor not implemented");
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    auto handle_permute = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1:
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                return {ggml_backend_meta_split_axis(tensor->op_params[src_ss[0].axis]), {0}, {src_ss[0].nr[0]}, 1};
            }
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_transpose = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                return {ggml_backend_meta_split_axis(int(src_ss[0].axis) ^ 1), {0}, {src_ss[0].nr[0]}, 1};
            }
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3:
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_get_rows = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        return handle_generic(src_ss, /*scalar_only =*/ true);
    };

    auto handle_set_rows = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        {
            // dedicated attention: if any operand lives on one member and the rest
            // are mirrored or on the same member, the whole write happens there
            // (e.g. the dsv4 sparse mask: owner-computed top-k ids scatter into a
            // mirrored mask that only the owner's attention consumes)
            int owner = split_state_owner(src_ss[0]);
            if (owner < 0) {
                owner = split_state_owner(src_ss[1]);
            }
            if (owner < 0) {
                owner = split_state_owner(src_ss[2]);
            }
            if (owner >= 0) {
                bool consistent = true;
                for (int i = 0; i < 3; i++) {
                    consistent = consistent &&
                        (src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || split_state_owner(src_ss[i]) == owner);
                }
                if (consistent) {
                    if (split_state_owner(src_ss[2]) == owner) {
                        return src_ss[2];
                    }
                    return degenerate_state(GGML_BACKEND_SPLIT_AXIS_0, owner, tensor->ne[0]);
                }
            }
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1 ||
                src_ss[1].axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED ||
                !split_states_equal(src_ss[0], src_ss[2])) {
            GGML_ABORT("unsupported set_rows split combination: %s srcs [%s,%s,%s] owners [%d,%d,%d]",
                    tensor->name,
                    ggml_backend_meta_split_axis_name(src_ss[0].axis),
                    ggml_backend_meta_split_axis_name(src_ss[1].axis),
                    ggml_backend_meta_split_axis_name(src_ss[2].axis),
                    split_state_owner(src_ss[0]), split_state_owner(src_ss[1]), split_state_owner(src_ss[2]));
        }
        return src_ss[0];
    };

    auto handle_rope = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return src_ss[0];
    };

    auto handle_pad = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 0] == 0);
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 1] == 0);
        }
        return src_ss[0];
    };

    auto handle_flash_attn_ext = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        // fully mirrored attention (e.g. MLA models with a single shared latent KV head):
        // every device runs the whole attention block, nothing to synchronize
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
            src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
            src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            GGML_ASSERT(tensor->src[3] == nullptr || src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        {
            // dedicated attention: q/k/v (and mask/sinks, where present) all live
            // on one member; the whole attention op runs there
            const int owner0 = split_state_owner(src_ss[0]);
            bool dedicated = owner0 >= 0;
            for (int i = 1; i < 5 && dedicated; i++) {
                if (tensor->src[i] == nullptr) {
                    continue;
                }
                dedicated = src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || split_state_owner(src_ss[i]) == owner0;
            }
            if (dedicated) {
                return degenerate_state(GGML_BACKEND_SPLIT_AXIS_1, owner0, tensor->ne[1]);
            }
        }
        GGML_ASSERT(                             src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2);
        GGML_ASSERT(                             src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_2);
        GGML_ASSERT(                             src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_2);
        GGML_ASSERT(tensor->src[4] == nullptr || src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_0);
        return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
    };

    auto handle_ssm_conv = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == src_ss[1].axis) {
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {
                return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
            }
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1) {
                return {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
            }
        }
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_gated_delta_net = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_1);
        // state shape is [S_v, S_v, H_v, n_seqs] (s0 only); the heads dim is its own axis 2,
        // so a head-aligned split on the input cache lands on axis 2 here.
        GGML_ASSERT(src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_2 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_1 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_0);
        return {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
    };

    auto calculate_split_state = [&]() -> ggml_backend_meta_split_state {
        if (ggml_nelements(tensor) == 0) {
            return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        if (ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE && tensor->view_src == nullptr) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
            const ggml_backend_meta_device_context * dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
            ggml_backend_meta_split_state ret = dev_ctx->get_split_state(tensor, dev_ctx->get_split_state_ud);
            if (ret.axis >= 0 && ret.axis <= GGML_MAX_DIMS) {
                const int64_t granularity = ret.axis == GGML_BACKEND_SPLIT_AXIS_0 ? ggml_blck_size(tensor->type) : 1;
                int64_t ne_sum = 0;
                for (size_t s = 0; s < ret.n_segments; s++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        GGML_ASSERT(ret.ne[s*n_bufs + j] % granularity == 0);
                        ne_sum += ret.ne[s*n_bufs + j] * ret.nr[s];
                    }
                }
                GGML_ASSERT(ne_sum == tensor->ne[ret.axis]);
            }
            return ret;
        }

        std::vector<ggml_backend_meta_split_state> src_ss(GGML_MAX_SRC, {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1});
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                src_ss[i] = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
                continue;
            }
            src_ss[i] = ggml_backend_meta_get_split_state(stc, tensor->src[i], /*assume_sync =*/ true);
            if (buf_ctx->debug > 1) {
                GGML_LOG_DEBUG("SRC_RESOLVE: %s[%s] src%zu %s[%s] -> %s\n",
                        tensor->name, ggml_op_name(tensor->op), i, tensor->src[i]->name,
                        ggml_op_name(tensor->src[i]->op), ggml_backend_meta_split_axis_name(src_ss[i].axis));
            }
            GGML_ASSERT(src_ss[i].axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        }

        ggml_backend_meta_split_state split_state;
        switch (tensor->op) {
            case GGML_OP_NONE: {
                split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
            } break;
            case GGML_OP_DUP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_ADD:
            case GGML_OP_ADD_ID: {
                split_state = handle_bin_bcast(src_ss);
            } break;
            case GGML_OP_ADD1:
            case GGML_OP_ACC: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SUB:
            case GGML_OP_MUL:
            case GGML_OP_DIV: {
                split_state = handle_bin_bcast(src_ss);
            } break;
            case GGML_OP_SQR:
            case GGML_OP_SQRT:
            case GGML_OP_LOG:
            case GGML_OP_SIN:
            case GGML_OP_COS: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_SUM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SUM_ROWS:
            case GGML_OP_CUMSUM:
            case GGML_OP_MEAN:
            case GGML_OP_ARGMAX:
            case GGML_OP_COUNT_EQUAL: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_REPEAT:
            case GGML_OP_REPEAT_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_CONCAT: {
                split_state = handle_concat(src_ss);
            } break;
            case GGML_OP_SILU_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
            case GGML_OP_RMS_NORM_BACK:
            case GGML_OP_GROUP_NORM:
            case GGML_OP_L2_NORM: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_MUL_MAT:
            case GGML_OP_MUL_MAT_ID: {
                split_state = handle_mul_mat(src_ss);
            } break;
            case GGML_OP_OUT_PROD: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_SET: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CPY: {
                split_state = handle_cpy(src_ss);
            } break;
            case GGML_OP_CONT:
            case GGML_OP_RESHAPE: {
                split_state = handle_reshape(src_ss);
            } break;
            case GGML_OP_VIEW: {
                split_state = handle_view(src_ss);
            } break;
            case GGML_OP_PERMUTE: {
                split_state = handle_permute(src_ss);
            } break;
            case GGML_OP_TRANSPOSE: {
                split_state = handle_transpose(src_ss);
            } break;
            case GGML_OP_GET_ROWS: {
                split_state = handle_get_rows(src_ss);
            } break;
            case GGML_OP_GET_ROWS_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SET_ROWS: {
                split_state = handle_set_rows(src_ss);
            } break;
            case GGML_OP_DIAG:
            case GGML_OP_DIAG_MASK_INF:
            case GGML_OP_DIAG_MASK_ZERO: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SOFT_MAX:
            case GGML_OP_SOFT_MAX_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_ROPE: {
                split_state = handle_rope(src_ss);
            } break;
            case GGML_OP_ROPE_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CLAMP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_CONV_TRANSPOSE_1D:
            case GGML_OP_IM2COL:
            case GGML_OP_IM2COL_BACK:
            case GGML_OP_IM2COL_3D:
            case GGML_OP_CONV_2D:
            case GGML_OP_CONV_3D:
            case GGML_OP_CONV_2D_DW:
            case GGML_OP_CONV_TRANSPOSE_2D:
            case GGML_OP_POOL_1D:
            case GGML_OP_POOL_2D:
            case GGML_OP_POOL_2D_BACK:
            case GGML_OP_UPSCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_PAD: {
                split_state = handle_pad(src_ss);
            } break;
            case GGML_OP_PAD_REFLECT_1D:
            case GGML_OP_ROLL:
            case GGML_OP_ARANGE:
            case GGML_OP_TIMESTEP_EMBEDDING: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_ARGSORT:
            case GGML_OP_TOP_K: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_LEAKY_RELU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_FILL: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_FLASH_ATTN_EXT: {
                split_state = handle_flash_attn_ext(src_ss);
            } break;
            case GGML_OP_FLASH_ATTN_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SSM_CONV: {
                split_state = handle_ssm_conv(src_ss);
            } break;
            case GGML_OP_SSM_SCAN:
            case GGML_OP_WIN_PART:
            case GGML_OP_WIN_UNPART:
            case GGML_OP_GET_REL_POS:
            case GGML_OP_ADD_REL_POS:
            case GGML_OP_RWKV_WKV6:
            case GGML_OP_GATED_LINEAR_ATTN:
            case GGML_OP_RWKV_WKV7:
            case GGML_OP_SOLVE_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_GATED_DELTA_NET: {
                split_state = handle_gated_delta_net(src_ss);
            } break;
            case GGML_OP_UNARY: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_MAP_CUSTOM1:
            case GGML_OP_MAP_CUSTOM2:
            case GGML_OP_MAP_CUSTOM3:
            case GGML_OP_CUSTOM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CROSS_ENTROPY_LOSS:
            case GGML_OP_CROSS_ENTROPY_LOSS_BACK: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_OPT_STEP_ADAMW:
            case GGML_OP_OPT_STEP_SGD:
            case GGML_OP_GLU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            default: {
                GGML_ABORT("ggml op not implemented: %s", ggml_op_name(tensor->op));
                split_state = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            } break;
        }
        if (split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS) {
            bool first_src_split_by_axis = true;
            const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);

            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr || src_ss[i].axis < 0 || src_ss[i].axis >= GGML_MAX_DIMS) {
                    continue;
                }
                if (first_src_split_by_axis) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        // Take over ratio from src:
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[s*n_bufs + j] = 0;
                        }
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[j] += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        split_state.ne[j] *= tensor->ne[split_state.axis];
                        if (split_state.ne[j] != 0 || tensor->src[i]->ne[src_ss[i].axis] != 0) {
                            const int64_t div = tensor->src[i]->ne[src_ss[i].axis] * split_state.nr[0];
                            GGML_ASSERT(split_state.ne[j] % div == 0);
                            split_state.ne[j] /= div;
                        }
                    }
                } else {
                    GGML_ASSERT(split_state.n_segments == 1);
                    for (size_t j = 0; j < n_bufs; j++) {
                        // Assert that ratio is consistent:
                        int64_t sum = 0;
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            sum += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        GGML_ASSERT(split_state.ne[j]*split_state.nr[0] * tensor->src[i]->ne[src_ss[i].axis]
                                                                 == sum * tensor->ne[split_state.axis]);
                    }
                }
                first_src_split_by_axis = false;
            }
            GGML_ASSERT(!first_src_split_by_axis);
        }
        return split_state;
    };

    const std::pair key = std::make_pair(tensor, assume_sync);
    auto it = buf_ctx->split_state_cache.find(key);
    if (it != buf_ctx->split_state_cache.end() && memcmp(it->second.second, (const char *) tensor, sizeof(it->second.second)) != 0) {
        // the metadata address was reused for a different tensor: drop only this
        // entry; entries of other live graphs (distinct arenas) remain valid
        buf_ctx->split_state_cache.erase(it);
        it = buf_ctx->split_state_cache.end();
    }

    if (it == buf_ctx->split_state_cache.end()) {
        buf_ctx->split_state_cache[key].first = calculate_split_state();
        memcpy(buf_ctx->split_state_cache[key].second, tensor, sizeof(buf_ctx->split_state_cache[key].second));
        if (buf_ctx->debug > 0) {
            std::string srcs_info;
            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr) {
                    continue;
                }
                if (!srcs_info.empty()) {
                    srcs_info += ", ";
                }
                const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor->src[0], true);
                GGML_ASSERT(split_state.n_segments == 1);
                const char * axis_name = ggml_backend_meta_split_axis_name(split_state.axis);
                std::string ne_info;
                for (size_t j = 0; j < n_bufs; j++) {
                    if (!ne_info.empty()) {
                        ne_info += ", ";
                    }
                    ne_info += std::to_string(split_state.ne[j]) + "x" + std::to_string(split_state.nr[0]);
                }
                srcs_info += std::string(tensor->src[i]->name) + "[" + ggml_op_name(tensor->src[i]->op) + ", " + axis_name + ", {" + ne_info + "}]";
            }
            std::string ne_info;
            for (size_t j = 0; j < n_bufs; j++) {
                if (!ne_info.empty()) {
                    ne_info += ", ";
                }
                const ggml_backend_meta_split_state & ss = buf_ctx->split_state_cache[key].first;
                ne_info += std::to_string(ss.ne[j]) + "x" + std::to_string(ss.nr[0]);
            }
            GGML_LOG_DEBUG("SPLIT_STATE: {%s} -> %s[%s, %s, {%s}]\n", srcs_info.c_str(), tensor->name, ggml_op_name(tensor->op),
                ggml_backend_meta_split_axis_name(buf_ctx->split_state_cache[key].first.axis), ne_info.c_str());
        }
    }

    ggml_backend_meta_split_state ret = buf_ctx->split_state_cache[key].first;
    GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_NONE);
#ifndef NDEBUG
    if (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
        int64_t ne_ret = 0;
        for (size_t s = 0; s < ret.n_segments; s++) {
            for (size_t j = 0; j < n_bufs; j++) {
                ne_ret += ret.ne[s*n_bufs + j] * ret.nr[s];
            }
        }
        assert(ne_ret == tensor->ne[int(ret.axis)]);
    }
#endif // NDEBUG
    return ret;
}

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const struct ggml_tensor * tensor, bool assume_sync) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    return ggml_backend_meta_get_split_state(buf_ctx->get_simple_tensor_container(tensor), tensor, assume_sync);
}

static void * ggml_backend_meta_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return (void *) 0x1000000000000000; // FIXME
}

static enum ggml_status ggml_backend_meta_buffer_init_tensor_impl(ggml_backend_meta_simple_tensor_container & stc, ggml_tensor * tensor, bool allow_memo = false) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    const size_t n_simple_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);

    // memo for the galloc-facing init path only: per-token graph re-allocation
    // registers the same tensors at the same addresses with the same links, and
    // the existing shadows (data placement is deterministic from tensor->data)
    // remain exactly right. The graph-build force path must NOT take this: it
    // exists to overwrite registrations whose links went stale.
    if (allow_memo) {
        auto it = stc.simple_tensors.find(tensor);
        if (it != stc.simple_tensors.end() && it->second.fingerprint.matches(tensor) &&
                it->second.reg_view_src == tensor->view_src &&
                it->second.reg_flags    == tensor->flags &&
                memcmp(it->second.reg_srcs, tensor->src, sizeof(tensor->src)) == 0) {
            return GGML_STATUS_SUCCESS;
        }
    }

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(stc, tensor, /*assume_sync =*/ true);
    GGML_ASSERT(ggml_nelements(tensor) == 0 || split_state.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
    GGML_ASSERT(split_state.n_segments <= 16);

    int split_dim = split_state.axis;
    int64_t ne[GGML_MAX_DIMS];
    size_t  nb[GGML_MAX_DIMS];
    for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
        ne[k] = tensor->ne[k];
        nb[k] = tensor->nb[k];
    }

    std::vector<ggml_tensor *> simple_tensors;
    simple_tensors.reserve(n_simple_bufs);
    for (size_t j = 0; j < n_simple_bufs; j++) {
        ggml_context          * simple_ctx = stc.shadow_ctx(j); // grows when the arena is full
        ggml_backend_buffer_t   simple_buf = buf_ctx->bufs[j].get();

        if ((simple_buf != nullptr) && ggml_backend_buffer_is_multi_buffer(simple_buf)) {
            // see https://github.com/ggml-org/llama.cpp/issues/22197
            GGML_ABORT("multi buffers are not supported by the meta backend");
        }

        if (split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
            // TODO: the following assert fails for llama-parallel even though the results are correct:
            // GGML_ASSERT(ggml_is_contiguously_allocated(tensor));
            ne[split_dim] = 0;
            for (size_t s = 0; s < split_state.n_segments; s++) {
                ne[split_dim] += split_state.ne[s*n_simple_bufs + j] * split_state.nr[s];
            }
            for (int i = 0; i < GGML_MAX_DIMS; i++) {
                if (tensor->nb[i] > tensor->nb[split_dim]) {
                    nb[i] = tensor->nb[i] * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }

        ggml_tensor * t_ij = ggml_new_tensor(simple_ctx, tensor->type, GGML_MAX_DIMS, ne);
        t_ij->op = tensor->op;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            t_ij->nb[i] = nb[i];
        }
        t_ij->flags = tensor->flags;
        memcpy(t_ij->op_params, tensor->op_params, sizeof(tensor->op_params));
        ggml_set_name(t_ij, tensor->name);
        t_ij->buffer = simple_buf;
        t_ij->view_src = tensor->view_src;
        t_ij->view_offs = tensor->view_offs;
        if (t_ij->view_src != nullptr && ggml_backend_buffer_is_meta(t_ij->view_src->buffer)) {
            t_ij->view_src = ggml_backend_meta_buffer_simple_tensor(tensor->view_src, j);
            if (t_ij->view_offs > 0 && split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
                GGML_ASSERT(tensor->ne[split_dim] != 0);
                const int split_dim_view_src = ggml_backend_meta_get_split_state(tensor->view_src, /*assume_sync =*/ true).axis;
                GGML_ASSERT(split_dim_view_src >= 0 && split_dim_view_src < GGML_MAX_DIMS);

                // The offset can be internal to the data split, in those cases the view offset should not be scaled.
                // If however, the offset is larger than the data split then it needs to be scaled proportionally.
                bool split_internal_offset = t_ij->view_offs <= tensor->view_src->nb[split_dim_view_src];
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    const size_t dim_size = tensor->ne[i] * tensor->nb[i];
                    if (tensor->view_offs <= dim_size && dim_size < tensor->nb[split_dim]) {
                        split_internal_offset = true;
                        break;
                    }
                }
                if (!split_internal_offset) {
                    t_ij->view_offs = t_ij->view_offs * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }
        if (t_ij->view_src != nullptr) {
            t_ij->data = (char *) t_ij->view_src->data + t_ij->view_offs;
        } else if (simple_buf != nullptr) {
            t_ij->data = (char *) ggml_backend_buffer_get_base(simple_buf)
                + size_t(tensor->data) - size_t(ggml_backend_buffer_get_base(tensor->buffer));
        }
        t_ij->extra = tensor->extra;
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            t_ij->src[i] = tensor->src[i];
            if (tensor->src[i] == tensor) {
                t_ij->src[i] = t_ij;
            } else if (t_ij->src[i] != nullptr && ggml_backend_buffer_is_meta(t_ij->src[i]->buffer)) {
                t_ij->src[i] = ggml_backend_meta_buffer_simple_tensor(tensor->src[i], j);
            }
        }

        simple_tensors.push_back(t_ij);
    }

    // If one of the sources has a zero-sized slice, disable the computation:
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i] == nullptr || !ggml_backend_buffer_is_meta(tensor->src[i]->buffer)) {
            continue;
        }

        const ggml_backend_meta_split_state split_state_src = ggml_backend_meta_get_split_state(tensor->src[i], /*assume_sync =*/ true);
        if (split_state_src.axis < 0 || split_state_src.axis >= GGML_MAX_DIMS) {
            continue;
        }
        for (size_t j = 0; j < n_simple_bufs; j++) {
            int64_t ne_sum = 0;
            for (size_t s = 0; s < split_state_src.n_segments; s++) {
                ne_sum += split_state_src.ne[s*n_simple_bufs + j] * split_state_src.nr[s];
            }
            if (ne_sum == 0) {
                simple_tensors[j]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
            }
        }
    }

    auto & entry = stc.simple_tensors[tensor];
    entry.shadows     = std::move(simple_tensors);
    entry.fingerprint = ggml_backend_meta_tensor_fingerprint(tensor);
    entry.split_state = split_state;
    memcpy(entry.reg_srcs, tensor->src, sizeof(entry.reg_srcs));
    entry.reg_view_src = tensor->view_src;
    entry.reg_flags    = tensor->flags;
    stc.by_data[tensor->data] = &entry;

    return GGML_STATUS_SUCCESS;
}

// resolve a tensor by identity (data address + fingerprint) rather than struct address -
// RPC commands deserialize a fresh struct each time and its src pointers are meaningless,
// so neither pointer lookups nor split-state re-derivation can be trusted for them.
// src_stc receives the compute container that owns the entry, or nullptr for static.
static const ggml_backend_meta_simple_tensor_entry * ggml_backend_meta_buffer_find_by_identity(
        const struct ggml_tensor * tensor, ggml_backend_meta_simple_tensor_container ** src_stc) {
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;

    *src_stc = nullptr;
    {
        auto it = buf_ctx->stc_static.by_data.find(tensor->data);
        if (it != buf_ctx->stc_static.by_data.end() && it->second->fingerprint.matches(tensor)) {
            return it->second;
        }
    }
    const int n_stc = (int) buf_ctx->stc_compute.size();
    for (int k = 0; k < n_stc; k++) {
        ggml_backend_meta_simple_tensor_container & stc =
            buf_ctx->stc_compute[(buf_ctx->stc_compute_index - k + n_stc) % n_stc];
        auto it = stc.by_data.find(tensor->data);
        if (it != stc.by_data.end() && it->second->fingerprint.matches(tensor)) {
            *src_stc = &stc;
            return it->second;
        }
    }
    return nullptr;
}

// if the tensor is known by identity, alias its registration under the current struct
// address and seed the split-state cache, so every downstream pointer-based lookup works
// without ever walking the (possibly dangling) src pointers of a deserialized struct
static bool ggml_backend_meta_buffer_adopt_identity(const struct ggml_tensor * tensor) {
    if (ggml_backend_meta_buffer_simple_tensor(tensor, 0) != nullptr) {
        return true; // already resolvable by pointer
    }
    ggml_backend_meta_simple_tensor_container * src_stc = nullptr;
    const ggml_backend_meta_simple_tensor_entry * e = ggml_backend_meta_buffer_find_by_identity(tensor, &src_stc);
    if (e == nullptr) {
        return false;
    }
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;

    // the alias copies shadow POINTERS, so it must not outlive the container that owns
    // them: a ring-sourced alias registers into the source's own slot (both die in the
    // same ggml_reset; ring lookups are fingerprint-checked, so address recycling is
    // safe). A static-sourced alias goes into the current ring slot instead - its
    // shadows are immortal so eviction only costs a re-adoption, whereas registering
    // it into the static container would be wrong: the static pointer lookup skips
    // fingerprint checks and a recycled struct address would resolve to stale shadows.
    ggml_backend_meta_simple_tensor_container & cur =
        src_stc != nullptr ? *src_stc : buf_ctx->stc_compute[buf_ctx->stc_compute_index];
    auto & ne = cur.simple_tensors[tensor];
    ne = *e; // node-based map: inserting into the source's own map does not invalidate e
    cur.by_data[tensor->data] = &ne;

    for (int sync = 0; sync < 2; sync++) {
        auto & cached = buf_ctx->split_state_cache[std::make_pair(tensor, sync != 0)];
        cached.first = ne.split_state;
        memcpy(cached.second, (const char *) tensor, sizeof(cached.second));
    }
    return true;
}

static enum ggml_status ggml_backend_meta_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    buf_ctx->stc_compute_index = buf_ctx->stc_compute_index_next;
    return ggml_backend_meta_buffer_init_tensor_impl(buf_ctx->get_simple_tensor_container(tensor), tensor, /*allow_memo =*/ true);
}

// Recreate the shadow slices of a tensor (and, recursively, of its view source
// and sources) after they were evicted by the compute-container rotation.
// Registers into the current compute container; the tensor's data offsets are
// still valid because the underlying simple buffers never move.
// force skips the lookup and always rebuilds the tensor's own shadows: graph nodes
// arriving over RPC may have stale registrations made from bare INIT_TENSOR structs
// (no src links, possibly overwritten arena memory) that fingerprints cannot detect.
static struct ggml_tensor * ggml_backend_meta_buffer_ensure_simple_tensor(const struct ggml_tensor * tensor, size_t index, bool force = false) {
    if (!force) {
        ggml_tensor * t = ggml_backend_meta_buffer_simple_tensor(tensor, index);
        if (t != nullptr) {
            return t;
        }
    }

    if (tensor->view_src != nullptr && ggml_backend_buffer_is_meta(tensor->view_src->buffer)) {
        ggml_backend_meta_buffer_ensure_simple_tensor(tensor->view_src, index);
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i] != nullptr && tensor->src[i] != tensor && ggml_backend_buffer_is_meta(tensor->src[i]->buffer)) {
            ggml_backend_meta_buffer_ensure_simple_tensor(tensor->src[i], index);
        }
    }

    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    ggml_backend_meta_buffer_init_tensor_impl(buf_ctx->get_simple_tensor_container(tensor), (ggml_tensor *) tensor);

    return ggml_backend_meta_buffer_simple_tensor(tensor, index);
}

// a bare leaf struct (no view links) whose data address falls inside a registered static
// entry's range but failed identity adoption is an alias the meta layer cannot place -
// typically a view whose links were lost crossing RPC. Deriving a fallback (mirrored)
// state for it would read garbage and, worse, WRITE across the packed per-device slices
// of neighboring tensors, so fail loudly instead. The RPC client is responsible for
// resolving views to their root tensor before serializing (see rpc_resolve_view).
static void ggml_backend_meta_buffer_reject_unknown_alias(const struct ggml_tensor * tensor, const char * op) {
    if (tensor->view_src != nullptr) {
        return; // in-process views carry walkable links; derivation handles them
    }
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    const auto & by_data = buf_ctx->stc_static.by_data;
    auto it = by_data.upper_bound(tensor->data);
    if (it == by_data.begin()) {
        return;
    }
    --it;
    const ggml_backend_meta_tensor_fingerprint & fp = it->second->fingerprint;
    size_t entry_nbytes = 0;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        entry_nbytes = std::max(entry_nbytes, (size_t) fp.ne[i] * fp.nb[i]);
    }
    if ((const char *) tensor->data < (const char *) it->first + entry_nbytes) {
        GGML_ABORT("%s: tensor '%s' aliases registered island tensor '%s' but cannot be resolved - "
                   "views must be translated to their root tensor before crossing RPC", op, tensor->name, fp.name);
    }
}

static void ggml_backend_meta_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    // RPC commands arrive as freshly deserialized structs - resolve them by identity
    // first so no lookup or derivation ever walks their src pointers
    if (!ggml_backend_meta_buffer_adopt_identity(tensor)) {
        ggml_backend_meta_buffer_reject_unknown_alias(tensor, __func__);
    }
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_buffer_ensure_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_data,
                            simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,
                            row_count, simple_tensor->nb[1], tensor->nb[1]);
                        offset_data       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(offset_data*row_count == size);
            return;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t r = 0; r < split_state.nr[s]; r++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_ensure_simple_tensor(tensor, j);
                    const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_data,
                        simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,
                        row_count, simple_tensor->nb[2], tensor->nb[2]);
                    offset_data       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(offset_data*row_count == size);
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_ensure_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                const size_t simple_offset = i_start * chunk_size_j;
                ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_ensure_simple_tensor(tensor, j);
                ggml_backend_tensor_set(simple_tensor, data, offset, size);
            }
        } break;
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
            GGML_ASSERT(tensor->type == GGML_TYPE_F32);
            const int64_t ne = ggml_nelements(tensor);
            std::vector<float> tmp;
            tmp.reserve(ne);
            for (int64_t i = 0; i < ne; i++) {
                tmp.push_back(((const float *) data)[i] / n_bufs);
            }
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_ensure_simple_tensor(tensor, j);
                ggml_backend_tensor_set(simple_tensor, tmp.data(), offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    // RPC commands arrive as freshly deserialized structs - resolve them by identity
    // first so no lookup or derivation ever walks their src pointers
    if (!ggml_backend_meta_buffer_adopt_identity(tensor)) {
        ggml_backend_meta_buffer_reject_unknown_alias(tensor, __func__);
    }
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_ensure_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_data,
                            simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,
                            row_count, simple_tensor->nb[1], tensor->nb[1]);
                        offset_data       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(offset_data*row_count == size);
            return;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t r = 0; r < split_state.nr[s]; r++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_ensure_simple_tensor(tensor, j);
                    const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_data,
                        simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,
                        row_count, simple_tensor->nb[2], tensor->nb[2]);
                    offset_data       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(offset_data*row_count == size);
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_bufs; j++){
                const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_ensure_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                const size_t simple_offset = i_start * chunk_size_j;
                ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_ensure_simple_tensor(tensor, 0);
            ggml_backend_tensor_get(simple_tensor, data, offset, size);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
            // a partial tensor's logical value is the sum of the per-device slices
            // (the dual of the PARTIAL set_tensor case, which divides by n_bufs).
            // Mainly used by debug readers (e.g. eval-callback) inspecting nodes
            // before their AllReduce boundary.
            GGML_ASSERT(tensor->type == GGML_TYPE_F32);
            GGML_ASSERT(offset % sizeof(float) == 0 && size % sizeof(float) == 0);
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_ensure_simple_tensor(tensor, 0);
            ggml_backend_tensor_get(simple_tensor, data, offset, size);
            std::vector<float> tmp(size / sizeof(float));
            for (size_t j = 1; j < n_bufs; j++) {
                simple_tensor = ggml_backend_meta_buffer_ensure_simple_tensor(tensor, j);
                ggml_backend_tensor_get(simple_tensor, tmp.data(), offset, size);
                float * out = (float *) data;
                for (size_t k = 0; k < tmp.size(); k++) {
                    out[k] += tmp[k];
                }
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    const size_t n_buffers = ggml_backend_meta_buffer_n_bufs(buffer);
    for (size_t i = 0; i < n_buffers; i++) {
        ggml_backend_buffer_clear(ggml_backend_meta_buffer_simple_buffer(buffer, i), value);
    }
}

static void ggml_backend_meta_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    for (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        ggml_backend_buffer_reset(ggml_backend_meta_buffer_simple_buffer(buffer, i));
    }
}

// propagate to the member buffers: usage distinguishes weight-class from
// compute-class members for the surgical re-provision journal and guard
static void ggml_backend_meta_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    for (auto & b : buf_ctx->bufs) {
        ggml_backend_buffer_set_usage(b.get(), usage);
    }
}

static const ggml_backend_buffer_i ggml_backend_meta_buffer_iface = {
    /* .free_buffer     = */ ggml_backend_meta_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_meta_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_meta_buffer_init_tensor,
    /* .memset_tensor   = */ nullptr, // TODO implement
    /* .set_tensor      = */ ggml_backend_meta_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_meta_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_meta_buffer_clear,
    /* .reset           = */ ggml_backend_meta_buffer_reset,
    /* .set_usage       = */ ggml_backend_meta_buffer_set_usage,
};

bool ggml_backend_buffer_is_meta(ggml_backend_buffer_t buf) {
    return buf != nullptr && buf->iface.free_buffer == ggml_backend_meta_buffer_iface.free_buffer;
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    const ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*ggml_tensor_overhead(), // FIXME
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_backend_meta_simple_tensor_container stc_static;

    size_t max_size = 0;
    std::vector<ggml_backend_buffer_t> bufs;
    bufs.reserve(n_simple_bufts);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        bufs.push_back(ggml_backend_buft_alloc_buffer(ggml_backend_meta_buft_simple_buft(buft, i), size));
        if (bufs.back() == nullptr) {
            // a member alloc can fail at runtime (e.g. a dead RPC worker during a
            // sched re-reserve) - fail the allocation cleanly, the caller's error
            // path (decode failure -> reload/surgical recovery) handles it
            GGML_LOG_ERROR("%s: member %zu allocation of %zu bytes failed\n", __func__, i, size);
            for (ggml_backend_buffer_t b : bufs) {
                if (b != nullptr) {
                    ggml_backend_buffer_free(b);
                }
            }
            return nullptr;
        }
        max_size = std::max(max_size, ggml_backend_buffer_get_size(bufs.back()));
    }
    ggml_backend_meta_buffer_context * buf_ctx = new ggml_backend_meta_buffer_context(stc_static, params, n_simple_bufts, bufs);

    return ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, buf_ctx, max_size);
}

struct ggml_backend_buffer * ggml_backend_meta_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    constexpr size_t compute_headroom = 16; // Maximum number of views per statically allocated tensor that can be created between evals.
    // headroom so the static arena never retires mid-creation: the per-buft walk below
    // only sees the current arena, so shadows must not spill into a retired one
    const ggml_init_params params_static = {
        /*.mem_size   =*/ ggml_get_mem_size(ctx) + 4*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    const ggml_init_params params_compute = {
        /*.mem_size   =*/ compute_headroom*ggml_get_mem_size(ctx),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_backend_meta_simple_tensor_container stc_static   (params_static,  n_simple_bufts);

    std::vector<ggml_backend_buffer_t> bufs(n_simple_bufts, nullptr);
    ggml_backend_meta_buffer_context * meta_buf_ctx = new ggml_backend_meta_buffer_context(stc_static, params_compute, n_simple_bufts, bufs);

    ggml_backend_buffer_t meta_buf = ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, meta_buf_ctx, 0);
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        t->buffer = meta_buf;
        ggml_backend_meta_buffer_init_tensor_impl(meta_buf_ctx->stc_static, t);
        t->data = (void *) 0x2000000000000000; // FIXME
    }
    for (size_t i = 0; i < n_simple_bufts; i++) {
        ggml_context * ctx = meta_buf_ctx->stc_static.ctxs[i].get();
        ggml_backend_buffer_type_t simple_buft = ggml_backend_meta_buft_simple_buft(buft, i);

        // If a ggml_context has nothing to allocate (only zero-sized tensors, views or
        // pre-allocated tensors), ggml_backend_alloc_ctx_tensors_from_buft returns NULL.
        // For those edge cases, allocate a dummy buffer instead.
        bool any_allocatable = false;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (ggml_nelements(t) != 0 && t->data == nullptr && t->view_src == nullptr) {
                any_allocatable = true;
                break;
            }
        }
        if (any_allocatable) {
            meta_buf_ctx->bufs[i].reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, simple_buft));
        } else {
            meta_buf_ctx->bufs[i].reset(ggml_backend_buft_alloc_buffer(simple_buft, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                t->buffer = meta_buf_ctx->bufs[i].get();
            }
        }
        if (!meta_buf_ctx->bufs[i]) {
            // a member's allocation failure (e.g. an OOM-killed RPC worker) fails the
            // load cleanly instead of aborting the process
            GGML_LOG_ERROR("%s: failed to allocate member %zu buffer (%s)\n", __func__, i, ggml_backend_buft_name(simple_buft));
            ggml_backend_buffer_free(meta_buf);
            return nullptr;
        }
        meta_buf->size = std::max(meta_buf->size, ggml_backend_buffer_get_size(meta_buf_ctx->bufs[i].get()));
    }
    return meta_buf;
}

//
// meta backend
//

static ggml_guid_t ggml_backend_meta_guid() {
    static ggml_guid guid = {0xf1, 0x0e, 0x34, 0xcf, 0x9c, 0x6f, 0x43, 0xcb, 0x96, 0x92, 0xbe, 0x8e, 0xbb, 0x71, 0x3f, 0xda};
    return &guid;
}

struct ggml_backend_meta_context {
    struct cgraph_config {
        ggml_cgraph * cgraph_main = nullptr;
        int           offset      = 0; // Node offset vs. original graph
        // whether this subgraph's boundary was created by a PARTIAL node (its last
        // node needs an AllReduce). The LAST subgraph of a graph can also end
        // PARTIAL - e.g. when the scheduler splits a model graph right after a
        // partial node, or under per-node debug execution - and skipping its
        // reduce silently corrupts every consumer of that tensor.
        bool          reduce      = false;
    };
    struct backend_config {
        ggml_backend_t backend;

        std::vector<cgraph_config>           cgraphs;
        std::vector<ggml_tensor *>           nodes;
        std::vector<ggml_backend_buffer_ptr> bufs;

        backend_config(ggml_backend_t backend, const size_t n_reduce_steps) : backend(backend) {
            bufs.resize(n_reduce_steps);
        }
    };
    // one fully prepared subgraph decomposition of an outer graph. The scheduler
    // hands the meta backend the same few graph pieces every token but with a
    // FRESH uid per split rebuild, so uid equality alone can never recognize
    // them - builds are cached under a content hash instead. The active build's
    // per-member state lives in backend_configs (so the compute path stays
    // unchanged); an inactive build parks it here.
    struct meta_build {
        uint64_t key       = 0; // content hash of the outer graph
        uint64_t outer_uid = 0; // outer cgraph->uid at build time (uid fast path)
        uint64_t built_seq = 0; // rebuild counter at creation (shadow-ring validity)
        uint64_t used_seq  = 0; // LRU
        size_t   n_subgraphs = 0;
        ggml_context_ptr ctx;   // owns this build's cgraph_main structures
        std::vector<std::vector<cgraph_config>> cgraphs; // [n_backends][...]
        std::vector<std::vector<ggml_tensor *>> nodes;   // [n_backends][n_nodes]
    };
    std::string                 name;
    std::vector<backend_config> backend_configs;
    ggml_context_ptr            ctx;
    std::vector<ggml_cgraph *>  cgraphs_aux;
    std::vector<ggml_tensor *>  nodes_aux;
    size_t                      n_reduce_steps;
    size_t                      max_tmp_size  = 0;
    size_t                      max_subgraphs = 0;
    size_t                      n_subgraphs   = 0;

    std::vector<std::unique_ptr<meta_build>> builds;
    meta_build * build_active = nullptr;
    uint64_t     rebuild_seq  = 0;
    uint64_t     use_seq      = 0;
    // compute-node shadows of a build live in the stc ring slot that was current
    // when it was built; every rebuild advances each used buffer's ring by one
    // slot, so a build older than ring_slots-1 rebuilds may point at reset
    // shadows and must be discarded. Mirrors the buffer context's ring sizing.
    int          ring_slots;

    // GGML_META_TIMING accumulators (compute vs reduce-boundary attribution)
    int64_t tm_compute_us = 0;
    int64_t tm_reduce_us  = 0;
    int64_t tm_reduces    = 0;
    int64_t tm_graphs     = 0;
    int64_t tm_build_hits   = 0;
    int64_t tm_build_misses = 0;

    void *                               comm_ctx       = nullptr;
    ggml_backend_comm_allreduce_tensor_t comm_allreduce = nullptr;
    ggml_backend_cpy_tensor_batch_async_t cpy_batch     = nullptr;
    ggml_backend_get_tensor_batch_t       get_batch     = nullptr;
    ggml_backend_boundary_fused_send_t    fused_send    = nullptr;
    ggml_backend_boundary_fused_recv_t    fused_recv    = nullptr;

    // members whose reg provides the batched-get proc reach their data over a wire
    // (RPC-class); a member without it is local and can root a star reduce
    std::vector<bool>    wire_member;
    std::vector<uint8_t> star_scratch; // host staging for star-reduce partials

    ggml_backend_meta_context(ggml_backend_dev_t meta_dev, const char * params) {
        const size_t n_devs = ggml_backend_meta_dev_n_devs(meta_dev);
        n_reduce_steps = std::ceil(std::log2(n_devs));
        // must match the stc_compute ring sizing in ggml_backend_meta_buffer_context
        const char * GGML_META_MAX_GRAPHS = getenv("GGML_META_MAX_GRAPHS");
        ring_slots = std::max(2, GGML_META_MAX_GRAPHS ? atoi(GGML_META_MAX_GRAPHS) : 8);
        name = "Meta(";
        std::vector<ggml_backend_t> simple_backends;
        backend_configs.reserve(n_devs);
        simple_backends.reserve(n_devs);
        for (size_t i = 0; i < n_devs; i++) {
            ggml_backend_dev_t simple_dev = ggml_backend_meta_dev_simple_dev(meta_dev, i);
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_dev_name(simple_dev);
            simple_backends.push_back(ggml_backend_dev_init(simple_dev, params));
            backend_configs.emplace_back(simple_backends.back(), n_reduce_steps);
        }
        name += ")";

        if (n_devs > 1) {
            ggml_backend_comm_init_t comm_init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_init");
            if (comm_init != nullptr) {
                comm_ctx = comm_init(simple_backends.data(), simple_backends.size());
            }
            // batch procs live on the RPC reg, which is not necessarily member 0
            // (e.g. Meta(CUDA0,RPC,RPC)): take them from the first member that has them
            wire_member.resize(n_devs, false);
            for (size_t i = 0; i < n_devs; i++) {
                ggml_backend_reg_t reg_i = ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[i]));
                void * cb = ggml_backend_reg_get_proc_address(reg_i, "ggml_backend_cpy_tensor_batch_async");
                void * gb = ggml_backend_reg_get_proc_address(reg_i, "ggml_backend_get_tensor_batch");
                wire_member[i] = gb != nullptr;
                if (cpy_batch == nullptr) {
                    cpy_batch = (ggml_backend_cpy_tensor_batch_async_t) cb;
                }
                if (get_batch == nullptr) {
                    get_batch = (ggml_backend_get_tensor_batch_t) gb;
                }
                if (fused_send == nullptr) {
                    fused_send = (ggml_backend_boundary_fused_send_t)
                        ggml_backend_reg_get_proc_address(reg_i, "ggml_backend_boundary_fused_send");
                }
                if (fused_recv == nullptr) {
                    fused_recv = (ggml_backend_boundary_fused_recv_t)
                        ggml_backend_reg_get_proc_address(reg_i, "ggml_backend_boundary_fused_recv");
                }
            }
        }
        if (comm_ctx != nullptr) {
            comm_allreduce = (ggml_backend_comm_allreduce_tensor_t)
                ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(
                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_allreduce_tensor");
            GGML_ASSERT(comm_allreduce != nullptr);
        }
    }

    ~ggml_backend_meta_context() {
        if (comm_ctx != nullptr) {
            ggml_backend_comm_free_t comm_free = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_configs[0].backend)), "ggml_backend_comm_free");
            GGML_ASSERT(comm_free != nullptr);
            comm_free(comm_ctx);
        }
        for (auto & bc : backend_configs) {
            ggml_backend_free(bc.backend);
        }
    }
};

static const char * ggml_backend_meta_get_name(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) backend->context;
    return backend_ctx->name.c_str();
}

static void ggml_backend_meta_free(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    delete backend_ctx;
    delete backend;
}

static void ggml_backend_meta_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(split_state.n_segments == 1);
    GGML_ASSERT(split_state.nr[0]      == 1);

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_ensure_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                ggml_backend_tensor_set_2d_async(simple_backend, simple_tensor, (const char *) data + offset_j, offset, chunk_size_j,
                    i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_backends; j++) {
                ggml_backend_tensor_set_async(
                    ggml_backend_meta_simple_backend(backend, j), ggml_backend_meta_buffer_ensure_simple_tensor(tensor, j), data, offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(split_state.n_segments == 1);
    GGML_ASSERT(split_state.nr[0]      == 1);

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_ensure_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                ggml_backend_tensor_get_2d_async(simple_backend, simple_tensor, (char *) data + offset_j, offset, chunk_size_j,
                    i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, 0);
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_ensure_simple_tensor(tensor, 0);
            ggml_backend_tensor_get_async(simple_backend, simple_tensor, data, offset, size);
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_synchronize(ggml_backend_t backend) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    for (size_t i = 0; i < n_backends; i++) {
        ggml_backend_synchronize(ggml_backend_meta_simple_backend(backend, i));
    }
}

// FNV-1a over every field that determines what a subgraph computes. Struct
// addresses are deliberately excluded: shadow structs are recreated per rebuild,
// but if ops, shapes, params, flags and data/src addresses all match, executing
// the previously transmitted graph is byte-identical.
static uint64_t ggml_backend_meta_subgraph_uid(const ggml_cgraph * g) {
    uint64_t h = 1469598103934665603ULL;
    auto mix = [&h](const void * data, size_t n) {
        const uint8_t * b = (const uint8_t *) data;
        for (size_t k = 0; k < n; k++) {
            h = (h ^ b[k]) * 1099511628211ULL;
        }
    };
    auto mix_tensor = [&](const ggml_tensor * t) {
        if (t == nullptr) {
            const int none = -1;
            mix(&none, sizeof(none));
            return;
        }
        mix(&t->type, sizeof(t->type));
        mix(t->ne, sizeof(t->ne));
        mix(t->nb, sizeof(t->nb));
        mix(&t->data, sizeof(t->data));
        mix(&t->view_offs, sizeof(t->view_offs));
    };
    for (int k = 0; k < g->n_nodes; k++) {
        const ggml_tensor * t = g->nodes[k];
        mix(&t->op, sizeof(t->op));
        mix(t->op_params, sizeof(t->op_params));
        mix(&t->flags, sizeof(t->flags));
        mix_tensor(t);
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            mix_tensor(t->src[s]);
        }
        mix_tensor(t->view_src);
    }
    return h == 0 ? 1 : h;
}

// build-cache key for an INCOMING outer graph: same fields as
// ggml_backend_meta_subgraph_uid but mixed word-wise - this runs on every
// graph_compute call (scheduler splits arrive with a fresh uid each token, so
// content is the only stable identity), the byte-wise version only per rebuild.
// The value never crosses a protocol boundary.
static uint64_t ggml_backend_meta_outer_graph_key(const ggml_cgraph * g) {
    uint64_t h = 1469598103934665603ULL;
    auto mix = [&h](const void * data, size_t n) {
        const uint8_t * b = (const uint8_t *) data;
        while (n >= sizeof(uint64_t)) {
            uint64_t w;
            memcpy(&w, b, sizeof(w));
            h = (h ^ w) * 1099511628211ULL;
            b += sizeof(w);
            n -= sizeof(w);
        }
        uint64_t tail = 0;
        if (n > 0) {
            memcpy(&tail, b, n);
            h = (h ^ tail) * 1099511628211ULL;
        }
    };
    auto mix_tensor = [&](const ggml_tensor * t) {
        if (t == nullptr) {
            const uint64_t none = ~0ULL;
            mix(&none, sizeof(none));
            return;
        }
        mix(&t->type, sizeof(t->type));
        mix(t->ne, sizeof(t->ne));
        mix(t->nb, sizeof(t->nb));
        mix(&t->data, sizeof(t->data));
        mix(&t->view_offs, sizeof(t->view_offs));
        if (t->buffer == nullptr || !ggml_backend_buffer_is_meta(t->buffer)) {
            // non-meta structs (host views at piece boundaries and their view
            // sources) enter the prepared member graphs as raw pointers, so a
            // build is only reusable while these exact structs are alive - key
            // them by identity, not just content
            mix(&t, sizeof(t));
        }
    };
    mix(&g->n_nodes, sizeof(g->n_nodes));
    for (int k = 0; k < g->n_nodes; k++) {
        const ggml_tensor * t = g->nodes[k];
        mix(&t->op, sizeof(t->op));
        mix(t->op_params, sizeof(t->op_params));
        mix(&t->flags, sizeof(t->flags));
        mix_tensor(t);
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            mix_tensor(t->src[s]);
        }
        mix_tensor(t->view_src);
    }
    return h == 0 ? 1 : h;
}

static enum ggml_status ggml_backend_meta_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(cgraph->grads == nullptr);
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;

    // park the active build's per-member state back into its cache entry so a
    // different build can be activated (or a fresh one written) in its place
    auto park_active = [&]() {
        ggml_backend_meta_context::meta_build * a = backend_ctx->build_active;
        if (a == nullptr) {
            return;
        }
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            a->cgraphs[j].swap(bcj.cgraphs);
            a->nodes[j].swap(bcj.nodes);
        }
        backend_ctx->build_active = nullptr;
    };

    // look up a prepared build: by outer uid when the graph carries one, else by
    // content hash. A build older than ring_slots-1 rebuilds may reference reset
    // shadow-ring slots and is dropped instead of trusted.
    backend_ctx->use_seq++;
    ggml_backend_meta_context::meta_build * build = nullptr;
    uint64_t outer_key = 0;
    for (size_t k = 0; k < backend_ctx->builds.size(); ) {
        ggml_backend_meta_context::meta_build * b = backend_ctx->builds[k].get();
        if (backend_ctx->rebuild_seq - b->built_seq >= (uint64_t) (backend_ctx->ring_slots - 1)) {
            if (b == backend_ctx->build_active) {
                park_active();
            }
            backend_ctx->builds.erase(backend_ctx->builds.begin() + k);
            continue;
        }
        k++;
    }
    if (cgraph->uid != 0) {
        for (auto & b : backend_ctx->builds) {
            if (b->outer_uid == cgraph->uid) {
                build = b.get();
                break;
            }
        }
    }
    if (build == nullptr) {
        outer_key = ggml_backend_meta_outer_graph_key(cgraph);
        for (auto & b : backend_ctx->builds) {
            if (b->key == outer_key) {
                build = b.get();
                break;
            }
        }
    }

    const bool needs_rebuild = build == nullptr;
    if (build != nullptr) {
        backend_ctx->tm_build_hits++;
        build->used_seq  = backend_ctx->use_seq;
        build->outer_uid = cgraph->uid; // the scheduler renames pieces every split rebuild
        if (build != backend_ctx->build_active) {
            park_active();
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                build->cgraphs[j].swap(bcj.cgraphs);
                build->nodes[j].swap(bcj.nodes);
            }
            backend_ctx->build_active = build;
        }
        backend_ctx->n_subgraphs = build->n_subgraphs;
    } else {
        backend_ctx->tm_build_misses++;
        backend_ctx->rebuild_seq++;
        park_active();
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            bcj.nodes.assign(cgraph->n_nodes, nullptr);
            bcj.cgraphs.assign(cgraph->n_nodes, {});
        }
    }

    if (needs_rebuild) {
        std::set<ggml_backend_buffer_t> used_buffers;
        for (int i = 0; i < cgraph->n_leafs; i++) {
            if (ggml_backend_buffer_is_meta(cgraph->leafs[i]->buffer)) {
                used_buffers.emplace(cgraph->leafs[i]->buffer);
            }
        }
        for (int i = 0; i < cgraph->n_nodes; i++) {
            if (ggml_backend_buffer_is_meta(cgraph->nodes[i]->buffer)) {
                used_buffers.emplace(cgraph->nodes[i]->buffer);
            }
        }
        for (ggml_backend_buffer_t buf : used_buffers) {
            ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buf->context;
            buf_ctx->stc_compute_index_next = (buf_ctx->stc_compute_index + 1) % (int) buf_ctx->stc_compute.size();
            ggml_backend_meta_simple_tensor_container & stc = buf_ctx->stc_compute[buf_ctx->stc_compute_index_next];
            for (ggml_context_ptr & ctx : stc.ctxs) {
                ggml_reset(ctx.get());
            }
            stc.ctxs_retired.clear(); // frees overflow arenas chained by shadow_ctx
            stc.simple_tensors.clear();
            stc.by_data.clear();
        }
        size_t n_subgraphs  = 0;
        size_t max_tmp_size = 0;

        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];

            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (node->view_src != nullptr && node->view_src->op == GGML_OP_NONE &&
                        (node->view_src->buffer == nullptr /* e.g. coordinator-side CPU tensors over RPC */ ||
                         ggml_backend_buffer_is_host(node->view_src->buffer))) {
                    // FIXME s_copy_main is on the CPU and its view seems to be incorrectly added to the graph nodes.
                    // For regular usage this doesn't matter since it's a noop but trying to call ggml_backend_meta_buffer_simple_tensor results in a crash.
                    bcj.nodes[i] = node;
                    continue;
                }
                if (j == 0) {
                    // always rebuild node shadows from the graph itself (once per node): earlier
                    // registrations may be stale - evicted by ring rotation for llama-side cached
                    // graphs, or made from bare INIT_TENSOR structs over RPC whose src links are
                    // missing and whose arena was overwritten by later commands. Leaves (weights,
                    // caches, inputs) keep their upload-time registrations.
                    bcj.nodes[i] = ggml_backend_meta_buffer_ensure_simple_tensor(node, j, /*force =*/ true);
                } else {
                    bcj.nodes[i] = ggml_backend_meta_buffer_simple_tensor(node, j);
                }
                GGML_ASSERT(bcj.nodes[i]);
            }
        }

        {
            // For MoE models it may make sense to delay the AllReduce in order to reduce I/O:
            auto get_i_delayed = [&](const int i) -> int {
                // GGML_META_NO_DELAY=1: reduce at every PARTIAL node (diagnostic - isolates
                // the delay pattern matcher when hunting numerical corruption)
                static const bool no_delay = getenv("GGML_META_NO_DELAY") != nullptr;
                if (no_delay) {
                    return i;
                }
                int id = i; // i_delayed
                int idr = i; // i_delayed return, last safe return value

                ggml_tensor * node = cgraph->nodes[id];
                int32_t n_used = ggml_node_get_use_count(cgraph, id);

                // Skip MIRRORED nodes that don't consume node
                auto skip_unrelated = [&]() {
                    while (id + 1 < cgraph->n_nodes) {
                        ggml_tensor * next = cgraph->nodes[id+1];
                        if (ggml_backend_meta_get_split_state(next, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                            break;
                        }
                        bool safe = true;
                        for (int s = 0; s < GGML_MAX_SRC; s++) {
                            if (next->src[s] == nullptr) {
                                continue;
                            }
                            if (next->src[s] == node) {
                                safe = false;
                                break;
                            }
                            if (ggml_backend_meta_get_split_state(next->src[s], false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                                safe = false;
                                break;
                            }
                        }
                        if (!safe) {
                            break;
                        }
                        id++;
                    }
                };

                skip_unrelated();
                if (id + 1 >= cgraph->n_nodes) {
                    return idr;
                }
                {
                    // delaying past a consumer is only sound when it is the SOLE
                    // consumer: any other user of the partial node would read the
                    // UNREDUCED values (hyper-connection models feed the attention
                    // output to several residual streams - found as gradual
                    // member drift once dedicated attention made attn_out PARTIAL)
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (n_used == 1 && next->op == GGML_OP_ADD_ID && next->src[0] == node &&
                            ggml_backend_meta_get_split_state(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL &&
                            ggml_backend_meta_get_split_state(next->src[2], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        node = next;
                        id++;
                        idr = id;
                        n_used = ggml_node_get_use_count(cgraph, id);
                    }
                }
                // Chain of MULs with MIRRORED src[1]
                while (true) {
                    skip_unrelated();
                    if (id + 1 >= cgraph->n_nodes) {
                        return idr;
                    }
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (n_used == 1 && next->op == GGML_OP_MUL && next->src[0] == node &&
                            ggml_backend_meta_get_split_state(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        node = next;
                        id++;
                        idr = id;
                        n_used = ggml_node_get_use_count(cgraph, id);
                    } else {
                        break;
                    }
                }

                if (n_used != node->ne[1] || id + 2*n_used-1 >= cgraph->n_nodes) {
                    return idr;
                }
                for (int32_t k = 0; k < n_used; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_VIEW || next->view_src != node || next->view_offs != k*node->nb[1] ||
                            next->ne[0] != node->ne[0] || next->ne[1] != node->ne[2] || next->nb[1] != node->nb[2] ||
                            ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id - (n_used-1)] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                for (int32_t k = 0; k < n_used - 2; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                idr = id;
                return idr;
            };

            int i_start = 0;
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                const bool host_view = node->view_src != nullptr && node->view_src->op == GGML_OP_NONE &&
                        (node->view_src->buffer == nullptr /* e.g. coordinator-side CPU tensors over RPC */ ||
                         ggml_backend_buffer_is_host(node->view_src->buffer));
                // a host view has no meta split state, but if it is the LAST node the
                // trailing subgraph still has to be closed (scheduler splits can end
                // on one - seen with the deepseek4 indexer handoff)
                if (host_view && i + 1 < cgraph->n_nodes) {
                    continue;
                }
                ggml_backend_meta_split_state split_state;
                split_state.axis = GGML_BACKEND_SPLIT_AXIS_NONE;
                if (!host_view) {
                    split_state = ggml_backend_meta_get_split_state(node, /*assume_sync =*/ false);
                }
                if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                    max_tmp_size = std::max(max_tmp_size, ggml_nbytes(node));
                }
                const bool new_subgraph = i + 1 == cgraph->n_nodes || split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL;
                if (!new_subgraph) {
                    continue;
                }

                const int i_delayed = get_i_delayed(i);

                // GGML_META_DEBUG_REDUCE=1: print where each AllReduce boundary lands.
                // The placement logic is independent of n_devices, so a 1-GPU meta run
                // shows exactly which nodes sit inside a delayed partial window.
                static const bool debug_reduce = getenv("GGML_META_DEBUG_REDUCE") != nullptr;
                if (debug_reduce) {
                    fprintf(stderr, "REDUCE: partial %4d '%s'[%s] -> boundary %4d '%s'[%s]\n",
                            i, node->name, ggml_op_name(node->op),
                            i_delayed, cgraph->nodes[i_delayed]->name, ggml_op_name(cgraph->nodes[i_delayed]->op));
                }

                // If we can delay the AllReduce we need to consider the interaction with zero-sized tensor slices.
                // A backend with such a slice would normally have valid data after participating in the AllReduce with a node that has
                //     its compute flag disabled and thus gets its data zeroed out.
                // If the AllReduce is delayed then the nodes until that point also need to have their compute flag disabled.
                if (i_delayed > i) {
                    for (size_t j = 0; j < n_backends; j++) {
                        auto & bcj = backend_ctx->backend_configs[j];
                        if ((bcj.nodes[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                            for (int ii = i + 1; ii <= i_delayed; ii++) {
                                bcj.nodes[ii]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                            }
                        }
                    }
                }

                i = i_delayed;

                for (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    bcj.cgraphs[n_subgraphs].offset = i_start;
                    // reduce iff this boundary exists because of a PARTIAL node -
                    // this includes a PARTIAL node that happens to be the LAST node
                    // of the graph (scheduler split or per-node debug execution),
                    // which the old position-based check silently skipped
                    bcj.cgraphs[n_subgraphs].reduce = split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL;
                }
                n_subgraphs++;
                i_start = i + 1;
            }
            GGML_ASSERT(i_start == cgraph->n_nodes);
        }

        backend_ctx->n_subgraphs = n_subgraphs;

        if (max_tmp_size > backend_ctx->max_tmp_size) {
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                for (size_t i = 0; i < backend_ctx->n_reduce_steps; i++) {
                    bcj.bufs[i].reset(ggml_backend_alloc_buffer(bcj.backend, max_tmp_size));
                }
            }
            backend_ctx->max_tmp_size = max_tmp_size;
        }

        // aux graphs/nodes are shared scratch, rewritten on every compute call -
        // grow them to the largest decomposition seen so far
        if (n_subgraphs > backend_ctx->max_subgraphs) {
            backend_ctx->max_subgraphs = n_subgraphs;
            const size_t n_nodes_per_device = 3 * backend_ctx->n_reduce_steps; // tmp + ADD (+zeroing) graph per step and device
            const size_t n_cgraphs_per_device = 2 * backend_ctx->n_reduce_steps; // ADD ( + zeroing) graph per step and device
            const size_t mem_per_device_graphs_aux = n_cgraphs_per_device*backend_ctx->max_subgraphs*ggml_graph_overhead_custom(1, cgraph->grads);
            const size_t mem_per_device_nodes_aux = n_nodes_per_device*backend_ctx->max_subgraphs*ggml_tensor_overhead();
            const ggml_init_params params = {
                /*.mem_size   =*/ n_backends * (mem_per_device_graphs_aux + mem_per_device_nodes_aux),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            backend_ctx->ctx.reset(ggml_init(params));
            backend_ctx->cgraphs_aux.resize(n_backends*n_cgraphs_per_device*backend_ctx->max_subgraphs);
            for (size_t k = 0; k < backend_ctx->cgraphs_aux.size(); k++) {
                backend_ctx->cgraphs_aux[k] = ggml_new_graph_custom(backend_ctx->ctx.get(), 1, cgraph->grads);
            }
            backend_ctx->nodes_aux.resize(n_backends*n_nodes_per_device*backend_ctx->max_subgraphs);
            for (size_t k = 0; k < backend_ctx->nodes_aux.size(); k++) {
                backend_ctx->nodes_aux[k] = ggml_new_tensor_1d(backend_ctx->ctx.get(), GGML_TYPE_F32, 1);
            }
        }

        // the build owns its subgraph structures: exact-size its arena from the
        // offsets the split walk just produced
        ggml_context_ptr build_ctx;
        {
            size_t mem = 0;
            const auto & cfg0 = backend_ctx->backend_configs[0].cgraphs;
            for (size_t i = 0; i < n_subgraphs; i++) {
                const size_t i_node_start = cfg0[i].offset;
                const size_t i_node_stop = i + 1 < n_subgraphs ? cfg0[i + 1].offset : cgraph->n_nodes;
                mem += ggml_graph_overhead_custom(i_node_stop - i_node_start, cgraph->grads);
            }
            const ggml_init_params params = {
                /*.mem_size   =*/ n_backends * mem,
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            build_ctx.reset(ggml_init(params));
            GGML_ASSERT(build_ctx != nullptr);
        }

        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            for (size_t i_graph = 0; i_graph < n_subgraphs; i_graph++) {
                const size_t i_node_start = bcj.cgraphs[i_graph].offset;
                const size_t i_node_stop = i_graph + 1 < n_subgraphs ? bcj.cgraphs[i_graph + 1].offset : cgraph->n_nodes;
                ggml_cgraph * cgraph_ij = ggml_new_graph_custom(build_ctx.get(), i_node_stop - i_node_start, /*grads =*/ false);
                bcj.cgraphs[i_graph].cgraph_main = cgraph_ij;
                cgraph_ij->n_nodes = i_node_stop - i_node_start;
                ggml_hash_set_reset(&cgraph_ij->visited_hash_set);
                for (size_t i_node = i_node_start; i_node < i_node_stop; i_node++) {
                    ggml_tensor * node_ij = bcj.nodes[i_node];
                    cgraph_ij->nodes[i_node - i_node_start] = node_ij;
                    const size_t hash_pos_orig = ggml_hash_find(&cgraph->visited_hash_set, cgraph->nodes[i_node]);
                    const size_t hash_pos_ij = ggml_hash_insert(&cgraph_ij->visited_hash_set, node_ij);
                    cgraph_ij->use_counts[hash_pos_ij] = cgraph->use_counts[hash_pos_orig];
                }
                // content-derived uid: equal hashes mean the server-side execution is
                // byte-identical (ops, shapes, params, flags, data and src addresses),
                // so rebuilds keep the SAME uid for unchanged subgraphs and the RPC
                // graph cache stays hot even when the scheduler re-splits the outer
                // graph every token (deepseek4's multi-split graphs did exactly that:
                // fresh uids each token = a full re-serialization per subgraph)
                cgraph_ij->uid = ggml_backend_meta_subgraph_uid(cgraph_ij);
            }
        }

        // register the build; its per-member state stays live in backend_configs
        // until another build is activated. Entries past the shadow-ring horizon
        // were dropped during lookup, cap the rest by LRU.
        while (backend_ctx->builds.size() >= (size_t) (backend_ctx->ring_slots - 1)) {
            size_t lru = 0;
            for (size_t k = 1; k < backend_ctx->builds.size(); k++) {
                if (backend_ctx->builds[k]->used_seq < backend_ctx->builds[lru]->used_seq) {
                    lru = k;
                }
            }
            backend_ctx->builds.erase(backend_ctx->builds.begin() + lru);
        }
        auto build_new = std::make_unique<ggml_backend_meta_context::meta_build>();
        build_new->key         = outer_key;
        build_new->outer_uid   = cgraph->uid;
        build_new->built_seq   = backend_ctx->rebuild_seq;
        build_new->used_seq    = backend_ctx->use_seq;
        build_new->n_subgraphs = n_subgraphs;
        build_new->ctx         = std::move(build_ctx);
        build_new->cgraphs.resize(n_backends);
        build_new->nodes.resize(n_backends);
        backend_ctx->builds.push_back(std::move(build_new));
        backend_ctx->build_active = backend_ctx->builds.back().get();
    }

    size_t iga = 0; // i graph aux
    size_t ina = 0; // i node aux

    auto get_node_aux = [&](ggml_tensor * t) -> ggml_tensor * {
        ggml_tensor * ret = backend_ctx->nodes_aux[ina++];
        memset(ret, 0, sizeof(ggml_tensor));
        ret->op   = GGML_OP_NONE;
        ret->type = t->type;
        for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
            ret->ne[k] = t->ne[k];
            ret->nb[k] = t->nb[k];
        }
        return ret;
    };
    auto set_tmp_data = [&](ggml_tensor * tensor, const size_t j, const size_t i_buf) {
        auto & bcj = backend_ctx->backend_configs[j];
        ggml_backend_buffer_ptr & buf_ptr = bcj.bufs[i_buf];
        if (!buf_ptr || ggml_backend_buffer_get_size(buf_ptr.get()) < backend_ctx->max_tmp_size) {
            buf_ptr.reset(ggml_backend_alloc_buffer(bcj.backend, backend_ctx->max_tmp_size));
        }
        tensor->buffer = buf_ptr.get();
        tensor->data   = ggml_backend_buffer_get_base(buf_ptr.get());
    };
    // FIXME usage_counts
    auto get_cgraph_aux = [&]() -> ggml_cgraph * {
        ggml_cgraph * ret = backend_ctx->cgraphs_aux[iga++];
        return ret;
    };

    // GGML_META_TIMING is declared up here because it gates the fused pipeline below
    static const bool tm_enabled = getenv("GGML_META_TIMING") != nullptr;

    // fused boundary pipeline (RPC proto 4.5): at a star boundary each wire member's
    // NEXT subgraph - and the request for its next partial - rides in the same
    // message as this boundary's reduced value, collapsing the per-boundary exchange
    // to one round trip. Entry is the first star boundary of a piece (plain gather),
    // exit is the last subgraph (plain distribute); the pipeline never spans meta
    // graph_compute calls. Disabled in timing mode: tm_sync_all pings members while
    // a fused response is pending, which would desync the stream.
    static const bool no_fused = getenv("GGML_META_NO_FUSED") != nullptr;
    // requires the star reduce: a carried fetch is only ever consumed by the star
    // gather, so with GGML_META_NO_STAR the pipeline must stay off too
    const bool fused_enabled = !no_fused && !tm_enabled &&
        getenv("GGML_META_NO_STAR") == nullptr &&
        backend_ctx->fused_send != nullptr && backend_ctx->fused_recv != nullptr;
    std::vector<int>  fused_carried(n_backends, 0);       // member's next N subgraph dispatches already sent
    std::vector<char> fused_fetch_pending(n_backends, 0); // member's boundary-i partial arrives via fused_recv

    // Preferentially use backend-specific allreduce_tensor_async (e.g. NCCL for CUDA), use a generic fallback if unavailable:
    auto allreduce_fallback = [&](size_t i) -> ggml_status {
        std::vector<ggml_cgraph *> step_cgraphs(n_backends, nullptr);

        auto boundary_node = [&](size_t j) -> ggml_tensor * {
            ggml_cgraph * cg = backend_ctx->backend_configs[j].cgraphs[i].cgraph_main;
            return cg->nodes[cg->n_nodes - 1];
        };

        // push `value` into every member's boundary node (except j_skip); a wire
        // member's message also carries the CHAIN of its subgraphs up to and
        // including the next reduce subgraph plus - when that boundary is
        // star-eligible - the request for its partial, so the next reduce cycle
        // needs no further coordinator round trip (fused pipeline, proto 4.5)
        auto fused_distribute = [&](size_t j_skip, const void * value, size_t nbytes_v, bool allow_chain = true) {
            const size_t i_next    = i + 1;
            const bool   have_next = allow_chain && fused_enabled && i_next < backend_ctx->n_subgraphs;
            size_t i_r = i_next; // first reduce subgraph at or after i_next
            while (i_r < backend_ctx->n_subgraphs &&
                   !backend_ctx->backend_configs[0].cgraphs[i_r].reduce) {
                i_r++;
            }
            const bool   chain_full = have_next && i_r < backend_ctx->n_subgraphs;
            const size_t chain_end  = chain_full ? i_r : backend_ctx->n_subgraphs - 1;
            auto next_node = [&](size_t j) -> ggml_tensor * {
                ggml_cgraph * cg = backend_ctx->backend_configs[j].cgraphs[i_r].cgraph_main;
                return cg->nodes[cg->n_nodes - 1];
            };
            bool   next_star   = false;
            size_t next_nbytes = 0;
            if (chain_full) {
                size_t n_comp = 0;
                bool   ok2    = true;
                for (size_t j = 0; j < n_backends && ok2; j++) {
                    ggml_tensor * node = next_node(j);
                    if (node->flags & GGML_TENSOR_FLAG_COMPUTE) {
                        n_comp++;
                        ok2 = node->type == GGML_TYPE_F32 && ggml_is_contiguous(node);
                    }
                }
                next_star = ok2 && n_comp >= 2;
                if (next_star) {
                    next_nbytes = ggml_nbytes(next_node(0));
                }
            }
            std::vector<ggml_cgraph *> chain;
            for (size_t j = 0; j < n_backends; j++) {
                if (j == j_skip) {
                    continue;
                }
                auto & bcj = backend_ctx->backend_configs[j];
                if (have_next && backend_ctx->wire_member[j]) {
                    chain.clear();
                    for (size_t g = i_next; g <= chain_end; g++) {
                        chain.push_back(bcj.cgraphs[g].cgraph_main);
                    }
                    const bool want_fetch = next_star && (next_node(j)->flags & GGML_TENSOR_FLAG_COMPUTE);
                    if (backend_ctx->fused_send(bcj.backend, boundary_node(j), value, nbytes_v,
                            chain.data(), (int) chain.size(),
                            want_fetch ? next_node(j) : nullptr, want_fetch ? next_nbytes : 0)) {
                        fused_carried[j]       = (int) chain.size();
                        fused_fetch_pending[j] = want_fetch ? 1 : 0;
                        continue;
                    }
                }
                ggml_backend_tensor_set(boundary_node(j), value, 0, nbytes_v);
            }
        };

        // members whose boundary node actually computed: zero-share members
        // (dedicated attention exits, zero -ts slices) contribute exact zeros,
        // so they take no part in the reduce and receive the result by copy
        std::vector<size_t> part;
        part.reserve(n_backends);
        for (size_t j = 0; j < n_backends; j++) {
            if (boundary_node(j)->flags & GGML_TENSOR_FLAG_COMPUTE) {
                part.push_back(j);
            }
        }
        if (part.empty()) {
            // every member holds a zero-size slice: the reduced value is zero.
            // FILL, not SCALE by 0: the buffers hold recycled garbage that can
            // be NaN/Inf, and 0.0f * NaN == NaN
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                ggml_tensor * node = boundary_node(j);
                ggml_tensor * node_zero = get_node_aux(node);
                node_zero->op = GGML_OP_FILL;
                node_zero->src[0] = node;
                ggml_set_op_params_f32(node_zero, 0, 0.0f);
                node_zero->data = node->data;
                node_zero->buffer = node->buffer;
                node_zero->flags |= GGML_TENSOR_FLAG_COMPUTE;

                step_cgraphs[j] = get_cgraph_aux();
                step_cgraphs[j]->nodes[0] = node_zero;
                step_cgraphs[j]->n_nodes = 1;
                const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
            return GGML_STATUS_SUCCESS;
        }
        // star reduce: with a local member available as root, fetch every computing
        // member's partial in one batched read (requests overlap on the wire; each
        // read is ordered behind its member's compute by the connection), sum on the
        // host and broadcast identical result bytes with async writes. Replaces the
        // fold+butterfly whose sequential blocking round trips per step dominated
        // boundary latency at decode batch sizes. GGML_META_NO_STAR=1 restores the
        // butterfly. Kept: butterfly for all-wire fleets (no local root).
        static const bool no_star = getenv("GGML_META_NO_STAR") != nullptr;
        if (!no_star && part.size() >= 2 && !backend_ctx->wire_member.empty()) {
            int j_root = -1;
            for (size_t j = 0; j < n_backends; j++) {
                if (!backend_ctx->wire_member[j]) {
                    j_root = (int) j;
                    break;
                }
            }
            bool star_ok = j_root >= 0;
            for (size_t k = 0; star_ok && k < part.size(); k++) {
                ggml_tensor * node = boundary_node(part[k]);
                star_ok = node->type == GGML_TYPE_F32 && ggml_is_contiguous(node);
            }
            if (star_ok) {
                const size_t nbytes = ggml_nbytes(boundary_node(part[0]));
                const size_t n_vals = nbytes/sizeof(float);
                auto & scratch = backend_ctx->star_scratch;
                if (scratch.size() < part.size()*nbytes) {
                    scratch.resize(part.size()*nbytes);
                }
                std::vector<ggml_backend_t>      get_backends(part.size());
                std::vector<const ggml_tensor *> get_tensors(part.size());
                std::vector<void *>              get_datas(part.size());
                std::vector<size_t>              get_sizes(part.size(), nbytes);
                size_t n_plain = 0;
                for (size_t k = 0; k < part.size(); k++) {
                    const size_t j = part[k];
                    auto & bcj = backend_ctx->backend_configs[j];
                    if (fused_fetch_pending[j]) {
                        continue; // its partial arrives as a fused response, collected below
                    }
                    if (!backend_ctx->wire_member[j]) {
                        // a local partial is read directly from device memory - order
                        // it behind the compute that produced it
                        ggml_backend_synchronize(bcj.backend);
                    }
                    get_backends[n_plain] = bcj.backend;
                    get_tensors[n_plain]  = boundary_node(j);
                    get_datas[n_plain]    = scratch.data() + k*nbytes;
                    n_plain++;
                }
                if (backend_ctx->get_batch != nullptr && n_plain > 0) {
                    backend_ctx->get_batch((int) n_plain, get_backends.data(), get_tensors.data(), get_datas.data(), get_sizes.data());
                } else {
                    for (size_t k = 0; k < n_plain; k++) {
                        ggml_backend_tensor_get(get_tensors[k], get_datas[k], 0, nbytes);
                    }
                }
                for (size_t k = 0; k < part.size(); k++) {
                    const size_t j = part[k];
                    if (fused_fetch_pending[j]) {
                        backend_ctx->fused_recv(backend_ctx->backend_configs[j].backend,
                                                scratch.data() + k*nbytes, nbytes);
                        fused_fetch_pending[j] = 0;
                    }
                }
                float * acc = (float *) scratch.data();
                for (size_t k = 1; k < part.size(); k++) {
                    const float * p = (const float *) (scratch.data() + k*nbytes);
                    for (size_t v = 0; v < n_vals; v++) {
                        acc[v] += p[v];
                    }
                }
                fused_distribute(SIZE_MAX, acc, nbytes);
                return GGML_STATUS_SUCCESS;
            }
        }
        if (part.size() >= 2 && part.size() < n_backends) {
            // MEASURED (full V4, Meta(CUDA0,.11,.15), -ts 0,3,2): reducing only
            // among the computing workers looked cheaper on paper but ran 1.34
            // vs 1.80 t/s - it trades coordinator-local star hops through the
            // zeroed CUDA member for worker-to-worker fenced pulls over GbE plus
            // a tail copy-out. Keep zero members in the butterfly (their slices
            // are FILLed below); only the single-computer case skips it.
            part.clear();
            for (size_t j = 0; j < n_backends; j++) {
                part.push_back(j);
                ggml_tensor * node = boundary_node(j);
                if (node->flags & GGML_TENSOR_FLAG_COMPUTE) {
                    continue;
                }
                // FILL, not SCALE by 0: recycled buffers hold NaN/Inf and
                // 0.0f * NaN == NaN would poison the reduce
                auto & bcj = backend_ctx->backend_configs[j];
                ggml_tensor * node_zero = get_node_aux(node);
                node_zero->op = GGML_OP_FILL;
                node_zero->src[0] = node;
                ggml_set_op_params_f32(node_zero, 0, 0.0f);
                node_zero->data = node->data;
                node_zero->buffer = node->buffer;
                node_zero->flags |= GGML_TENSOR_FLAG_COMPUTE;

                step_cgraphs[j] = get_cgraph_aux();
                step_cgraphs[j]->nodes[0] = node_zero;
                step_cgraphs[j]->n_nodes = 1;
                const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
            std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);
        }
        if (part.size() == 1) {
            // the sum IS the single computer's value: broadcast it
            const size_t j_src = part[0];
            auto & bcs = backend_ctx->backend_configs[j_src];
            ggml_tensor * node_src = boundary_node(j_src);
            // local sole computer + fused pipeline: read the value once and push it
            // with each wire member's next subgraph chain in one message (the copy
            // path below pays a bridged copy per member and carries nothing)
            // GGML_META_FUSED_BCAST: 0 = off, 1 = fused SET only, 2 = full chain
            // carriage (default). Diagnostic A/B levels.
            static const int fused_bcast = getenv("GGML_META_FUSED_BCAST") ? atoi(getenv("GGML_META_FUSED_BCAST")) : 2;
            if (fused_bcast > 0 && fused_enabled && !backend_ctx->wire_member[j_src] &&
                node_src->type == GGML_TYPE_F32 && ggml_is_contiguous(node_src)) {
                const size_t nbytes = ggml_nbytes(node_src);
                auto & scratch = backend_ctx->star_scratch;
                if (scratch.size() < nbytes) {
                    scratch.resize(nbytes);
                }
                ggml_backend_synchronize(bcs.backend);
                ggml_backend_tensor_get(node_src, scratch.data(), 0, nbytes);
                fused_distribute(j_src, scratch.data(), nbytes, /*allow_chain =*/ fused_bcast >= 2);
                return GGML_STATUS_SUCCESS;
            }
            std::vector<ggml_backend_t> backends_src, backends_dst;
            std::vector<ggml_tensor *>  srcs, dsts;
            for (size_t j = 0; j < n_backends; j++) {
                if (j == j_src) {
                    continue;
                }
                backends_src.push_back(bcs.backend);
                backends_dst.push_back(backend_ctx->backend_configs[j].backend);
                srcs.push_back(node_src);
                dsts.push_back(boundary_node(j));
            }
            if (backend_ctx->cpy_batch != nullptr && srcs.size() > 1) {
                backend_ctx->cpy_batch((int) srcs.size(), backends_src.data(), backends_dst.data(), srcs.data(), dsts.data());
            } else {
                for (size_t k = 0; k < srcs.size(); k++) {
                    ggml_backend_tensor_copy_async(backends_src[k], backends_dst[k], srcs[k], dsts[k]);
                }
            }
            return GGML_STATUS_SUCCESS;
        }

        // pulls of one reduce step are collected and flushed together: the batch
        // proc (RPC) overlaps them on the wire, the per-pair path serializes on
        // each blocking ack
        struct w2w_copy {
            ggml_backend_t backend_src;
            ggml_backend_t backend_dst;
            ggml_tensor *  src;
            ggml_tensor *  dst;
        };
        std::vector<w2w_copy> pending_copies;
        pending_copies.reserve(n_backends);

        auto flush_copies = [&]() {
            if (backend_ctx->cpy_batch != nullptr && pending_copies.size() > 1) {
                std::vector<ggml_backend_t> backends_src;
                std::vector<ggml_backend_t> backends_dst;
                std::vector<ggml_tensor *>  srcs;
                std::vector<ggml_tensor *>  dsts;
                for (const w2w_copy & c : pending_copies) {
                    backends_src.push_back(c.backend_src);
                    backends_dst.push_back(c.backend_dst);
                    srcs.push_back(c.src);
                    dsts.push_back(c.dst);
                }
                backend_ctx->cpy_batch((int) pending_copies.size(), backends_src.data(), backends_dst.data(), srcs.data(), dsts.data());
            } else {
                for (const w2w_copy & c : pending_copies) {
                    ggml_backend_tensor_copy_async(c.backend_src, c.backend_dst, c.src, c.dst);
                }
            }
            pending_copies.clear();
        };

        auto push_data = [&](const size_t j_src, const size_t j_dst, const size_t i_buf) {
            assert(step_cgraphs[j_dst] == nullptr);
            auto & bcj_src = backend_ctx->backend_configs[j_src];
            auto & bcj_dst = backend_ctx->backend_configs[j_dst];

            ggml_tensor * node_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * node_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            GGML_ASSERT(ggml_is_contiguous(node_src));
            GGML_ASSERT(ggml_is_contiguous(node_dst));

            ggml_tensor * node_tmp = get_node_aux(node_dst);
            set_tmp_data(node_tmp, j_dst, i_buf);

            pending_copies.push_back({bcj_src.backend, bcj_dst.backend, node_src, node_tmp});

            ggml_tensor * node_red = get_node_aux(node_dst);
            node_red->view_src = node_dst->view_src == nullptr ? node_dst : node_dst->view_src;
            node_red->view_offs = node_dst->view_offs;
            node_red->op = GGML_OP_ADD;
            node_red->src[0] = node_dst;
            node_red->src[1] = node_tmp;
            node_red->flags |= GGML_TENSOR_FLAG_COMPUTE;
            ggml_backend_view_init(node_red);

            ggml_cgraph * cgraph_aux = get_cgraph_aux();
            cgraph_aux->nodes[0] = node_red;
            cgraph_aux->n_nodes = 1;
            step_cgraphs[j_dst] = cgraph_aux;
        };

        // fold/butterfly over the PARTICIPANTS only (indices map through part[])
        const size_t n_part = part.size();
        size_t offset_k = n_part/2;
        while ((offset_k & (offset_k - 1)) != 0) {
            offset_k--;
        }
        const size_t offset_k_max = offset_k;
        size_t i_buf = 0;

        // If n_part is not a power of 2, fold in the excess prior to butterfly reduction:
        for (size_t k_src = 2*offset_k_max; k_src < n_part; k_src++) {
            const size_t j_dst = part[k_src - 2*offset_k_max];
            push_data(part[k_src], j_dst, i_buf);
            flush_copies();
            const ggml_status status = ggml_backend_graph_compute_async(backend_ctx->backend_configs[j_dst].backend, step_cgraphs[j_dst]);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
            i_buf = 1;
        }

        // Butterfly reduction:
        for (; offset_k >= 1; offset_k /= 2) {
            std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

            for (size_t k = 0; k < 2*offset_k_max; k++) {
                const size_t k_other = k ^ offset_k;
                if (k_other >= n_part) {
                    continue;
                }
                push_data(part[k], part[k_other], i_buf);
            }
            flush_copies();

            for (size_t k = 0; k < 2*offset_k_max; k++) {
                if (step_cgraphs[part[k]] == nullptr) {
                    continue;
                }
                auto & bcj = backend_ctx->backend_configs[part[k]];
                const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[part[k]]);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
            i_buf++;
        }
        assert(i_buf <= backend_ctx->n_reduce_steps);

        // copy the reduced tensors back to the folded excess and OUT to the
        // members that contributed exact zeros
        for (size_t k = 2*offset_k_max; k < n_part; k++) {
            auto & bcj_src = backend_ctx->backend_configs[part[k - 2*offset_k_max]];
            pending_copies.push_back({bcj_src.backend, backend_ctx->backend_configs[part[k]].backend,
                                      boundary_node(part[k - 2*offset_k_max]), boundary_node(part[k])});
        }
        {
            size_t kp = 0;
            auto & bcj_src = backend_ctx->backend_configs[part[0]];
            for (size_t j = 0; j < n_backends; j++) {
                if (kp < n_part && part[kp] == j) {
                    kp++;
                    continue;
                }
                pending_copies.push_back({bcj_src.backend, backend_ctx->backend_configs[j].backend,
                                          boundary_node(part[0]), boundary_node(j)});
            }
        }
        flush_copies();

        return GGML_STATUS_SUCCESS;
    };


    // GGML_META_TIMING=1: attribute wall time to kernel compute vs AllReduce
    // boundaries. Adds explicit device syncs at each boundary, so it slightly
    // serializes execution - a diagnostic mode, not for production serving.
    // (tm_enabled itself is declared above the reduce lambda - it gates fusion.)
    const auto tm_sync_all = [&]() {
        for (size_t j = 0; j < n_backends; j++) {
            ggml_backend_synchronize(backend_ctx->backend_configs[j].backend);
        }
    };
    int64_t tm_last = tm_enabled ? (tm_sync_all(), ggml_time_us()) : 0;

    for (size_t i = 0; i < backend_ctx->n_subgraphs; i++) {
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            if (fused_carried[j] > 0) {
                // this subgraph's dispatch rode a previous boundary's fused message
                fused_carried[j]--;
                continue;
            }
            const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, bcj.cgraphs[i].cgraph_main);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }
        if (tm_enabled) {
            tm_sync_all();
            const int64_t t = ggml_time_us();
            backend_ctx->tm_compute_us += t - tm_last;
            tm_last = t;
        }

        if (n_backends > 1 && backend_ctx->backend_configs[0].cgraphs[i].reduce) {
            bool backend_allreduce_success = false;
            if (backend_ctx->comm_ctx) {
                std::vector<ggml_tensor *> nodes;
                nodes.reserve(n_backends);
                for (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    ggml_cgraph * cgraph_ij = bcj.cgraphs[i].cgraph_main;
                    nodes.push_back(cgraph_ij->nodes[cgraph_ij->n_nodes-1]);
                }
                backend_allreduce_success = backend_ctx->comm_allreduce(backend_ctx->comm_ctx, nodes.data());
            }

            if (!backend_allreduce_success) {
                const ggml_status status = allreduce_fallback(i);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
            if (tm_enabled) {
                tm_sync_all();
                const int64_t t = ggml_time_us();
                backend_ctx->tm_reduce_us += t - tm_last;
                tm_last = t;
                backend_ctx->tm_reduces++;
            }
        }
    }
    if (tm_enabled && ++backend_ctx->tm_graphs >= 128) {
        fprintf(stderr, "META_TIMING: %" PRId64 " graphs: compute %.2f ms/graph, reduce %.2f ms/graph over %.1f boundaries/graph, build cache %" PRId64 "/%" PRId64 " hits\n",
                backend_ctx->tm_graphs,
                backend_ctx->tm_compute_us / 1000.0 / backend_ctx->tm_graphs,
                backend_ctx->tm_reduce_us  / 1000.0 / backend_ctx->tm_graphs,
                (double) backend_ctx->tm_reduces / backend_ctx->tm_graphs,
                backend_ctx->tm_build_hits, backend_ctx->tm_build_hits + backend_ctx->tm_build_misses);
        backend_ctx->tm_compute_us = 0;
        backend_ctx->tm_reduce_us  = 0;
        backend_ctx->tm_reduces    = 0;
        backend_ctx->tm_graphs     = 0;
        backend_ctx->tm_build_hits   = 0;
        backend_ctx->tm_build_misses = 0;
    }
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_meta_i = {
    /* .get_name                = */ ggml_backend_meta_get_name,
    /* .free                    = */ ggml_backend_meta_free,
    /* .set_tensor_async        = */ ggml_backend_meta_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_meta_get_tensor_async,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ ggml_backend_meta_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_meta_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

bool ggml_backend_is_meta(ggml_backend_t backend) {
    return backend != nullptr && backend->iface.get_name == ggml_backend_meta_i.get_name;
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_meta_context * backend_ctx = new ggml_backend_meta_context(dev, params);

    ggml_backend_t backend = new struct ggml_backend;
    backend->guid    = ggml_backend_meta_guid();
    backend->iface   = ggml_backend_meta_i;
    backend->device  = dev;
    backend->context = backend_ctx;
    return backend;
}

size_t ggml_backend_meta_n_backends(ggml_backend_t meta_backend) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs.size();
}

ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t index) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs[index].backend;
}
