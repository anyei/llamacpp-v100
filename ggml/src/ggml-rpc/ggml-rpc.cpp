#include "ggml-rpc.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include "transport.h"

#include <array>
#include <atomic>
#include <cinttypes>
#include <random>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <chrono>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <cstring>
#include <fstream>
#include <filesystem>
#ifndef _WIN32
#    include <fcntl.h>
#    include <unistd.h>
#endif
#include <algorithm>

static const char * RPC_DEBUG = std::getenv("GGML_RPC_DEBUG");

#define LOG_DBG(...) \
    do { if (RPC_DEBUG) GGML_LOG_DEBUG(__VA_ARGS__); } while (0)


namespace fs = std::filesystem;

// macro for nicer error messages on server crash
// TASKS.md #29b: a dropped/dead worker must fail the coordinator's REQUEST, not the
// process. Any RPC failure marks the endpoint below; subsequent operations degrade to
// cheap no-ops (outputs zeroed), graph_compute reports GGML_STATUS_FAILED, and the
// caller (llama) surfaces a clean decode error. Restarting the coordinator re-splits
// the model across the workers that are still alive (--rpc-discover +
// --rpc-skip-unavailable make that fully automatic).
static std::mutex g_rpc_failed_mutex;
static std::unordered_set<std::string> g_rpc_failed_endpoints;

static void rpc_mark_failed(const std::string & endpoint, const char * where) {
    std::lock_guard<std::mutex> lock(g_rpc_failed_mutex);
    if (g_rpc_failed_endpoints.insert(endpoint).second) {
        GGML_LOG_ERROR("RPC worker '%s' failed in %s: connection lost or worker crashed. "
                       "Failing the affected requests; restart the coordinator to re-split "
                       "across the live workers.\n",
                       endpoint.empty() ? "?" : endpoint.c_str(), where);
    }
}

static bool rpc_endpoint_failed(const std::string & endpoint) {
    std::lock_guard<std::mutex> lock(g_rpc_failed_mutex);
    return g_rpc_failed_endpoints.count(endpoint) > 0;
}

bool ggml_backend_rpc_any_endpoint_failed(void) {
    std::lock_guard<std::mutex> lock(g_rpc_failed_mutex);
    return !g_rpc_failed_endpoints.empty();
}


// all RPC structures must be packed
#pragma pack(push, 1)
// ggml_tensor is serialized into rpc_tensor
struct rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    char name[GGML_MAX_NAME];

    char padding[4];
};

static_assert(sizeof(rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");

// RPC commands
enum rpc_cmd {
    RPC_CMD_ALLOC_BUFFER = 0,
    RPC_CMD_GET_ALIGNMENT,
    RPC_CMD_GET_MAX_SIZE,
    RPC_CMD_BUFFER_GET_BASE,
    RPC_CMD_FREE_BUFFER,
    RPC_CMD_BUFFER_CLEAR,
    RPC_CMD_SET_TENSOR,
    RPC_CMD_SET_TENSOR_HASH,
    RPC_CMD_GET_TENSOR,
    RPC_CMD_COPY_TENSOR,
    RPC_CMD_GRAPH_COMPUTE,
    RPC_CMD_GET_DEVICE_MEMORY,
    RPC_CMD_INIT_TENSOR,
    RPC_CMD_GET_ALLOC_SIZE,
    RPC_CMD_HELLO,
    RPC_CMD_DEVICE_COUNT,
    RPC_CMD_GRAPH_RECOMPUTE,
    RPC_CMD_SET_SPLIT_STATES,
    RPC_CMD_GET_DEVICE_DESC,
    RPC_CMD_BUFFER_SET_USAGE,
    // proto 4.2: completion marker for async command tracking. The server executes
    // commands strictly in order per connection, so the (empty) response to a PING
    // proves every command sent before it has finished. Events map onto PING
    // sequence numbers client-side; no server-side state is needed.
    RPC_CMD_PING,
    // proto 4.2: worker-to-worker activation transfer. The coordinator tells the
    // destination worker to pull a tensor directly from the source worker instead
    // of bridging the bytes through itself. The pull is fenced: it waits until the
    // source has executed >= fence_seq commands on the coordinator's connection
    // (identified via GET_CONN_ID), i.e. until the compute that produces the data
    // has actually run.
    RPC_CMD_GET_CONN_ID,
    RPC_CMD_GET_TENSOR_FENCED,
    RPC_CMD_COPY_FROM_REMOTE,
    // proto 4.3: uid-keyed server-side graph cache. A client that submits many
    // distinct graphs per token (the meta/tensor-parallel backend submits one per
    // AllReduce boundary) sends each graph once and then recomputes it by uid,
    // instead of re-serializing every subgraph every token.
    RPC_CMD_GRAPH_COMPUTE_UID,
    RPC_CMD_GRAPH_RECOMPUTE_UID,
    // proto 4.5: fused boundary command. One message optionally carries a small
    // WRITE (the previous boundary's reduced value), a graph RECOMPUTE by uid and
    // a boundary-tensor READ whose bytes come back in the response - the whole
    // steady-state EP boundary cycle in one round trip instead of three commands
    // plus a separate fetch turnaround.
    RPC_CMD_GRAPH_FUSED,
    // proto 4.7 (TASKS.md #35 fleet UI): worker introspection + ops. All three are
    // handled OUTSIDE the exec lock in the serve loop so they respond even while a
    // graph is computing (a hung worker can still be inspected and shut down).
    RPC_CMD_GET_LOG,
    RPC_CMD_SHUTDOWN,
    RPC_CMD_GET_SCORE,
    // proto 4.8 (TASKS.md #44): SET_TENSOR_HASH carrying source provenance so a
    // --model-dir worker can serve an arbitrary slice of a local GGUF tensor
    RPC_CMD_SET_TENSOR_HASH2,
    RPC_CMD_COUNT,
};

static_assert(RPC_CMD_HELLO == 14, "RPC_CMD_HELLO must be always 14");

// Try RPC_CMD_SET_TENSOR_HASH first when data size is larger than this threshold
const size_t HASH_THRESHOLD = 10 * 1024 * 1024;

struct rpc_msg_hello_req {
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

struct rpc_msg_hello_rsp {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    uint8_t padding;
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

struct rpc_msg_device_count_rsp {
    uint32_t device_count;
};

// proto 4.7 fleet commands. GET_LOG's response is raw variable-size text
// (the standard | size(8) | data | framing carries the length).
struct rpc_msg_get_log_req {
    uint64_t max_bytes;
};

// refuse a stray/garbage SHUTDOWN unless it carries the magic
#define RPC_SHUTDOWN_MAGIC 0x35464C45u // "5FLE"

struct rpc_msg_shutdown_req {
    uint32_t magic;
};

struct rpc_msg_shutdown_rsp {
    uint8_t ok;
};

struct rpc_msg_get_score_rsp {
    float   bw_gbps;
    float   mm_gflops;
    uint8_t valid;
    uint8_t padding[3];
};

struct rpc_msg_get_alloc_size_req {
    uint32_t   device;
    rpc_tensor tensor;
    rpc_tensor srcs[GGML_MAX_SRC];
};

struct rpc_msg_get_alloc_size_rsp {
    uint64_t alloc_size;
};

struct rpc_msg_init_tensor_req {
    rpc_tensor tensor;
};

struct rpc_msg_alloc_buffer_req {
    uint32_t device;
    uint64_t size;
};

struct rpc_msg_alloc_buffer_rsp {
    uint64_t remote_ptr;
    uint64_t remote_size;
};

struct rpc_msg_get_alignment_req {
    uint32_t device;
};

struct rpc_msg_get_alignment_rsp {
    uint64_t alignment;
};

struct rpc_msg_get_max_size_req {
    uint32_t device;
};

struct rpc_msg_get_max_size_rsp {
    uint64_t max_size;
};

struct rpc_msg_buffer_get_base_req {
    uint64_t remote_ptr;
};

struct rpc_msg_buffer_get_base_rsp {
    uint64_t base_ptr;
};

struct rpc_msg_free_buffer_req {
    uint64_t remote_ptr;
};

struct rpc_msg_buffer_clear_req {
    uint64_t remote_ptr;
    uint8_t value;
};

struct rpc_msg_set_tensor_hash_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t hash;
};

struct rpc_msg_set_tensor_hash_rsp {
    uint8_t result;
};

// proto 4.8 (TASKS.md #44): the offered bytes are src_n_rows reads of src_row_size
// at src_offset + i*src_row_stride within the named GGUF tensor's data. A worker
// with --model-dir assembles the slice by pread and hash-verifies before applying;
// n_rows == 1 is the contiguous case.
struct rpc_msg_set_tensor_hash2_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t hash;
    uint64_t src_offset;
    uint64_t src_row_size;
    uint64_t src_n_rows;
    uint64_t src_row_stride;
    char     src_name[GGML_MAX_NAME];
};

struct rpc_msg_get_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
};

struct rpc_msg_copy_tensor_req {
    rpc_tensor src;
    rpc_tensor dst;
};

struct rpc_msg_copy_tensor_rsp {
    uint8_t result;
};

struct rpc_msg_get_conn_id_rsp {
    uint64_t conn_id;
};

struct rpc_msg_get_tensor_fenced_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint64_t fence_conn; // connection id on the receiving server whose progress gates this read
    uint64_t fence_seq;  // read allowed once that connection has executed >= fence_seq commands
};

// RPC_CMD_COPY_FROM_REMOTE request is variable-size:
// | rpc_tensor src | rpc_tensor dst | fence_conn (8) | fence_seq (8) | ep_len (4) | src endpoint bytes |
struct rpc_msg_copy_from_remote_rsp {
    uint8_t result;
};

struct rpc_msg_get_device_memory_req {
    uint32_t device;
};

struct rpc_msg_get_device_memory_rsp {
    uint64_t free_mem;
    uint64_t total_mem;
};

struct rpc_msg_get_device_desc_req {
    uint32_t device;
};

struct rpc_msg_buffer_set_usage_req {
    uint64_t remote_ptr;
    uint32_t usage;
};

struct rpc_msg_get_device_desc_rsp {
    char desc[128];
};

struct rpc_msg_graph_recompute_req {
    uint32_t device;
};

struct rpc_msg_graph_recompute_uid_req {
    uint32_t device;
    uint32_t pad;
    uint64_t uid;
};

// server-side per-device cap on cached uid-keyed graphs (proto 4.3). The client
// tracks at most half of this, so a uid the client remembers is always cached
// server-side; eviction on either side only costs a re-send, never a failure.
#define RPC_GRAPH_UID_CACHE_CAP 512

#pragma pack(pop)

// RPC data structures

static ggml_guid_t ggml_backend_rpc_guid() {
    static ggml_guid guid = {0x99, 0x68, 0x5b, 0x6c, 0xd2, 0x83, 0x3d, 0x24, 0x25, 0x36, 0x72, 0xe1, 0x5b, 0x0e, 0x14, 0x03};
    return &guid;
}

struct ggml_backend_rpc_device_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    std::string description;
    uint64_t    last_graph_uid;
    bool        worker_is_cpu; // the worker-side device is its CPU (RAM), not a GPU

    // proto 4.3 uid-keyed graph cache: uids known to be stored server-side, in
    // insertion order for client-side eviction. Tied to the socket they were sent
    // on - a reconnected socket starts with an empty server cache.
    std::unordered_set<uint64_t> sent_graph_uids;
    std::deque<uint64_t>         sent_graph_uids_order;
    void *                       sent_graph_uids_sock = nullptr;
};

struct ggml_backend_rpc_buffer_type_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    size_t      alignment;
    size_t      max_size;
};

struct ggml_backend_rpc_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
};

struct ggml_backend_rpc_buffer_context {
    std::shared_ptr<socket_t> sock;
    void * base_ptr;
    uint64_t remote_ptr;
};

// surgical re-provision journal (TASKS.md #29c refinement): while a weight-class
// RPC buffer is populated, record enough to rebuild it on a RESTARTED worker at
// the same endpoint without the coordinator re-reading the model - big writes as
// (offset, size, hash) replayed via SET_TENSOR_HASH against the worker's local
// cache, small writes as raw bytes (bounded). Offsets are relative to the buffer
// base so the replay works at whatever address the fresh server hands out.
struct rpc_buffer_journal {
    struct set_entry {
        rpc_tensor root;   // data field holds the OFFSET from base
        uint64_t   offset;
        uint64_t   size;
        uint64_t   hash;      // valid when spill_off == UINT64_MAX
        uint64_t   spill_off; // literal bytes at this offset of the spill file
        // source provenance (proto 4.8, TASKS.md #44): non-empty src_name replays
        // via SET_TENSOR_HASH2 so a --model-dir worker rebuilds slices that never
        // touched its disk cache
        char       src_name[GGML_MAX_NAME] = {0};
        uint64_t   src_offset     = 0;
        uint64_t   src_row_size   = 0;
        uint64_t   src_n_rows     = 0;
        uint64_t   src_row_stride = 0;
    };
    // keyed by BUFFER-relative start (root offset from base + write offset): the
    // key must be unique per buffer region - keying by the tensor-relative offset
    // alone collided across tensors (every tensor's chunk 0 is offset 0) and
    // silently dropped all but the last tensor's chunks from the replay
    std::map<uint64_t, set_entry> sets;
    std::vector<rpc_tensor>       init_tensors; // data field holds the OFFSET
    // a fresh load zero-fills KV-class buffers; replay must too, or the re-created
    // allocation serves recycled garbage (NaN-poisoning, the FILL-not-SCALE lesson)
    bool    cleared     = false;
    uint8_t clear_value = 0;
    // small writes go to an anonymous temp file, not RAM: segmented tensors
    // upload as thousands of strided per-row writes (hundreds of MB)
    FILE *   spill      = nullptr;
    uint64_t spill_size = 0;
    bool     disabled   = false; // spill budget exceeded or unavailable

    ~rpc_buffer_journal() {
        if (spill != nullptr) {
            fclose(spill);
        }
    }
    rpc_buffer_journal() = default;
    rpc_buffer_journal(const rpc_buffer_journal &) = delete;
    rpc_buffer_journal & operator=(const rpc_buffer_journal &) = delete;
};
static std::mutex g_rpc_journal_mutex;
static std::unordered_map<ggml_backend_buffer_t, rpc_buffer_journal> g_rpc_journals;

static size_t rpc_journal_small_budget() {
    static const size_t budget = []() {
        const char * env = getenv("GGML_RPC_JOURNAL_MAX_MIB");
        return (size_t) (env != nullptr ? atoll(env) : 4096) * 1024u * 1024u;
    }();
    return budget;
}

// RPC helper functions

// Computes FNV-1a hash of the data
static uint64_t fnv_hash(const uint8_t * data, size_t len) {
    const uint64_t fnv_prime = 0x100000001b3ULL;
    uint64_t hash = 0xcbf29ce484222325ULL;

    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_prime;
    }
    return hash;
}

static bool send_msg(socket_ptr sock, const void * msg, size_t msg_size) {
    if (!sock->send_data(&msg_size, sizeof(msg_size))) {
        return false;
    }
    return sock->send_data(msg, msg_size);
}

static bool recv_msg(socket_ptr sock, void * msg, size_t msg_size) {
    uint64_t size;
    if (!sock->recv_data(&size, sizeof(size))) {
        return false;
    }
    if (size != msg_size) {
        return false;
    }
    return sock->recv_data(msg, msg_size);
}

static bool recv_msg(socket_ptr sock, std::vector<uint8_t> & input) {
    uint64_t size;
    if (!sock->recv_data(&size, sizeof(size))) {
        return false;
    }
    try {
        input.resize(size);
    } catch (const std::bad_alloc & e) {
        GGML_LOG_ERROR("Failed to allocate input buffer of size %" PRIu64 "\n", size);
        return false;
    }
    return sock->recv_data(input.data(), size);
}

static bool parse_endpoint(const std::string & endpoint, std::string & host, int & port) {
    size_t pos = endpoint.find(':');
    if (pos == std::string::npos) {
        return false;
    }
    host = endpoint.substr(0, pos);
    try {
        port = std::stoi(endpoint.substr(pos + 1));
    } catch (...) {
        return false;
    }
    return true;
}

// loopback endpoints (a same-box worker discovered via the loopback beacon, or a
// static --rpc 127.0.0.1/localhost entry) are only valid from the coordinator's host
static bool rpc_endpoint_is_loopback(const std::string & endpoint) {
    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        return false;
    }
    return host == "localhost" || host.rfind("127.", 0) == 0;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// No response
// tensor-parallel islands: split states uploaded by the coordinator, resolved by
// tensor name from the worker-side meta device callback (see tools/rpc/rpc-server.cpp)
static std::unordered_map<std::string, ggml_backend_meta_split_state> g_rpc_split_states;
static std::mutex g_rpc_split_states_mutex;

static bool rpc_split_states_ingest(const uint8_t * data, size_t size) {
    std::lock_guard<std::mutex> lock(g_rpc_split_states_mutex);
    size_t off = 0;
    while (off < size) {
        if (off + sizeof(uint32_t) > size) {
            return false;
        }
        uint32_t name_len;
        memcpy(&name_len, data + off, sizeof(name_len));
        off += sizeof(name_len);
        if (name_len == 0 || off + name_len + sizeof(ggml_backend_meta_split_state) > size) {
            return false;
        }
        std::string name((const char *) data + off, name_len);
        off += name_len;
        ggml_backend_meta_split_state state;
        memcpy(&state, data + off, sizeof(state));
        off += sizeof(state);
        g_rpc_split_states[std::move(name)] = state;
    }
    return true;
}

bool ggml_backend_rpc_split_state_lookup(const char * name, struct ggml_backend_meta_split_state * out) {
    std::lock_guard<std::mutex> lock(g_rpc_split_states_mutex);
    auto it = g_rpc_split_states.find(name);
    if (it == g_rpc_split_states.end()) {
        return false;
    }
    *out = it->second;
    return true;
}

// client-side async command tracking (proto 4.2). The server executes commands
// strictly in order per connection, so a deferred RPC_CMD_PING marker's (empty)
// response proves that every command sent before it has completed. Blocking
// commands must drain outstanding PING responses first - responses arrive on the
// same stream in command order.
struct ggml_backend_rpc_async_state {
    std::mutex mutex;
    uint64_t pings_sent = 0;
    uint64_t pings_done = 0;
    // every command sent on this socket, HELLO included - must stay in lockstep
    // with the server's per-connection executed counter (fence sequencing)
    uint64_t cmds_sent = 0;
    uint8_t  server_minor = 0; // from HELLO; gates use of 4.2+ commands
    uint64_t conn_id = 0;      // this socket's connection id on the server (0 = not fetched)
    std::string endpoint;      // for failure attribution (set by get_socket)
    struct rpc_ep_stat * stat = nullptr; // per-endpoint counters (set with endpoint)
};

static std::mutex g_rpc_async_reg_mutex;
static std::unordered_map<socket_t *, std::unique_ptr<ggml_backend_rpc_async_state>> g_rpc_async_reg;

static ggml_backend_rpc_async_state & rpc_async_state(socket_t * sock) {
    std::lock_guard<std::mutex> lock(g_rpc_async_reg_mutex);
    auto & e = g_rpc_async_reg[sock];
    if (!e) {
        e = std::make_unique<ggml_backend_rpc_async_state>();
    }
    return *e;
}

// the registry is keyed by raw socket address: when a connection dies and a new
// socket_t is allocated at the same address, it must NOT inherit the dead
// connection's counters (cmds_sent would run ahead of the new connection's
// executed count on the server and every fence against it would hang; stale
// pings_sent would make drains wait for responses that will never come).
// Call for every freshly created socket, before its HELLO.
static void rpc_async_state_reset(socket_t * sock) {
    std::lock_guard<std::mutex> lock(g_rpc_async_reg_mutex);
    g_rpc_async_reg.erase(sock);
}

// caller must hold st.mutex
// GGML_RPC_STATS=1: client-side per-command call/byte counters, dumped at exit (diagnostic)
struct rpc_client_stats {
    std::atomic<uint64_t> count[RPC_CMD_COUNT] = {};
    std::atomic<uint64_t> bytes[RPC_CMD_COUNT] = {};
    ~rpc_client_stats() {
        for (int i = 0; i < RPC_CMD_COUNT; i++) {
            if (count[i] > 0) {
                fprintf(stderr, "RPC_STATS: cmd %2d: %10" PRIu64 " calls %14" PRIu64 " bytes\n",
                        i, count[i].load(), bytes[i].load());
            }
        }
    }
};

static rpc_client_stats * rpc_stats() {
    static const bool enabled = getenv("GGML_RPC_STATS") != nullptr;
    if (!enabled) {
        return nullptr;
    }
    static rpc_client_stats st;
    return &st;
}

// per-endpoint transfer counters (TASKS.md #35e), always on: a handful of atomic
// adds per command is noise next to a socket round trip. Entries are never freed
// (an endpoint's identity outlives its sockets, and fleets are small).
struct rpc_ep_stat {
    std::atomic<uint64_t> bytes_sent{0};
    std::atomic<uint64_t> bytes_recv{0};
    std::atomic<uint64_t> n_calls{0};
    std::atomic<uint64_t> ewma_latency_us{0};
    std::atomic<uint64_t> weights_cached_bytes{0};
    std::atomic<uint64_t> weights_streamed_bytes{0};

    void observe_latency(uint64_t us) {
        // EWMA alpha 1/8; a racing store may drop one sample - fine for a gauge
        uint64_t prev = ewma_latency_us.load(std::memory_order_relaxed);
        ewma_latency_us.store(prev == 0 ? us : (prev * 7 + us) / 8, std::memory_order_relaxed);
    }
};

// file-scope so the per-endpoint enumerator (fleet load view) can walk it; an
// endpoint appears the moment the client first connects to it, so during a
// streamed first load this map IS the live list of workers receiving weights.
static std::mutex g_rpc_ep_stat_mutex;
static std::unordered_map<std::string, std::unique_ptr<rpc_ep_stat>> g_rpc_ep_stats;

static rpc_ep_stat * rpc_ep_stat_get(const std::string & endpoint) {
    std::lock_guard<std::mutex> lock(g_rpc_ep_stat_mutex);
    auto & e = g_rpc_ep_stats[endpoint];
    if (!e) {
        e = std::make_unique<rpc_ep_stat>();
    }
    return e.get();
}

bool ggml_backend_rpc_endpoint_stats(const char * endpoint, struct ggml_backend_rpc_ep_stats * out) {
    if (endpoint == nullptr || out == nullptr) {
        return false;
    }
    rpc_ep_stat * st = rpc_ep_stat_get(endpoint);
    out->bytes_sent      = st->bytes_sent.load(std::memory_order_relaxed);
    out->bytes_recv      = st->bytes_recv.load(std::memory_order_relaxed);
    out->n_calls         = st->n_calls.load(std::memory_order_relaxed);
    out->ewma_latency_us = st->ewma_latency_us.load(std::memory_order_relaxed);
    out->weights_cached_bytes   = st->weights_cached_bytes.load(std::memory_order_relaxed);
    out->weights_streamed_bytes = st->weights_streamed_bytes.load(std::memory_order_relaxed);
    return true;
}

// enumerate every endpoint the client has talked to, with its live counters
// (TASKS.md #35a load view). Race-free: no model-thread state, just the global
// stats map - safe to call from an HTTP thread while the model is still loading.
void ggml_backend_rpc_foreach_endpoint_stat(
        void (*cb)(const char * endpoint, const struct ggml_backend_rpc_ep_stats * st, void * user_data),
        void * user_data) {
    if (cb == nullptr) {
        return;
    }
    std::vector<std::pair<std::string, ggml_backend_rpc_ep_stats>> snapshot;
    {
        std::lock_guard<std::mutex> lock(g_rpc_ep_stat_mutex);
        snapshot.reserve(g_rpc_ep_stats.size());
        for (const auto & [ep, st] : g_rpc_ep_stats) {
            ggml_backend_rpc_ep_stats s;
            s.bytes_sent      = st->bytes_sent.load(std::memory_order_relaxed);
            s.bytes_recv      = st->bytes_recv.load(std::memory_order_relaxed);
            s.n_calls         = st->n_calls.load(std::memory_order_relaxed);
            s.ewma_latency_us = st->ewma_latency_us.load(std::memory_order_relaxed);
            s.weights_cached_bytes   = st->weights_cached_bytes.load(std::memory_order_relaxed);
            s.weights_streamed_bytes = st->weights_streamed_bytes.load(std::memory_order_relaxed);
            snapshot.emplace_back(ep, s);
        }
    }
    for (const auto & [ep, s] : snapshot) {
        cb(ep.c_str(), &s, user_data);
    }
}

// worker-side log ring served by RPC_CMD_GET_LOG (TASKS.md #35c). rpc-server wires
// its ggml log callback to ggml_backend_rpc_log_append; the server-lifecycle logs in
// this file go through GGML_LOG_* and land here the same way.
static constexpr size_t RPC_LOG_RING_CAP = 64 * 1024;
static std::mutex  g_rpc_log_mutex;
static std::string g_rpc_log_ring;

void ggml_backend_rpc_log_append(const char * text) {
    if (text == nullptr || *text == '\0') {
        return;
    }
    std::lock_guard<std::mutex> lock(g_rpc_log_mutex);
    g_rpc_log_ring += text;
    if (g_rpc_log_ring.size() > RPC_LOG_RING_CAP) {
        // trim to the first line boundary that fits the cap
        size_t cut = g_rpc_log_ring.size() - RPC_LOG_RING_CAP;
        size_t nl  = g_rpc_log_ring.find('\n', cut);
        g_rpc_log_ring.erase(0, nl == std::string::npos ? cut : nl + 1);
    }
}

static std::string rpc_log_tail(size_t max_bytes) {
    std::lock_guard<std::mutex> lock(g_rpc_log_mutex);
    if (g_rpc_log_ring.size() <= max_bytes) {
        return g_rpc_log_ring;
    }
    return g_rpc_log_ring.substr(g_rpc_log_ring.size() - max_bytes);
}

// worker-side speed score, published by rpc-server after its startup benchmark
// (TASKS.md #35f); read by the beacon payload and RPC_CMD_GET_SCORE
static std::atomic<float> g_worker_score_bw{0.0f};
static std::atomic<float> g_worker_score_fl{0.0f};
static std::atomic<bool>  g_worker_score_valid{false};

void ggml_backend_rpc_set_worker_score(float bw_gbps, float mm_gflops) {
    g_worker_score_bw.store(bw_gbps);
    g_worker_score_fl.store(mm_gflops);
    g_worker_score_valid.store(true);
}

// remote shutdown is OFF by default (TASKS.md #35d review): RPC is unauthenticated,
// so a peer that can reach the compute port must not be able to kill the process.
// rpc-server opts in with --allow-shutdown when it runs under a supervisor.
static std::atomic<bool> g_worker_shutdown_enabled{false};

void ggml_backend_rpc_set_shutdown_enabled(bool enabled) {
    g_worker_shutdown_enabled.store(enabled);
}

static bool send_rpc_cmd_raw(socket_ptr sock, ggml_backend_rpc_async_state & st, enum rpc_cmd cmd, const void * input, size_t input_size) {
    if (rpc_client_stats * s = rpc_stats()) {
        s->count[cmd]++;
        s->bytes[cmd] += input_size;
    }
    if (st.stat != nullptr) {
        st.stat->n_calls.fetch_add(1, std::memory_order_relaxed);
        st.stat->bytes_sent.fetch_add(input_size + sizeof(uint8_t) + sizeof(uint64_t), std::memory_order_relaxed);
    }
    uint8_t cmd_byte = cmd;
    if (!sock->send_data(&cmd_byte, sizeof(cmd_byte))) {
        return false;
    }
    if (!sock->send_data(&input_size, sizeof(input_size))) {
        return false;
    }
    if (!sock->send_data(input, input_size)) {
        return false;
    }
    st.cmds_sent++;
    return true;
}

static bool rpc_drain_pings_locked(socket_ptr sock, ggml_backend_rpc_async_state & st, uint64_t target) {
    while (st.pings_done < target) {
        uint64_t size;
        if (!sock->recv_data(&size, sizeof(size))) {
            return false;
        }
        if (size != 0) {
            return false; // stream out of sync - a PING response is always empty
        }
        st.pings_done++;
    }
    return true;
}

static bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size) {
    ggml_backend_rpc_async_state & st = rpc_async_state(sock.get());
    std::lock_guard<std::mutex> lock(st.mutex);
    return send_rpc_cmd_raw(sock, st, cmd, input, input_size);
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
static bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size, void * output, size_t output_size) {
    ggml_backend_rpc_async_state & st = rpc_async_state(sock.get());
    std::lock_guard<std::mutex> lock(st.mutex);
    if (!rpc_drain_pings_locked(sock, st, st.pings_sent)) {
        return false;
    }
    const auto t_start = std::chrono::steady_clock::now();
    if (!send_rpc_cmd_raw(sock, st, cmd, input, input_size)) {
        return false;
    }
    uint64_t out_size;
    if (!sock->recv_data(&out_size, sizeof(out_size))) {
        return false;
    }
    if (out_size != output_size) {
        return false;
    }
    if (!sock->recv_data(output, output_size)) {
        return false;
    }
    if (st.stat != nullptr) {
        st.stat->bytes_recv.fetch_add(output_size + sizeof(uint64_t), std::memory_order_relaxed);
        st.stat->observe_latency(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t_start).count());
    }
    return true;
}

// send a completion marker without waiting; returns its sequence number
static uint64_t rpc_ping_async(socket_ptr sock) {
    ggml_backend_rpc_async_state & st = rpc_async_state(sock.get());
    std::lock_guard<std::mutex> lock(st.mutex);
    bool status = send_rpc_cmd_raw(sock, st, RPC_CMD_PING, nullptr, 0);
    if (!status) {
        rpc_mark_failed(st.endpoint, __func__);
    }
    return ++st.pings_sent;
}

// wait until the marker with the given sequence number (0 = all sent so far) has completed
static void rpc_sync_pings(socket_ptr sock, uint64_t seq) {
    ggml_backend_rpc_async_state & st = rpc_async_state(sock.get());
    std::lock_guard<std::mutex> lock(st.mutex);
    const uint64_t target = seq == 0 ? st.pings_sent : std::min(seq, st.pings_sent);
    bool status = rpc_drain_pings_locked(sock, st, target);
    if (!status) {
        rpc_mark_failed(st.endpoint, __func__);
        st.pings_done = target; // never re-wait for pongs that will not arrive
    }
}

// RPC client-side implementation

// Performs HELLO handshake with transport auto-negotiation.
// Advertises local capabilities via conn_caps; if the server responds with
// matching capabilities, the socket is upgraded transparently.
static bool negotiate_hello(const std::shared_ptr<socket_t> & sock) {
    rpc_msg_hello_req request = {};
    rpc_msg_hello_rsp response = {};

    sock->get_caps(request.conn_caps);

    bool status = send_rpc_cmd(sock, RPC_CMD_HELLO, &request, sizeof(request), &response, sizeof(response));
    if (!status) {
        GGML_LOG_ERROR("RPC HELLO failed: server dropped the connection\n");
        return false;
    }

    if (response.major != RPC_PROTO_MAJOR_VERSION) {
        GGML_LOG_ERROR("RPC server version mismatch: %d.%d.%d (need major %d)\n",
                       response.major, response.minor, response.patch, RPC_PROTO_MAJOR_VERSION);
        return false;
    }
    // forward-tolerant: a server with a HIGHER minor is fine - it is a strict
    // superset, and we only ever use commands up to OUR minor. Clamping here (vs
    // the old `minor > ours -> reject`) keeps mixed-minor fleets working so a
    // rolling upgrade in either order never takes the fleet down (TASKS.md #35).
    rpc_async_state(sock.get()).server_minor = std::min<uint8_t>(response.minor, RPC_PROTO_MINOR_VERSION);

    sock->update_caps(response.conn_caps);
    return true;
}

// this socket's connection id on the server (proto 4.2+), fetched once and cached
static uint64_t rpc_get_conn_id(socket_ptr sock) {
    ggml_backend_rpc_async_state & st = rpc_async_state(sock.get());
    {
        std::lock_guard<std::mutex> lock(st.mutex);
        if (st.conn_id != 0) {
            return st.conn_id;
        }
        if (st.server_minor < 2) {
            return 0;
        }
    }
    rpc_msg_get_conn_id_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_CONN_ID, nullptr, 0, &response, sizeof(response));
    if (!status) {
        rpc_mark_failed(st.endpoint, __func__);
        return 0; // callers treat 0 as "no fence available"
    }
    std::lock_guard<std::mutex> lock(st.mutex);
    st.conn_id = response.conn_id;
    return st.conn_id;
}

static std::shared_ptr<socket_t> get_socket_impl(const std::string & endpoint, bool fresh) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static std::unordered_map<std::string, std::weak_ptr<socket_t>> sockets;

    auto it = sockets.find(endpoint);
    if (!fresh && it != sockets.end()) {
        if (auto sock = it->second.lock()) {
            return sock;
        }
    }
    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        GGML_LOG_ERROR("Failed to parse endpoint: %s\n", endpoint.c_str());
        return nullptr;
    }

    if (!rpc_transport_init()) {
        return nullptr;
    }
    auto sock = socket_t::connect(host.c_str(), port);
    if (sock == nullptr) {
        return nullptr;
    }
    // fresh connection: never inherit a dead socket's counters (address reuse)
    rpc_async_state_reset(sock.get());
    if (!negotiate_hello(sock)) {
        return nullptr;
    }
    LOG_DBG("[%s] connected to %s\n", __func__, endpoint.c_str());
    {
        ggml_backend_rpc_async_state & st = rpc_async_state(sock.get());
        st.endpoint = endpoint;
        st.stat     = rpc_ep_stat_get(endpoint);
    }
    sockets[endpoint] = sock;
    return sock;
}

// dedicated one-shot connection for fleet ops (log fetch / shutdown / score):
// deliberately NOT registered in the socket cache, so it can never displace or
// interleave with the compute connection, and works while that one is busy.
// A short TCP probe FIRST bounds the wait: without it a dead worker would block
// the caller (an HTTP thread) for the full ~130s kernel SYN timeout.
static std::shared_ptr<socket_t> rpc_connect_ephemeral(const std::string & endpoint) {
    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        GGML_LOG_ERROR("Failed to parse endpoint: %s\n", endpoint.c_str());
        return nullptr;
    }
    if (!rpc_transport_init()) {
        return nullptr;
    }
    if (!rpc_probe_endpoint(host.c_str(), port, 1500)) {
        return nullptr; // unreachable now - do not sink an HTTP thread into a blocking connect
    }
    auto sock = socket_t::connect(host.c_str(), port);
    if (sock == nullptr) {
        return nullptr;
    }
    rpc_async_state_reset(sock.get());
    if (!negotiate_hello(sock)) {
        // drop the async-state entry so an ephemeral connection never leaks it
        // (the compute path erases on socket reuse; ephemeral sockets never reuse)
        rpc_async_state_reset(sock.get());
        return nullptr;
    }
    ggml_backend_rpc_async_state & st = rpc_async_state(sock.get());
    st.endpoint = endpoint;
    // no st.stat: fleet polling must not pollute the compute transfer counters
    return sock;
}

// erases an ephemeral socket's async-state registry entry on scope exit; without
// it g_rpc_async_reg grows unbounded (fleet polling dials a fresh socket each op,
// and ephemeral sockets never reuse an address to trigger the reset-on-create)
struct rpc_ephemeral_guard {
    socket_t * sock;
    ~rpc_ephemeral_guard() { if (sock != nullptr) { rpc_async_state_reset(sock); } }
};

// bounded reachability probe for fleet callers (proto 4.7): a plain TCP connect
// with a timeout, so an HTTP status poll never blocks on a dead worker. Unlike
// ggml_backend_rpc_dev_reachable (which goes through the blocking cached-socket
// path), this cannot stall.
bool ggml_backend_rpc_endpoint_reachable(const char * endpoint, int timeout_ms) {
    if (endpoint == nullptr) {
        return false;
    }
    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        return false;
    }
    if (!rpc_transport_init()) {
        return false;
    }
    return rpc_probe_endpoint(host.c_str(), port, timeout_ms);
}

static std::shared_ptr<socket_t> get_socket(const std::string & endpoint) {
    return get_socket_impl(endpoint, /*fresh =*/ false);
}

static void ggml_backend_rpc_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    {
        std::lock_guard<std::mutex> lock(g_rpc_journal_mutex);
        g_rpc_journals.erase(buffer);
    }
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_msg_free_buffer_req request = {ctx->remote_ptr};
    if (getenv("GGML_RPC_DEBUG_BUF") != nullptr) {
        GGML_LOG_INFO("[rpc-buf-debug] free %s buffer 0x%" PRIx64 "\n",
                rpc_async_state(ctx->sock.get()).endpoint.c_str(), ctx->remote_ptr);
    }
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_FREE_BUFFER, &request, sizeof(request), nullptr, 0);
    if (!status) {
        // the worker is gone and freed everything with the connection anyway
        rpc_mark_failed(rpc_async_state(ctx->sock.get()).endpoint, __func__);
    }
    delete ctx;
}

static void * ggml_backend_rpc_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    if (ctx->base_ptr != nullptr) {
        return ctx->base_ptr;
    }
    rpc_msg_buffer_get_base_req request = {ctx->remote_ptr};
    rpc_msg_buffer_get_base_rsp response;
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_BUFFER_GET_BASE, &request, sizeof(request), &response, sizeof(response));
    if (!status) {
        rpc_mark_failed(rpc_async_state(ctx->sock.get()).endpoint, __func__);
        return nullptr;
    }
    if (response.base_ptr == 0) {
        GGML_LOG_ERROR("[%s] worker %s returned NULL base for buffer 0x%" PRIx64 " - the worker no longer knows this buffer\n",
                __func__, rpc_async_state(ctx->sock.get()).endpoint.c_str(), ctx->remote_ptr);
    }
    ctx->base_ptr = reinterpret_cast<void *>(response.base_ptr);
    return ctx->base_ptr;
}

static bool ggml_backend_buffer_is_rpc(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_rpc_buffer_free_buffer;
}

static rpc_tensor serialize_tensor(const ggml_tensor * tensor) {
    rpc_tensor result;
    if (!tensor) {
        memset(&result, 0, sizeof(result));
        return result;
    }

    result.id = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
    if (tensor->buffer && ggml_backend_buffer_is_rpc(tensor->buffer)) {
        ggml_backend_buffer_t buffer = tensor->buffer;
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        result.buffer = ctx != nullptr ? ctx->remote_ptr : 0;
        result.data = reinterpret_cast<uint64_t>(tensor->data);
    } else {
        result.buffer = 0;
        result.data   = 0;
    }
    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result.ne[i] = tensor->ne[i];
        result.nb[i] = tensor->nb[i];
    }
    result.op = tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result.op_params[i] = tensor->op_params[i];
    }
    result.flags = tensor->flags;
    for (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        result.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    result.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;

    // Avoid sending uninitialized data over the wire
    memset(result.name, 0, sizeof(result.name));
    memset(result.padding, 0, sizeof(result.padding));

    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
    return result;
}

static enum ggml_status ggml_backend_rpc_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;

    // CUDA backend on the server pads everything to 512 due to CUDA limitations.
    // Due to bandwidth constraints, we only call the server init tensor functions if necessary.
    // In particular, only quantized tensors need padding
    if (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr)) {
        rpc_msg_init_tensor_req request;

        request.tensor = serialize_tensor(tensor);

        if (buffer->usage != GGML_BACKEND_BUFFER_USAGE_COMPUTE && ctx->base_ptr != nullptr) {
            std::lock_guard<std::mutex> lock(g_rpc_journal_mutex);
            rpc_tensor jt = request.tensor;
            jt.data -= (uint64_t)(uintptr_t) ctx->base_ptr; // offset from base for replay
            g_rpc_journals[buffer].init_tensors.push_back(jt);
        }

        bool status = send_rpc_cmd(ctx->sock, RPC_CMD_INIT_TENSOR, &request, sizeof(request), nullptr, 0);
        if (!status) {
            rpc_mark_failed(rpc_async_state(ctx->sock.get()).endpoint, __func__);
            return GGML_STATUS_FAILED;
        }
    }
    return GGML_STATUS_SUCCESS;
}

// raw byte-range get/set on a view touches the same memory as its root tensor at a
// translated offset. Views must be resolved to the root before serializing: view links
// do not survive the wire, and a tensor-parallel island worker can only re-derive the
// per-device placement of tensors it has registered - a bare view struct would fall
// back to mirrored placement and silently corrupt the packed per-device slices.
static const ggml_tensor * rpc_resolve_view(const ggml_tensor * tensor, size_t & offset) {
    while (tensor->view_src != nullptr && tensor->view_src->buffer == tensor->buffer) {
        offset += (const char *) tensor->data - (const char *) tensor->view_src->data;
        tensor = tensor->view_src;
    }
    return tensor;
}

// source provenance for weight uploads (proto 4.8, TASKS.md #44): the caller (the
// meta backend's weight splitter, via the registry proc) announces which GGUF
// tensor the data pointer of subsequent set_tensor(_2d) calls aliases. Row
// geometry is derived per call; a wrong hint hash-misses on the worker and
// degrades to streaming. Thread-local: weight loads are synchronous.
struct rpc_src_ref {
    uint64_t src_offset;
    uint64_t row_size;
    uint64_t n_rows;
    uint64_t row_stride;
};

static thread_local struct {
    char     name[GGML_MAX_NAME];
    uint64_t base;
    bool     valid;
} g_rpc_src_hint;

void ggml_backend_rpc_source_hint(const char * name, uint64_t base_offset) {
    static const bool disabled = getenv("GGML_RPC_NO_SRC_HINT") != nullptr;
    if (disabled || name == nullptr || name[0] == '\0') {
        g_rpc_src_hint.valid = false;
        return;
    }
    snprintf(g_rpc_src_hint.name, sizeof(g_rpc_src_hint.name), "%s", name);
    g_rpc_src_hint.base  = base_offset;
    g_rpc_src_hint.valid = true;
}

// record a completed weight-class write so a restarted worker can be re-provisioned
// without the coordinator re-reading the model (big writes replay by hash against
// the worker's local cache, small writes replay literally, bounded)
static void rpc_journal_record_set(ggml_backend_buffer_t buffer, const rpc_tensor & root, uint64_t offset,
                                   const void * data, size_t size, uint64_t hash, bool have_hash,
                                   const rpc_src_ref * src = nullptr) {
    if (buffer->usage == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        return;
    }
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    if (ctx->base_ptr == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_rpc_journal_mutex);
    rpc_buffer_journal & j = g_rpc_journals[buffer];
    if (j.disabled) {
        return;
    }
    rpc_buffer_journal::set_entry e;
    e.root      = root;
    e.root.data = root.data - (uint64_t)(uintptr_t) ctx->base_ptr;
    e.offset    = offset;
    e.size      = size;
    e.hash      = hash;
    e.spill_off = UINT64_MAX;
    if (src != nullptr && have_hash && g_rpc_src_hint.valid) {
        snprintf(e.src_name, sizeof(e.src_name), "%s", g_rpc_src_hint.name);
        e.src_offset     = src->src_offset;
        e.src_row_size   = src->row_size;
        e.src_n_rows     = src->n_rows;
        e.src_row_stride = src->row_stride;
    }
    if (!have_hash) {
        if (j.spill == nullptr) {
            j.spill = tmpfile();
        }
        if (j.spill == nullptr || j.spill_size + size > rpc_journal_small_budget()) {
            GGML_LOG_WARN("[rpc journal] small-write spill unavailable or budget exceeded - "
                          "surgical re-provision disabled for a buffer (GGML_RPC_JOURNAL_MAX_MIB)\n");
            j.disabled = true;
            j.sets.clear();
            j.init_tensors.clear();
            return;
        }
        if (fseeko(j.spill, (off_t) j.spill_size, SEEK_SET) != 0 ||
            fwrite(data, 1, size, j.spill) != size) {
            j.disabled = true;
            j.sets.clear();
            j.init_tensors.clear();
            return;
        }
        e.spill_off   = j.spill_size;
        j.spill_size += size;
    }
    // buffer-relative start: distinct tensors at the same tensor-relative offset
    // must not collide (a rewrite of the same region still dedups to one entry)
    const uint64_t key = e.root.data + offset;
    j.sets[key] = std::move(e);
}

static void rpc_buffer_set_tensor_impl(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data,
                                       size_t offset, size_t size, const rpc_src_ref * src) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    const ggml_tensor * root = rpc_resolve_view(tensor, offset);
    rpc_tensor rpc_tensor = serialize_tensor(root);
    if (size > HASH_THRESHOLD) {
        const uint64_t hash = fnv_hash((const uint8_t*)data, size);
        const bool use_src  = src != nullptr && g_rpc_src_hint.valid
                           && rpc_async_state(ctx->sock.get()).server_minor >= 8;
        rpc_journal_record_set(buffer, rpc_tensor, offset, data, size, hash, true, use_src ? src : nullptr);
        rpc_msg_set_tensor_hash_rsp response;
        bool status;
        if (use_src) {
            rpc_msg_set_tensor_hash2_req request;
            memset(&request, 0, sizeof(request));
            request.tensor         = rpc_tensor;
            request.offset         = offset;
            request.hash           = hash;
            request.src_offset     = src->src_offset;
            request.src_row_size   = src->row_size;
            request.src_n_rows     = src->n_rows;
            request.src_row_stride = src->row_stride;
            snprintf(request.src_name, sizeof(request.src_name), "%s", g_rpc_src_hint.name);
            status = send_rpc_cmd(ctx->sock, RPC_CMD_SET_TENSOR_HASH2, &request, sizeof(request), &response, sizeof(response));
        } else {
            rpc_msg_set_tensor_hash_req request;
            request.tensor = rpc_tensor;
            request.offset = offset;
            request.hash   = hash;
            status = send_rpc_cmd(ctx->sock, RPC_CMD_SET_TENSOR_HASH, &request, sizeof(request), &response, sizeof(response));
        }
        if (!status) {
            rpc_mark_failed(rpc_async_state(ctx->sock.get()).endpoint, __func__);
            return;
        }
        if (response.result) {
            // the server has the same data, no need to send it
            if (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                rpc_ep_stat * st = rpc_async_state(ctx->sock.get()).stat;
                if (st != nullptr) {
                    st->weights_cached_bytes.fetch_add(size, std::memory_order_relaxed);
                }
            }
            return;
        }
    } else {
        rpc_journal_record_set(buffer, rpc_tensor, offset, data, size, 0, false);
    }
    if (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        rpc_ep_stat * st = rpc_async_state(ctx->sock.get()).stat;
        if (st != nullptr) {
            st->weights_streamed_bytes.fetch_add(size, std::memory_order_relaxed);
        }
    }
    // input serialization format: | rpc_tensor | offset (8 bytes) | data (size bytes)
    size_t input_size = sizeof(rpc_tensor) + sizeof(uint64_t) + size;
    std::vector<uint8_t> input(input_size, 0);
    memcpy(input.data(), &rpc_tensor, sizeof(rpc_tensor));
    memcpy(input.data() + sizeof(rpc_tensor), &offset, sizeof(offset));
    memcpy(input.data() + sizeof(rpc_tensor) + sizeof(offset), data, size);
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_SET_TENSOR, input.data(), input.size());
    if (!status) {
        rpc_mark_failed(rpc_async_state(ctx->sock.get()).endpoint, __func__);
    }
}

static void ggml_backend_rpc_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (g_rpc_src_hint.valid) {
        const rpc_src_ref src = { g_rpc_src_hint.base, (uint64_t) size, 1, 0 };
        rpc_buffer_set_tensor_impl(buffer, tensor, data, offset, size, &src);
    } else {
        rpc_buffer_set_tensor_impl(buffer, tensor, data, offset, size, nullptr);
    }
}

// strided host rows into a contiguous device range: gather client-side and upload in
// large chunks instead of one SET_TENSOR per row (the meta backend loads sliced
// weights this way - per-row uploads were ~12M round trips per model load)
static void ggml_backend_rpc_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data,
                                                  size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    const bool hinted = g_rpc_src_hint.valid;
    if (stride_tensor != size) {
        // destination rows are not contiguous - no batching possible
        for (size_t i = 0; i < n_copies; i++) {
            const rpc_src_ref src = { g_rpc_src_hint.base + i*stride_data, (uint64_t) size, 1, 0 };
            rpc_buffer_set_tensor_impl(buffer, tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size,
                                       hinted ? &src : nullptr);
        }
        return;
    }
    constexpr size_t max_chunk = 32u*1024*1024;
    const size_t rows_per_chunk = std::max<size_t>(1, max_chunk/size);
    std::vector<uint8_t> tmp(std::min(n_copies, rows_per_chunk)*size);
    for (size_t i0 = 0; i0 < n_copies; i0 += rows_per_chunk) {
        const size_t n = std::min(rows_per_chunk, n_copies - i0);
        for (size_t k = 0; k < n; k++) {
            memcpy(tmp.data() + k*size, (const char *) data + (i0 + k)*stride_data, size);
        }
        const rpc_src_ref src = { g_rpc_src_hint.base + i0*stride_data, (uint64_t) size, n, stride_data };
        rpc_buffer_set_tensor_impl(buffer, tensor, tmp.data(), offset + i0*stride_tensor, n*size,
                                   hinted ? &src : nullptr);
    }
}

static void ggml_backend_rpc_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    const ggml_tensor * root = rpc_resolve_view(tensor, offset);
    rpc_msg_get_tensor_req request;
    request.tensor = serialize_tensor(root);
    request.offset = offset;
    request.size = size;
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_GET_TENSOR, &request, sizeof(request), data, size);
    if (!status) {
        rpc_mark_failed(rpc_async_state(ctx->sock.get()).endpoint, __func__);
        memset(data, 0, size); // deterministic instead of stale garbage
    }
}

static bool ggml_backend_rpc_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_rpc(src->buffer)) {
        // check if src and dst are on the same server
        ggml_backend_buffer_t src_buffer = src->buffer;
        ggml_backend_rpc_buffer_context * src_ctx = (ggml_backend_rpc_buffer_context *)src_buffer->context;
        ggml_backend_buffer_t dst_buffer = dst->buffer;
        ggml_backend_rpc_buffer_context * dst_ctx = (ggml_backend_rpc_buffer_context *)dst_buffer->context;
        if (src_ctx->sock != dst_ctx->sock) {
            return false;
        }
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        rpc_msg_copy_tensor_req request;
        request.src = serialize_tensor(src);
        request.dst = serialize_tensor(dst);
        rpc_msg_copy_tensor_rsp response;
        bool status = send_rpc_cmd(ctx->sock, RPC_CMD_COPY_TENSOR, &request, sizeof(request), &response, sizeof(response));
        if (!status) {
            rpc_mark_failed(rpc_async_state(ctx->sock.get()).endpoint, __func__);
            return false;
        }
        return response.result;
    }
    return false;
}

static void ggml_backend_rpc_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    if (buffer->usage != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        std::lock_guard<std::mutex> lock(g_rpc_journal_mutex);
        rpc_buffer_journal & j = g_rpc_journals[buffer];
        j.cleared     = true;
        j.clear_value = value;
    }
    rpc_msg_buffer_clear_req request = {ctx->remote_ptr, value};
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_BUFFER_CLEAR, &request, sizeof(request), nullptr, 0);
    if (!status) {
        rpc_mark_failed(rpc_async_state(ctx->sock.get()).endpoint, __func__);
    }
}

static void ggml_backend_rpc_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    // mirror the usage on the remote buffer - the worker-side meta (tensor-parallel)
    // device relies on it to distinguish weight from compute tensors
    rpc_msg_buffer_set_usage_req request = { ctx->remote_ptr, (uint32_t) usage };
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_BUFFER_SET_USAGE, &request, sizeof(request), nullptr, 0);
    if (!status) {
        GGML_LOG_ERROR("failed to set remote buffer usage\n");
    }
}

static ggml_backend_buffer_i ggml_backend_rpc_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_rpc_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_rpc_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_rpc_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_rpc_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_rpc_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_rpc_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_rpc_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_rpc_buffer_clear,
    /* .reset           = */ NULL,
    /* .set_usage       = */ ggml_backend_rpc_buffer_set_usage,
};

static const char * ggml_backend_rpc_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_rpc_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    // fault injection: GGML_RPC_DEBUG_FAIL_ALLOC=ENDPOINT:SKIP:COUNT skips the
    // first SKIP allocations on ENDPOINT, fails the next COUNT, passes the rest
    // (and logs every request) - reproduces a worker rejecting a specific
    // buffer allocation (e.g. the pipeline-parallel compute buffer)
    static const char * fail_env = getenv("GGML_RPC_DEBUG_FAIL_ALLOC");
    if (fail_env != nullptr) {
        static std::mutex fail_mutex;
        static std::string fail_ep;
        static int64_t fail_skip  = 0;
        static int64_t fail_count = 0;
        static int64_t seen       = 0;
        bool fail = false;
        {
            std::lock_guard<std::mutex> lock(fail_mutex);
            if (fail_ep.empty()) {
                // parse from the right: the endpoint itself contains a colon
                std::string spec = fail_env;
                size_t c2 = spec.rfind(':');
                size_t c1 = c2 != std::string::npos ? spec.rfind(':', c2 - 1) : std::string::npos;
                if (c1 != std::string::npos) {
                    fail_ep    = spec.substr(0, c1);
                    fail_skip  = atoll(spec.c_str() + c1 + 1);
                    fail_count = atoll(spec.c_str() + c2 + 1);
                }
            }
            if (buft_ctx->endpoint == fail_ep) {
                fail = seen >= fail_skip && seen < fail_skip + fail_count;
                seen++;
            }
        }
        GGML_LOG_INFO("[rpc-alloc-debug] %s ep=%s size=%zu%s\n",
                      buft_ctx->name.c_str(), buft_ctx->endpoint.c_str(), size, fail ? " INJECTED-FAIL" : "");
        if (fail) {
            return nullptr;
        }
    }
    rpc_msg_alloc_buffer_req request = {buft_ctx->device, size};
    rpc_msg_alloc_buffer_rsp response;
    auto sock = get_socket(buft_ctx->endpoint);
    if (sock == nullptr) {
        rpc_mark_failed(buft_ctx->endpoint, __func__);
        return nullptr;
    }
    bool status = send_rpc_cmd(sock, RPC_CMD_ALLOC_BUFFER, &request, sizeof(request), &response, sizeof(response));
    if (!status) {
        rpc_mark_failed(buft_ctx->endpoint, __func__);
        return nullptr;
    }
    if (response.remote_ptr != 0) {
        if (getenv("GGML_RPC_DEBUG_BUF") != nullptr) {
            GGML_LOG_INFO("[rpc-buf-debug] alloc %s buffer 0x%" PRIx64 " size %zu\n",
                    buft_ctx->endpoint.c_str(), response.remote_ptr, size);
        }
        ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft,
            ggml_backend_rpc_buffer_interface,
            new ggml_backend_rpc_buffer_context{sock, nullptr, response.remote_ptr},
            response.remote_size);
        return buffer;
    } else {
        return nullptr;
    }
}

static size_t get_alignment(const std::shared_ptr<socket_t> & sock, uint32_t device) {
    rpc_msg_get_alignment_req request = {device};
    rpc_msg_get_alignment_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_ALIGNMENT, &request, sizeof(request), &response, sizeof(response));
    if (!status) {
        rpc_mark_failed(rpc_async_state(sock.get()).endpoint, __func__);
        return 64; // ggml base alignment; the endpoint is poisoned anyway
    }
    return response.alignment;
}

static size_t ggml_backend_rpc_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->alignment;
}

static size_t get_max_size(const std::shared_ptr<socket_t> & sock, uint32_t device) {
    rpc_msg_get_max_size_req request = {device};
    rpc_msg_get_max_size_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_MAX_SIZE, &request, sizeof(request), &response, sizeof(response));
    if (!status) {
        rpc_mark_failed(rpc_async_state(sock.get()).endpoint, __func__);
        return SIZE_MAX;
    }
    return response.max_size;
}

static size_t ggml_backend_rpc_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->max_size;
}

static size_t ggml_backend_rpc_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    // should we query the remote server for the actual size
    bool rpc_get = false;

    // See comments in init_tensor.
    rpc_get |= ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr);

    // ops that require additional memory for fleeting data on certain backends
    // ref: https://github.com/ggml-org/llama.cpp/pull/15966
    rpc_get |= tensor->op == GGML_OP_FLASH_ATTN_EXT;
    rpc_get |= tensor->op == GGML_OP_MUL_MAT_ID;

    // tensor-parallel islands store only per-device slices; the actual allocation size
    // depends on the worker-side split state, so it must always be queried
    rpc_get |= buft->device != nullptr &&
        strncmp(ggml_backend_dev_description(buft->device), "Meta[", 5) == 0;

    if (rpc_get) {
        ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
        auto sock = get_socket(buft_ctx->endpoint);
        if (sock == nullptr) {
            rpc_mark_failed(buft_ctx->endpoint, __func__);
            return ggml_nbytes(tensor);
        }

        rpc_msg_get_alloc_size_req request = {
            /*.device =*/ buft_ctx->device,
            /*.tensor =*/ serialize_tensor(tensor),
            /*.srcs   =*/ {},
        };

        // .get_alloc_size could be a function of the tensor's srcs, so we must serialize them as well
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            request.srcs[i] = serialize_tensor(tensor->src[i]);
        }

        // TODO: cache the alloc responses to avoid extra RPC calls?
        rpc_msg_get_alloc_size_rsp response;
        bool status = send_rpc_cmd(sock, RPC_CMD_GET_ALLOC_SIZE, &request, sizeof(request), &response, sizeof(response));
        if (!status) {
            rpc_mark_failed(rpc_async_state(sock.get()).endpoint, __func__);
            return ggml_nbytes(tensor);
        }

        return response.alloc_size;
    }

    return ggml_nbytes(tensor);
}

static ggml_backend_buffer_type_i ggml_backend_rpc_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_rpc_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_rpc_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_rpc_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_rpc_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_rpc_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

static const char * ggml_backend_rpc_name(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    return rpc_ctx->name.c_str();
}

static void ggml_backend_rpc_free(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    delete rpc_ctx;
    delete backend;
}

static void ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    // graph_compute and set_tensor are sent without waiting; a fresh marker's
    // response proves everything sent so far has executed on the server
    auto sock = get_socket(rpc_ctx->endpoint);
    if (sock == nullptr) {
        rpc_mark_failed(rpc_ctx->endpoint, __func__);
        return; // the endpoint is poisoned; graph_compute reports the failure
    }
    rpc_sync_pings(sock, rpc_ping_async(sock));
}

static void ggml_backend_rpc_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor->buffer != nullptr && ggml_backend_buffer_is_rpc(tensor->buffer));
    // the buffer path already sends SET_TENSOR without waiting for a response
    // (the >HASH_THRESHOLD dedup path blocks on one round trip; inputs are small)
    ggml_backend_rpc_buffer_set_tensor(tensor->buffer, tensor, data, offset, size);
    GGML_UNUSED(backend);
}

// a W2W pull that passed all fast-path checks and is ready to send; fence_seq is
// written immediately before sending (the batch path captures ALL fences before
// sending ANY request - see ggml_backend_rpc_cpy_tensor_batch_async)
struct rpc_w2w_pull {
    socket_ptr           sock_src;
    socket_ptr           sock_dst;
    std::string          dst_endpoint;
    std::vector<uint8_t> input;
    size_t               fence_seq_offset = 0;
};

// eligibility checks + request assembly shared by the single and batched W2W copy
// paths; false = this pair must go through the regular (bridged) copy instead
static bool rpc_prepare_w2w_pull(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, const ggml_tensor * dst, rpc_w2w_pull & pull) {
    static const bool disabled = getenv("GGML_RPC_NO_W2W") != nullptr;
    if (disabled) {
        return false; // escape hatch: force the coordinator-bridged copy
    }
    if (backend_src->iface.get_name != ggml_backend_rpc_name ||
        backend_dst->iface.get_name != ggml_backend_rpc_name) {
        return false; // either side is not an RPC backend
    }
    ggml_backend_rpc_context * src_ctx = (ggml_backend_rpc_context *)backend_src->context;
    ggml_backend_rpc_context * dst_ctx = (ggml_backend_rpc_context *)backend_dst->context;
    if (src_ctx->endpoint == dst_ctx->endpoint) {
        return false; // same server: the buffer-level COPY_TENSOR path already handles it
    }
    // endpoint strings are coordinator-relative: a loopback source (e.g. a worker on
    // the coordinator box heard via the loopback beacon) is unreachable from a remote
    // destination - it would dial ITSELF (TASKS.md #39). Bridge those copies.
    if (rpc_endpoint_is_loopback(src_ctx->endpoint) && !rpc_endpoint_is_loopback(dst_ctx->endpoint)) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true)) {
            GGML_LOG_WARN("[w2w] source '%s' is loopback and unreachable from remote workers - using the bridged copy\n",
                          src_ctx->endpoint.c_str());
        }
        return false;
    }
    if (src->buffer == nullptr || dst->buffer == nullptr ||
        !ggml_backend_buffer_is_rpc(src->buffer) || !ggml_backend_buffer_is_rpc(dst->buffer)) {
        return false;
    }
    if (ggml_nbytes(src) != ggml_nbytes(dst)) {
        return false;
    }
    auto sock_src = get_socket(src_ctx->endpoint);
    auto sock_dst = get_socket(dst_ctx->endpoint);
    if (sock_src == nullptr || sock_dst == nullptr) {
        return false; // bridged fallback; that path is failure-contained
    }
    ggml_backend_rpc_async_state & st_src = rpc_async_state(sock_src.get());
    ggml_backend_rpc_async_state & st_dst = rpc_async_state(sock_dst.get());
    if (st_src.server_minor < 2 || st_dst.server_minor < 2) {
        return false; // an old worker on either side: fall back to the bridged copy
    }
    // the tensors must actually live on these endpoints
    if (((ggml_backend_rpc_buffer_context *)src->buffer->context)->sock != sock_src ||
        ((ggml_backend_rpc_buffer_context *)dst->buffer->context)->sock != sock_dst) {
        return false;
    }
    const uint64_t fence_conn = rpc_get_conn_id(sock_src);
    if (fence_conn == 0) {
        return false;
    }
    // views: resolve to root + byte offset, exactly like the get/set paths - view
    // links do not survive the wire
    size_t src_offset = 0;
    size_t dst_offset = 0;
    const ggml_tensor * src_root = rpc_resolve_view(src, src_offset);
    const ggml_tensor * dst_root = rpc_resolve_view(dst, dst_offset);
    const rpc_tensor rsrc = serialize_tensor(src_root);
    const rpc_tensor rdst = serialize_tensor(dst_root);
    const uint64_t size = (uint64_t) ggml_nbytes(src);
    const std::string & ep = src_ctx->endpoint;
    const uint32_t ep_len = (uint32_t) ep.size();
    const uint64_t fence_seq = 0; // placeholder, written at send time

    // | rpc_tensor src | rpc_tensor dst | src_offset | dst_offset | size | fence_conn | fence_seq | ep_len | ep |
    std::vector<uint8_t> input(2*sizeof(rpc_tensor) + 5*sizeof(uint64_t) + sizeof(uint32_t) + ep_len);
    uint8_t * p = input.data();
    memcpy(p, &rsrc, sizeof(rsrc));                p += sizeof(rsrc);
    memcpy(p, &rdst, sizeof(rdst));                p += sizeof(rdst);
    uint64_t off64 = src_offset; memcpy(p, &off64, sizeof(off64)); p += sizeof(off64);
    off64 = dst_offset;          memcpy(p, &off64, sizeof(off64)); p += sizeof(off64);
    memcpy(p, &size, sizeof(size));                p += sizeof(size);
    memcpy(p, &fence_conn, sizeof(fence_conn));    p += sizeof(fence_conn);
    pull.fence_seq_offset = p - input.data();
    memcpy(p, &fence_seq, sizeof(fence_seq));      p += sizeof(fence_seq);
    memcpy(p, &ep_len, sizeof(ep_len));            p += sizeof(ep_len);
    memcpy(p, ep.data(), ep_len);

    pull.sock_src     = sock_src;
    pull.sock_dst     = sock_dst;
    pull.dst_endpoint = dst_ctx->endpoint;
    pull.input        = std::move(input);
    return true;
}

// everything sent to the source so far includes the compute producing src
static void rpc_w2w_pull_set_fence(rpc_w2w_pull & pull) {
    ggml_backend_rpc_async_state & st_src = rpc_async_state(pull.sock_src.get());
    std::lock_guard<std::mutex> lock(st_src.mutex);
    memcpy(pull.input.data() + pull.fence_seq_offset, &st_src.cmds_sent, sizeof(st_src.cmds_sent));
}

// worker-to-worker activation transfer (proto 4.2): instead of bridging the bytes
// through the coordinator (blocking GET from the source, then SET to the
// destination), tell the destination worker to pull directly from the source.
// The pull is fenced so it cannot read before the source has executed the compute
// that produces the data (read-after-write); the blocking ack means the
// coordinator queues no further source work until the read is out
// (write-after-read) - the same contract the CUDA cross-device copy implements
// with events, while the transfer itself takes one direct hop.
static bool ggml_backend_rpc_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    rpc_w2w_pull pull;
    if (!rpc_prepare_w2w_pull(backend_src, backend_dst, src, dst, pull)) {
        return false;
    }
    rpc_w2w_pull_set_fence(pull);

    rpc_msg_copy_from_remote_rsp response;
    bool status = send_rpc_cmd(pull.sock_dst, RPC_CMD_COPY_FROM_REMOTE, pull.input.data(), pull.input.size(), &response, sizeof(response));
    if (!status) {
        rpc_mark_failed(pull.dst_endpoint, __func__);
        return false; // the bridged fallback will fail-contained on the same endpoint
    }
    // a failed pull (e.g. the destination worker cannot reach the source) falls
    // back to the coordinator-bridged copy
    return response.result != 0;
}

// batched W2W pulls (the meta backend's reduce boundaries): capture every fence
// BEFORE sending any request, then send all requests, then collect the acks. The
// single-copy path above serializes opposite-direction pulls on their acks; a
// fence captured up front is safe (it already covers the compute that produced
// its src - dispatched before the reduce) and keeps each pull's request out of
// its sibling's fence, which is exactly the chain that forced them sequential.
static void ggml_backend_rpc_cpy_tensor_batch_async(int n_copies, ggml_backend_t * backends_src, ggml_backend_t * backends_dst, ggml_tensor ** srcs, ggml_tensor ** dsts) {
    std::vector<rpc_w2w_pull> pulls(n_copies);
    std::vector<bool>         fast(n_copies, false);
    for (int k = 0; k < n_copies; k++) {
        fast[k] = rpc_prepare_w2w_pull(backends_src[k], backends_dst[k], srcs[k], dsts[k], pulls[k]);
        // a 4.3 server holds its execution lock across the whole pull, so two
        // crossing in-flight pulls ABBA-deadlock; batch only when both ends
        // release the lock during the fetch (proto 4.4)
        if (fast[k] &&
            (rpc_async_state(pulls[k].sock_src.get()).server_minor < 4 ||
             rpc_async_state(pulls[k].sock_dst.get()).server_minor < 4)) {
            fast[k] = false;
        }
        // one in-flight request per destination socket keeps the response
        // bookkeeping trivial; a duplicate destination takes the blocking path
        for (int m = 0; fast[k] && m < k; m++) {
            if (fast[m] && pulls[m].sock_dst == pulls[k].sock_dst) {
                fast[k] = false;
            }
        }
    }
    std::vector<int> order;
    for (int k = 0; k < n_copies; k++) {
        if (fast[k]) {
            order.push_back(k);
            rpc_w2w_pull_set_fence(pulls[k]);
        }
    }
    {
        // per-socket locks are held from send to ack so no other traffic can
        // interleave; sorted acquisition (sockets are distinct, see above)
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return pulls[a].sock_dst.get() < pulls[b].sock_dst.get();
        });
        std::vector<std::unique_lock<std::mutex>> locks;
        locks.reserve(order.size());
        for (int k : order) {
            ggml_backend_rpc_async_state & st = rpc_async_state(pulls[k].sock_dst.get());
            locks.emplace_back(st.mutex);
            LOG_DBG("[w2w_batch] send k=%d dst=%s pings=%" PRIu64 "/%" PRIu64 " cmds=%" PRIu64 "\n",
                    k, pulls[k].dst_endpoint.c_str(), st.pings_done, st.pings_sent, st.cmds_sent);
            if (!rpc_drain_pings_locked(pulls[k].sock_dst, st, st.pings_sent) ||
                !send_rpc_cmd_raw(pulls[k].sock_dst, st, RPC_CMD_COPY_FROM_REMOTE, pulls[k].input.data(), pulls[k].input.size())) {
                rpc_mark_failed(pulls[k].dst_endpoint, __func__);
                fast[k] = false;
            }
            LOG_DBG("[w2w_batch] sent k=%d ok=%d\n", k, fast[k] ? 1 : 0);
        }
        for (int k : order) {
            if (!fast[k]) {
                continue;
            }
            LOG_DBG("[w2w_batch] recv k=%d dst=%s\n", k, pulls[k].dst_endpoint.c_str());
            uint64_t out_size;
            rpc_msg_copy_from_remote_rsp response;
            if (!pulls[k].sock_dst->recv_data(&out_size, sizeof(out_size)) ||
                out_size != sizeof(response) ||
                !pulls[k].sock_dst->recv_data(&response, sizeof(response))) {
                rpc_mark_failed(pulls[k].dst_endpoint, __func__);
                fast[k] = false;
                continue;
            }
            LOG_DBG("[w2w_batch] ack k=%d result=%d\n", k, (int) response.result);
            if (response.result == 0) {
                fast[k] = false; // soft failure: bridged fallback below
            }
        }
    }
    for (int k = 0; k < n_copies; k++) {
        if (!fast[k]) {
            ggml_backend_tensor_copy_async(backends_src[k], backends_dst[k], srcs[k], dsts[k]);
        }
    }
}

// batched blocking reads (the meta backend's star reduce): every request goes out
// before any response is read, so the servers' compute tails and the transfers
// overlap instead of paying one round trip per tensor. In-order serving makes each
// GET an implicit fence on the compute that produced its tensor - no explicit
// fence and no W2W machinery needed.
static void ggml_backend_rpc_get_tensor_batch(int n_gets, ggml_backend_t * backends, const ggml_tensor ** tensors, void ** datas, size_t * sizes) {
    struct get_req {
        socket_ptr             sock;
        std::string            endpoint;
        rpc_msg_get_tensor_req request;
    };
    std::vector<get_req> reqs(n_gets);
    std::vector<bool>    fast(n_gets, false);
    for (int k = 0; k < n_gets; k++) {
        if (backends[k]->iface.get_name != ggml_backend_rpc_name) {
            continue;
        }
        const ggml_tensor * t = tensors[k];
        if (t->buffer == nullptr || !ggml_backend_buffer_is_rpc(t->buffer)) {
            continue;
        }
        ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backends[k]->context;
        auto sock = get_socket(rpc_ctx->endpoint);
        if (sock == nullptr || ((ggml_backend_rpc_buffer_context *) t->buffer->context)->sock != sock) {
            continue;
        }
        size_t offset = 0;
        const ggml_tensor * root = rpc_resolve_view(t, offset);
        reqs[k].sock           = sock;
        reqs[k].endpoint       = rpc_ctx->endpoint;
        reqs[k].request.tensor = serialize_tensor(root);
        reqs[k].request.offset = offset;
        reqs[k].request.size   = sizes[k];
        fast[k] = true;
    }
    // same-socket requests stay pipelined (the server answers in order); distinct
    // sockets are sorted so the per-socket locks are acquired consistently
    std::vector<int> order;
    for (int k = 0; k < n_gets; k++) {
        if (fast[k]) {
            order.push_back(k);
        }
    }
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return reqs[a].sock.get() < reqs[b].sock.get();
    });
    {
        std::vector<std::unique_lock<std::mutex>> locks;
        locks.reserve(order.size());
        socket_t * locked  = nullptr;
        bool       sock_ok = true;
        for (int k : order) {
            ggml_backend_rpc_async_state & st = rpc_async_state(reqs[k].sock.get());
            if (reqs[k].sock.get() != locked) {
                locks.emplace_back(st.mutex);
                locked  = reqs[k].sock.get();
                sock_ok = rpc_drain_pings_locked(reqs[k].sock, st, st.pings_sent);
            }
            if (!sock_ok ||
                !send_rpc_cmd_raw(reqs[k].sock, st, RPC_CMD_GET_TENSOR, &reqs[k].request, sizeof(reqs[k].request))) {
                rpc_mark_failed(reqs[k].endpoint, __func__);
                sock_ok = false;
                fast[k] = false;
            }
        }
        for (int k : order) {
            if (!fast[k]) {
                continue;
            }
            uint64_t out_size;
            if (!reqs[k].sock->recv_data(&out_size, sizeof(out_size)) ||
                out_size != sizes[k] ||
                !reqs[k].sock->recv_data(datas[k], sizes[k])) {
                rpc_mark_failed(reqs[k].endpoint, __func__);
                memset(datas[k], 0, sizes[k]); // deterministic instead of stale garbage
            }
        }
    }
    for (int k = 0; k < n_gets; k++) {
        if (!fast[k]) {
            ggml_backend_tensor_get(tensors[k], datas[k], 0, sizes[k]);
        }
    }
}

// fused boundary command, client side (proto 4.5): one message carries the
// previous boundary's reduced value (SET), the next subgraph's recompute-by-uid
// (GRAPH) and the request for its boundary partial (FETCH) whose bytes arrive
// via ..._fused_recv. Returns false when the fast path is unavailable (old
// server, uid not yet uploaded, non-RPC tensors) - the caller falls back to the
// unfused commands. IMPORTANT contract for callers: while a fused response is
// pending on a connection, do not synchronize that backend (a PING would read
// the pending response frame and desync the stream) - pair every send that
// requested a FETCH with exactly one recv before any other blocking call.
static bool ggml_backend_rpc_boundary_fused_send(ggml_backend_t backend,
        ggml_tensor * set_tensor, const void * set_data, size_t set_size,
        struct ggml_cgraph ** cgraphs, int n_graphs, const ggml_tensor * fetch_tensor, size_t fetch_size) {
    if (backend->iface.get_name != ggml_backend_rpc_name) {
        return false;
    }
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backend->context;
    if (rpc_endpoint_failed(rpc_ctx->endpoint)) {
        return false;
    }
    auto sock = get_socket(rpc_ctx->endpoint);
    // 4.6, not 4.5: the chain format (uid LIST in the GRAPH segment) replaced the
    // short-lived single-uid 4.5 layout - a 4.5 server would misparse it
    if (sock == nullptr || rpc_async_state(sock.get()).server_minor < 6) {
        return false;
    }
    ggml_backend_dev_t rpc_dev = ggml_backend_get_device(backend);
    ggml_backend_rpc_device_context * rpc_dev_ctx = (ggml_backend_rpc_device_context *) rpc_dev->context;

    uint32_t flags = 0;
    // GRAPH chain: only uids the server already caches can be carried (same
    // invariant as GRAPH_RECOMPUTE_UID); fresh graphs go through the upload path
    if (n_graphs > 0) {
        if (cgraphs == nullptr || rpc_dev_ctx->sent_graph_uids_sock != sock.get()) {
            return false;
        }
        for (int g = 0; g < n_graphs; g++) {
            if (cgraphs[g] == nullptr || cgraphs[g]->uid == 0 ||
                rpc_dev_ctx->sent_graph_uids.count(cgraphs[g]->uid) == 0) {
                return false;
            }
        }
        flags |= 2;
    }
    rpc_tensor set_rt = {}, fetch_rt = {};
    uint64_t set_off = 0, fetch_off = 0;
    if (set_tensor != nullptr) {
        if (set_tensor->buffer == nullptr || !ggml_backend_buffer_is_rpc(set_tensor->buffer) ||
            ((ggml_backend_rpc_buffer_context *) set_tensor->buffer->context)->sock != sock) {
            return false;
        }
        size_t off = 0;
        set_rt  = serialize_tensor(rpc_resolve_view(set_tensor, off));
        set_off = off;
        flags |= 1;
    }
    if (fetch_tensor != nullptr) {
        if (fetch_tensor->buffer == nullptr || !ggml_backend_buffer_is_rpc(fetch_tensor->buffer) ||
            ((ggml_backend_rpc_buffer_context *) fetch_tensor->buffer->context)->sock != sock) {
            return false;
        }
        size_t off = 0;
        fetch_rt  = serialize_tensor(rpc_resolve_view(fetch_tensor, off));
        fetch_off = off;
        flags |= 4;
    }
    if (flags == 0) {
        return false;
    }

    std::vector<uint8_t> input;
    input.reserve(2*sizeof(uint32_t) + 2*sizeof(rpc_tensor) + 5*sizeof(uint64_t) + set_size);
    auto put = [&](const void * p, size_t n) {
        input.insert(input.end(), (const uint8_t *) p, (const uint8_t *) p + n);
    };
    put(&rpc_ctx->device, sizeof(uint32_t));
    put(&flags, sizeof(flags));
    if (flags & 1) {
        uint64_t sz = set_size;
        put(&set_rt, sizeof(set_rt));
        put(&set_off, sizeof(set_off));
        put(&sz, sizeof(sz));
        put(set_data, set_size);
    }
    if (flags & 2) {
        uint32_t n_uids = (uint32_t) n_graphs;
        put(&n_uids, sizeof(n_uids));
        for (int g = 0; g < n_graphs; g++) {
            uint64_t uid = cgraphs[g]->uid;
            put(&uid, sizeof(uid));
        }
    }
    if (flags & 4) {
        uint64_t sz = fetch_size;
        put(&fetch_rt, sizeof(fetch_rt));
        put(&fetch_off, sizeof(fetch_off));
        put(&sz, sizeof(sz));
    }

    ggml_backend_rpc_async_state & st = rpc_async_state(sock.get());
    std::lock_guard<std::mutex> lock(st.mutex);
    if (!rpc_drain_pings_locked(sock, st, st.pings_sent) ||
        !send_rpc_cmd_raw(sock, st, RPC_CMD_GRAPH_FUSED, input.data(), input.size())) {
        rpc_mark_failed(rpc_ctx->endpoint, __func__);
        return false;
    }
    return true;
}

static bool ggml_backend_rpc_boundary_fused_recv(ggml_backend_t backend, void * data, size_t size) {
    GGML_ASSERT(backend->iface.get_name == ggml_backend_rpc_name);
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backend->context;
    auto sock = get_socket(rpc_ctx->endpoint);
    if (sock == nullptr) {
        rpc_mark_failed(rpc_ctx->endpoint, __func__);
        memset(data, 0, size);
        return false;
    }
    ggml_backend_rpc_async_state & st = rpc_async_state(sock.get());
    std::lock_guard<std::mutex> lock(st.mutex);
    uint64_t out_size;
    if (!sock->recv_data(&out_size, sizeof(out_size)) ||
        out_size != size ||
        !sock->recv_data(data, size)) {
        rpc_mark_failed(rpc_ctx->endpoint, __func__);
        memset(data, 0, size); // deterministic instead of stale garbage
        return false;
    }
    return true;
}

// events (proto 4.2): an event is a PING sequence number on the endpoint's
// connection. Record = send a marker; synchronize/wait = drain to its response.
// Waiting on an event of the SAME endpoint is a no-op: the server executes
// commands in order, so later work is already serialized behind the marker.
struct ggml_backend_rpc_event_context {
    std::string endpoint;
    uint64_t    seq; // 0 = not recorded yet
};

static void ggml_backend_rpc_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    ggml_backend_rpc_event_context * ectx = (ggml_backend_rpc_event_context *)event->context;
    ectx->endpoint = rpc_ctx->endpoint;
    auto sock = get_socket(rpc_ctx->endpoint);
    if (sock == nullptr) {
        rpc_mark_failed(rpc_ctx->endpoint, __func__);
        ectx->seq = 0; // never recorded; waits on it are no-ops
        return;
    }
    ectx->seq = rpc_ping_async(sock);
}

static void ggml_backend_rpc_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    ggml_backend_rpc_event_context * ectx = (ggml_backend_rpc_event_context *)event->context;
    if (ectx->seq == 0) {
        return;
    }
    if (ectx->endpoint == rpc_ctx->endpoint) {
        return; // same connection: ordered by the server
    }
    // cross-endpoint: the coordinator is the only bridge - block until the event's
    // server has executed past the marker before issuing more work here
    auto sock = get_socket(ectx->endpoint);
    if (sock == nullptr) {
        rpc_mark_failed(ectx->endpoint, __func__);
        return;
    }
    rpc_sync_pings(sock, ectx->seq);
}


// serialize a tensor as a pure data source: type/shape/address only. Used for
// tensors a graph references but does not compute - their op history must not be
// re-serialized (walking src chains across subgraph boundaries made every subgraph
// message carry the closure of the whole model graph: O(n^2) bytes per token for
// the meta backend's per-boundary subgraphs). The data of such a tensor is always
// current server-side: whatever produced it ran earlier on this same connection.
static rpc_tensor serialize_tensor_leaf(const ggml_tensor * tensor) {
    rpc_tensor result = serialize_tensor(tensor);
    result.op    = GGML_OP_NONE;
    result.flags = 0;
    memset(result.op_params, 0, sizeof(result.op_params));
    for (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        result.src[i] = 0;
    }
    result.view_src  = 0;
    result.view_offs = 0;
    return result;
}

// The scheduler skips view ops when splitting (they execute as no-ops on every
// backend), so a view of another backend's tensor can sit positionally inside
// this split's node range - e.g. DSV4 creates state reshapes for ALL layers up
// front and views of the previous layer's l_out. All same-split consumers were
// rewired to local input copies, so such nodes are dead weight here - but their
// foreign addresses would make the worker reject the whole graph (TASKS.md #37).
static bool rpc_is_foreign_view_node(const ggml_tensor * node, const std::string & endpoint, uint32_t device) {
    const bool is_view_op = node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE ||
                            node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE;
    if (!is_view_op) {
        return false;
    }
    ggml_backend_buffer_t buf = node->view_src != nullptr ? node->view_src->buffer : node->buffer;
    if (buf == nullptr) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf);
    if (buft->iface.get_name != ggml_backend_rpc_buffer_type_name) {
        return true; // coordinator-local buffer: not resolvable on any worker
    }
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->endpoint != endpoint || buft_ctx->device != device;
}

// the split's nodes minus foreign view ops (see rpc_is_foreign_view_node)
static void collect_local_nodes(const ggml_cgraph * cgraph, const std::string & endpoint, uint32_t device,
                                std::vector<const ggml_tensor *> & nodes) {
    nodes.reserve(cgraph->n_nodes);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (rpc_is_foreign_view_node(node, endpoint, device)) {
            continue;
        }
        nodes.push_back(node);
    }
}

// collect the graph's nodes (with op links) plus everything they reference as leaves
static void collect_graph_tensors(const std::vector<const ggml_tensor *> & nodes, std::vector<rpc_tensor> & tensors) {
    const uint32_t n_nodes = nodes.size();
    std::unordered_set<const ggml_tensor*> node_set;
    node_set.reserve(n_nodes);
    for (uint32_t i = 0; i < n_nodes; i++) {
        node_set.insert(nodes[i]);
    }
    std::unordered_set<const ggml_tensor*> visited;
    auto add_leaf = [&](const ggml_tensor * t) {
        if (t == nullptr || node_set.count(t) > 0 || visited.count(t) > 0) {
            return;
        }
        visited.insert(t);
        tensors.push_back(serialize_tensor_leaf(t));
    };
    for (uint32_t i = 0; i < n_nodes; i++) {
        const ggml_tensor * node = nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            add_leaf(node->src[j]);
        }
        add_leaf(node->view_src);
        if (visited.insert(node).second) {
            tensors.push_back(serialize_tensor(node));
        }
    }
}

// proto 4.3 serialization: same wire layout as serialize_graph but prefixed with the graph uid
static void serialize_graph_uid(uint32_t device, uint64_t uid, const std::vector<const ggml_tensor *> & nodes, std::vector<uint8_t> & output) {
    const uint32_t n_nodes = nodes.size();
    std::vector<rpc_tensor> tensors;
    collect_graph_tensors(nodes, tensors);
    const uint32_t n_tensors = tensors.size();
    const size_t output_size = sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint64_t)
        + sizeof(n_nodes) + n_nodes*sizeof(uint64_t) + sizeof(n_tensors) + n_tensors*sizeof(rpc_tensor);
    output.resize(output_size, 0);
    uint8_t * dest = output.data();
    memcpy(dest, &device, sizeof(device));    dest += sizeof(device);
    uint32_t pad = 0;
    memcpy(dest, &pad, sizeof(pad));          dest += sizeof(pad);
    memcpy(dest, &uid, sizeof(uid));          dest += sizeof(uid);
    memcpy(dest, &n_nodes, sizeof(n_nodes));  dest += sizeof(n_nodes);
    for (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(dest + i*sizeof(uint64_t), &nodes[i], sizeof(uint64_t));
    }
    dest += n_nodes*sizeof(uint64_t);
    memcpy(dest, &n_tensors, sizeof(n_tensors)); dest += sizeof(n_tensors);
    memcpy(dest, tensors.data(), n_tensors*sizeof(rpc_tensor));
}

static void serialize_graph(uint32_t device, const std::vector<const ggml_tensor *> & nodes, std::vector<uint8_t> & output) {
    uint32_t n_nodes = nodes.size();
    std::vector<rpc_tensor> tensors;
    collect_graph_tensors(nodes, tensors);
    // serialization format:
    // | device (4 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    int output_size = 2*sizeof(uint32_t) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t) + n_tensors * sizeof(rpc_tensor);
    output.resize(output_size, 0);
    uint8_t * dest = output.data();
    memcpy(dest, &device, sizeof(device));
    dest += sizeof(device);
    memcpy(dest, &n_nodes, sizeof(n_nodes));
    dest += sizeof(n_nodes);
    for (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(dest + i * sizeof(uint64_t), &nodes[i], sizeof(uint64_t));
    }
    dest += n_nodes * sizeof(uint64_t);
    memcpy(dest, &n_tensors, sizeof(n_tensors));
    dest += sizeof(n_tensors);
    rpc_tensor * out_tensors = (rpc_tensor *)dest;
    memcpy(out_tensors, tensors.data(), n_tensors * sizeof(rpc_tensor));
}

static enum ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    ggml_backend_dev_t rpc_dev = ggml_backend_get_device(backend);
    ggml_backend_rpc_device_context * rpc_dev_ctx = (ggml_backend_rpc_device_context *)rpc_dev->context;

    GGML_ASSERT(cgraph->n_nodes > 0);
    if (rpc_endpoint_failed(rpc_ctx->endpoint)) {
        return GGML_STATUS_FAILED; // poisoned endpoint: fail the decode cleanly, don't touch the wire
    }
    auto sock = get_socket(rpc_ctx->endpoint);
    if (sock == nullptr) {
        rpc_mark_failed(rpc_ctx->endpoint, __func__);
        return GGML_STATUS_FAILED;
    }
    bool status;
    if (cgraph->uid != 0 && rpc_async_state(sock.get()).server_minor >= 3) {
        // proto 4.3: uid-keyed server cache - send each distinct graph once per
        // connection, then recompute it by uid
        if (rpc_dev_ctx->sent_graph_uids_sock != sock.get()) {
            // fresh connection: the server-side cache started empty
            rpc_dev_ctx->sent_graph_uids.clear();
            rpc_dev_ctx->sent_graph_uids_order.clear();
            rpc_dev_ctx->sent_graph_uids_sock = sock.get();
        }
        if (rpc_dev_ctx->sent_graph_uids.count(cgraph->uid) > 0) {
            rpc_msg_graph_recompute_uid_req request = { rpc_ctx->device, 0, cgraph->uid };
            status = send_rpc_cmd(sock, RPC_CMD_GRAPH_RECOMPUTE_UID, &request, sizeof(request));
        } else {
            std::vector<const ggml_tensor *> nodes;
            collect_local_nodes(cgraph, rpc_ctx->endpoint, rpc_ctx->device, nodes);
            if (nodes.empty()) {
                return GGML_STATUS_SUCCESS; // only foreign view ops: nothing to compute here
            }
            std::vector<uint8_t> input;
            serialize_graph_uid(rpc_ctx->device, cgraph->uid, nodes, input);
            status = send_rpc_cmd(sock, RPC_CMD_GRAPH_COMPUTE_UID, input.data(), input.size());
            if (status) {
                // client tracking cap must stay below the server cache cap so a
                // remembered uid is never a server-side miss
                constexpr size_t max_sent_uids = RPC_GRAPH_UID_CACHE_CAP/2;
                rpc_dev_ctx->sent_graph_uids.insert(cgraph->uid);
                rpc_dev_ctx->sent_graph_uids_order.push_back(cgraph->uid);
                if (rpc_dev_ctx->sent_graph_uids_order.size() > max_sent_uids) {
                    rpc_dev_ctx->sent_graph_uids.erase(rpc_dev_ctx->sent_graph_uids_order.front());
                    rpc_dev_ctx->sent_graph_uids_order.pop_front();
                }
            }
        }
    } else {
        bool reuse = cgraph->uid != 0 && rpc_dev_ctx->last_graph_uid == cgraph->uid;
        if (reuse) {
            rpc_msg_graph_recompute_req request;
            request.device = rpc_ctx->device;
            status = send_rpc_cmd(sock, RPC_CMD_GRAPH_RECOMPUTE, &request, sizeof(request));
        } else {
            rpc_dev_ctx->last_graph_uid = cgraph->uid;
            std::vector<const ggml_tensor *> nodes;
            collect_local_nodes(cgraph, rpc_ctx->endpoint, rpc_ctx->device, nodes);
            if (nodes.empty()) {
                rpc_dev_ctx->last_graph_uid = 0; // an empty graph was never stored: don't RECOMPUTE it
                return GGML_STATUS_SUCCESS;
            }
            std::vector<uint8_t> input;
            serialize_graph(rpc_ctx->device, nodes, input);
            status = send_rpc_cmd(sock, RPC_CMD_GRAPH_COMPUTE, input.data(), input.size());
        }
    }
    if (!status) {
        // a contained worker-side compute failure closes the connection (TASKS.md #29e) —
        // fail this decode instead of aborting the whole coordinator; the caller sees
        // GGML_STATUS_FAILED through the scheduler
        GGML_LOG_ERROR("[%s] graph compute failed on %s — the worker dropped the connection "
                       "(worker-side compute error or worker death)\n", __func__, rpc_ctx->endpoint.c_str());
        rpc_dev_ctx->last_graph_uid = 0; // never RECOMPUTE against a failed/new connection
        rpc_dev_ctx->sent_graph_uids.clear();
        rpc_dev_ctx->sent_graph_uids_order.clear();
        rpc_dev_ctx->sent_graph_uids_sock = nullptr;
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .set_tensor_async        = */ ggml_backend_rpc_set_tensor_async,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ ggml_backend_rpc_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_rpc_graph_compute,
    /* .event_record            = */ ggml_backend_rpc_event_record,
    /* .event_wait              = */ ggml_backend_rpc_event_wait,
    /* .graph_optimize          = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    std::string buft_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    // NOTE: buffer types are allocated and never freed; this is by design
    static std::unordered_map<std::string, ggml_backend_buffer_type_t> buft_map;
    auto it = buft_map.find(buft_name);
    if (it != buft_map.end()) {
        return it->second;
    }
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        GGML_LOG_ERROR("Failed to connect to %s\n", endpoint);
        return nullptr;
    }
    size_t alignment = get_alignment(sock, device);
    size_t max_size = get_max_size(sock, device);
    ggml_backend_rpc_buffer_type_context * buft_ctx = new ggml_backend_rpc_buffer_type_context {
        /* .endpoint  = */ endpoint,
        /* .device    = */ device,
        /* .name      = */ buft_name,
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    ggml_backend_buffer_type_t buft = new ggml_backend_buffer_type {
        /* .iface   = */ ggml_backend_rpc_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ buft_ctx
    };
    buft_map[buft_name] = buft;
    return buft;
}

ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device) {
    std::string dev_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    ggml_backend_rpc_context * ctx = new ggml_backend_rpc_context {
        /* .endpoint       = */ endpoint,
        /* .device         = */ device,
        /* .name           = */ dev_name,
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_rpc_guid(),
        /* .iface   = */ ggml_backend_rpc_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ ctx
    };
    return backend;
}

bool ggml_backend_is_rpc(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_rpc_guid());
}

static void get_device_memory(const std::shared_ptr<socket_t> & sock, uint32_t device, size_t * free, size_t * total) {
    rpc_msg_get_device_memory_req request;
    request.device = device;
    rpc_msg_get_device_memory_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_DEVICE_MEMORY, &request, sizeof(request), &response, sizeof(response));
    if (!status) {
        rpc_mark_failed(rpc_async_state(sock.get()).endpoint, __func__);
        *free  = 0;
        *total = 0;
        return;
    }
    *free = response.free_mem;
    *total = response.total_mem;
}

void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total) {
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        *free = 0;
        *total = 0;
        return;
    }
    get_device_memory(sock, device, free, total);
}

bool ggml_backend_rpc_set_split_states(const char * endpoint, const void * data, size_t size) {
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        GGML_LOG_ERROR("Failed to connect to %s\n", endpoint);
        return false;
    }
    return send_rpc_cmd(sock, RPC_CMD_SET_SPLIT_STATES, data, size);
}

// RPC server-side implementation

// One rpc_server instance is shared by every client connection (each connection
// runs in its own thread). exec_mutex serializes command execution: backends and
// the buffer registry are single-threaded by design, and holding the lock for the
// duration of a command gives cross-connection commands exactly the dependency
// semantics worker-to-worker copies need (a reader waits out an in-flight compute).
class rpc_server {
public:
    rpc_server(std::vector<ggml_backend_t> all_backends, const char * cache_dir)
        : backends(std::move(all_backends)), cache_dir(cache_dir) {
        stored_graphs.resize(backends.size());
        stored_graph_caches.resize(backends.size());
    }
    ~rpc_server();

    // worker-local model sourcing (TASKS.md #26): index the tensors of local GGUF files
    // by content hash so SET_TENSOR_HASH cache misses read from local disk instead of
    // streaming from the coordinator. Call before serving — the index is immutable after.
    void build_local_index(const char * model_dir);

    std::mutex exec_mutex;
    // set (under exec_mutex) by the serving thread before dispatching a command
    uint64_t current_conn = 0;

    size_t device_count() const { return backends.size(); }

    // free every buffer allocated by a connection when it closes - same reclaim
    // semantics as the old one-server-per-connection design
    void free_owned(uint64_t conn_id);

    void hello(rpc_msg_hello_rsp & response);
    bool alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response);
    bool get_alignment(const rpc_msg_get_alignment_req & request, rpc_msg_get_alignment_rsp & response);
    bool get_max_size(const rpc_msg_get_max_size_req & request, rpc_msg_get_max_size_rsp & response);
    bool buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response);
    bool free_buffer(const rpc_msg_free_buffer_req & request);
    bool buffer_clear(const rpc_msg_buffer_clear_req & request);
    bool set_tensor(const std::vector<uint8_t> & input);
    bool set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response);
    bool set_tensor_hash2(const rpc_msg_set_tensor_hash2_req & request, rpc_msg_set_tensor_hash_rsp & response);
    bool get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response);
    bool copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response);
    bool graph_compute(const std::vector<uint8_t> & input);
    bool graph_recompute(const rpc_msg_graph_recompute_req & request);
    bool graph_compute_uid(const std::vector<uint8_t> & input);
    bool graph_recompute_uid(const rpc_msg_graph_recompute_uid_req & request);
    bool graph_fused(const std::vector<uint8_t> & input, std::vector<uint8_t> & response);
    bool init_tensor(const rpc_msg_init_tensor_req & request);
    bool get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response);
    bool get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response);
    bool get_device_desc(const rpc_msg_get_device_desc_req & request, rpc_msg_get_device_desc_rsp & response);
    bool buffer_set_usage(const rpc_msg_buffer_set_usage_req & request);

    // fence bookkeeping (proto 4.2 worker-to-worker pulls): per-connection count of
    // executed commands, kept in lockstep with the client's sent-command counter
    // (HELLO included). A fenced read waits - without holding exec_mutex - until the
    // named connection has executed past the fence.
    void conn_started(uint64_t conn_id);  // counts the HELLO
    void mark_executed(uint64_t conn_id);
    void conn_closed(uint64_t conn_id);
    bool wait_fence(uint64_t conn_id, uint64_t seq);

    bool copy_from_remote(const std::vector<uint8_t> & input, rpc_msg_copy_from_remote_rsp & response);

    struct stored_graph {
        std::vector<uint8_t>   buffer;
        ggml_cgraph          * graph;
        uint64_t               owner_conn;
    };

private:
    bool get_cached_file(uint64_t hash, std::vector<uint8_t> & data);

    // run a graph compute with error containment (TASKS.md #29e): a backend failure —
    // including a CUDA error thrown under GGML_CUDA_ERROR_CONTAIN — fails THIS
    // connection instead of killing the worker. Repeated failures mean the backend is
    // unusable (e.g. a poisoned CUDA context) — exit so `restart: always` resurrects a
    // clean process instead of leaving a zombie that fails every request.
    bool compute_contained(uint32_t device, ggml_cgraph * graph, const char * what);
    std::atomic<int> consecutive_compute_failures{0};

    struct local_tensor_ref {
        std::string file;
        uint64_t    offset; // absolute file offset of the tensor data
        uint64_t    size;
    };
    bool get_local_tensor(uint64_t hash, std::vector<uint8_t> & data);
    bool set_tensor_hash_apply(const rpc_msg_set_tensor_hash_req & request, std::vector<uint8_t> & cached_file, rpc_msg_set_tensor_hash_rsp & response);
    // hash -> location in a local GGUF; built once at startup, read-only afterwards
    std::unordered_map<uint64_t, local_tensor_ref> local_index;

    // TASKS.md #44: name -> local GGUF locations, so SET_TENSOR_HASH2 can serve an
    // arbitrary SLICE of a tensor by pread (split-boundary moves keep hitting).
    // Multiple files may carry the same tensor name; the hash verify disambiguates.
    bool get_local_slice(const rpc_msg_set_tensor_hash2_req & request, std::vector<uint8_t> & data);
    std::unordered_map<std::string, std::vector<local_tensor_ref>> local_name_index;

    ggml_tensor * deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor);
    ggml_tensor * create_node(uint64_t id,
                              struct ggml_context * ctx,
                              const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                              std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map);

    // deserialize the | n_nodes | nodes | n_tensors | tensors | tail of a
    // (GRAPH_COMPUTE / GRAPH_COMPUTE_UID) payload into sg and run it
    bool deserialize_compute(uint32_t device, const uint8_t * src, size_t size, stored_graph & sg, const char * what);

    std::vector<ggml_backend_t> backends;
    const char * cache_dir;
    std::unordered_set<ggml_backend_buffer_t> buffers;
    std::unordered_map<ggml_backend_buffer_t, uint64_t> buffer_owners;
    // store the last computed graph for each backend
    std::vector<stored_graph> stored_graphs;
    // proto 4.3: uid-keyed graph cache per backend, LRU-capped
    struct stored_graph_cache {
        std::unordered_map<uint64_t, stored_graph> by_uid;
        std::deque<uint64_t>                       order;
    };
    std::vector<stored_graph_cache> stored_graph_caches;

    std::mutex fence_mutex;
    std::condition_variable fence_cv;
    std::unordered_map<uint64_t, uint64_t> conn_executed;
    std::unordered_set<uint64_t> conns_gone;
};

void rpc_server::conn_started(uint64_t conn_id) {
    {
        std::lock_guard<std::mutex> lock(fence_mutex);
        conn_executed[conn_id] = 1; // the HELLO
    }
    fence_cv.notify_all();
}

void rpc_server::mark_executed(uint64_t conn_id) {
    {
        std::lock_guard<std::mutex> lock(fence_mutex);
        conn_executed[conn_id]++;
    }
    fence_cv.notify_all();
}

void rpc_server::conn_closed(uint64_t conn_id) {
    {
        std::lock_guard<std::mutex> lock(fence_mutex);
        conns_gone.insert(conn_id);
    }
    fence_cv.notify_all();
}

bool rpc_server::wait_fence(uint64_t conn_id, uint64_t seq) {
    std::unique_lock<std::mutex> lock(fence_mutex);
    // generous ceiling: a fence only outlives its compute if something is wedged
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(10);
    while (conn_executed[conn_id] < seq) {
        if (conns_gone.count(conn_id) > 0) {
            return false; // the gating connection died before reaching the fence
        }
        if (fence_cv.wait_until(lock, deadline) == std::cv_status::timeout) {
            GGML_LOG_ERROR("[%s] timed out waiting for conn %" PRIu64 " to reach seq %" PRIu64 "\n",
                           __func__, conn_id, seq);
            return false;
        }
    }
    return true;
}

bool rpc_server::copy_from_remote(const std::vector<uint8_t> & input, rpc_msg_copy_from_remote_rsp & response) {
    // | rpc_tensor src | rpc_tensor dst | src_offset | dst_offset | size | fence_conn | fence_seq | ep_len | ep |
    response.result = 0;
    const size_t fixed = 2*sizeof(rpc_tensor) + 5*sizeof(uint64_t) + sizeof(uint32_t);
    if (input.size() < fixed) {
        return false;
    }
    const uint8_t * p = input.data();
    rpc_tensor rsrc, rdst;
    uint64_t src_offset, dst_offset, size, fence_conn, fence_seq;
    uint32_t ep_len;
    memcpy(&rsrc, p, sizeof(rsrc));               p += sizeof(rsrc);
    memcpy(&rdst, p, sizeof(rdst));               p += sizeof(rdst);
    memcpy(&src_offset, p, sizeof(src_offset));   p += sizeof(src_offset);
    memcpy(&dst_offset, p, sizeof(dst_offset));   p += sizeof(dst_offset);
    memcpy(&size, p, sizeof(size));               p += sizeof(size);
    memcpy(&fence_conn, p, sizeof(fence_conn));   p += sizeof(fence_conn);
    memcpy(&fence_seq, p, sizeof(fence_seq));     p += sizeof(fence_seq);
    memcpy(&ep_len, p, sizeof(ep_len));           p += sizeof(ep_len);
    if (input.size() != fixed + ep_len || ep_len == 0) {
        return false;
    }
    std::string src_endpoint((const char *) p, ep_len);

    // pull directly from the source worker - this process is a normal RPC client
    // there. Peer sockets are cached with strong references: the general registry
    // only holds weak_ptrs, and reconnecting (+ HELLO) per pull would add a round
    // trip to every handoff. Unreachable peers are negative-cached: without it every
    // pull re-dials (up to a full SYN timeout against a dead host), thousands of
    // times per token (TASKS.md #39).
    static std::mutex peer_mutex;
    static std::unordered_map<std::string, socket_ptr> peer_socks;
    static std::unordered_map<std::string, int64_t>    peer_dead_until_ms;
    const auto now_ms = [] {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    socket_ptr sock;
    {
        std::lock_guard<std::mutex> lock(peer_mutex);
        auto it = peer_socks.find(src_endpoint);
        if (it != peer_socks.end()) {
            sock = it->second;
        } else {
            auto dead = peer_dead_until_ms.find(src_endpoint);
            if (dead != peer_dead_until_ms.end() && now_ms() < dead->second) {
                return true; // still in backoff: soft failure, no dial
            }
        }
    }
    if (sock == nullptr) {
        sock = get_socket(src_endpoint);
        if (sock == nullptr) {
            GGML_LOG_ERROR("[%s] cannot reach source worker '%s' (backing off 5s)\n", __func__, src_endpoint.c_str());
            std::lock_guard<std::mutex> lock(peer_mutex);
            peer_dead_until_ms[src_endpoint] = now_ms() + 5000;
            return true; // soft failure: the coordinator falls back to the bridged copy
        }
        std::lock_guard<std::mutex> lock(peer_mutex);
        peer_socks[src_endpoint] = sock;
        peer_dead_until_ms.erase(src_endpoint);
    }
    rpc_msg_get_tensor_fenced_req req = { rsrc, src_offset, size, fence_conn, fence_seq };
    std::vector<uint8_t> data(size);
    if (!send_rpc_cmd(sock, RPC_CMD_GET_TENSOR_FENCED, &req, sizeof(req), data.data(), size)) {
        GGML_LOG_ERROR("[%s] fenced pull from '%s' failed\n", __func__, src_endpoint.c_str());
        std::lock_guard<std::mutex> lock(peer_mutex);
        peer_socks.erase(src_endpoint); // dead peer socket: reconnect on the next pull
        return true; // soft failure
    }

    // local write under the execution lock (the fetch above must run without it)
    {
        std::lock_guard<std::mutex> exec_lock(exec_mutex);
        struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ggml_context_ptr ctx_ptr { ggml_init(params) };
        GGML_ASSERT(ctx_ptr != nullptr);
        ggml_tensor * dst = deserialize_tensor(ctx_ptr.get(), &rdst);
        if (dst == nullptr || dst->buffer == nullptr) {
            GGML_LOG_ERROR("[%s] error deserializing dst tensor\n", __func__);
            return false;
        }
        ggml_backend_tensor_set(dst, data.data(), dst_offset, size);
    }
    response.result = 1;
    return true;
}

void rpc_server::free_owned(uint64_t conn_id) {
    for (auto it = buffer_owners.begin(); it != buffer_owners.end(); ) {
        if (it->second == conn_id) {
            ggml_backend_buffer_free(it->first);
            buffers.erase(it->first);
            it = buffer_owners.erase(it);
        } else {
            ++it;
        }
    }
}

void rpc_server::hello(rpc_msg_hello_rsp & response) {
    response.major = RPC_PROTO_MAJOR_VERSION;
    response.minor = RPC_PROTO_MINOR_VERSION;
    response.patch = RPC_PROTO_PATCH_VERSION;
    LOG_DBG("[%s] version: %d.%d.%d\n", __func__, response.major, response.minor, response.patch);
}

bool rpc_server::get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft;
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead()*(1 + GGML_MAX_SRC),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        GGML_LOG_ERROR("Null tensor pointer passed to server get_alloc_size function.\n");
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (request.srcs[i].id != 0) {
            tensor->src[i] = deserialize_tensor(ctx, &request.srcs[i]);
        }
    }

    LOG_DBG("[%s] device: %d, buffer: %p, data: %p\n", __func__, dev_id, (void*)tensor->buffer, tensor->data);
    if (tensor->buffer == nullptr) {
        //No buffer allocated.
        buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    } else {
        buft = tensor->buffer->buft;
    }

    response.alloc_size = ggml_backend_buft_get_alloc_size(buft, tensor);

    return true;
}

bool rpc_server::alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, request.size);
    response.remote_ptr = 0;
    response.remote_size = 0;
    if (buffer != nullptr) {
        response.remote_ptr = reinterpret_cast<uint64_t>(buffer);
        response.remote_size = buffer->size;
        LOG_DBG("[%s] device: %d, size: %" PRIu64 " -> remote_ptr: %" PRIx64 ", remote_size: %" PRIu64 "\n",
            __func__, dev_id, request.size, response.remote_ptr, response.remote_size);
        buffers.insert(buffer);
        buffer_owners[buffer] = current_conn;
    } else {
        LOG_DBG("[%s] device: %d, size: %" PRIu64 " -> failed\n", __func__, dev_id, request.size);
    }
    return true;
}

bool rpc_server::get_alignment(const rpc_msg_get_alignment_req & request, rpc_msg_get_alignment_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t alignment = ggml_backend_buft_get_alignment(buft);
    LOG_DBG("[%s] device: %d, alignment: %lu\n", __func__, dev_id, alignment);
    response.alignment = alignment;
    return true;
}

bool rpc_server::get_max_size(const rpc_msg_get_max_size_req & request, rpc_msg_get_max_size_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t max_size = ggml_backend_buft_get_max_size(buft);
    LOG_DBG("[%s] device: %d, max_size: %lu\n", __func__, dev_id, max_size);
    response.max_size = max_size;
    return true;
}

bool rpc_server::buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response) {
    LOG_DBG("[%s] remote_ptr: %" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    void * base = ggml_backend_buffer_get_base(buffer);
    response.base_ptr = reinterpret_cast<uint64_t>(base);
    return true;
}

bool rpc_server::free_buffer(const rpc_msg_free_buffer_req & request) {
    LOG_DBG("[%s] remote_ptr: %" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    ggml_backend_buffer_free(buffer);
    buffers.erase(buffer);
    buffer_owners.erase(buffer);
    return true;
}

bool rpc_server::buffer_clear(const rpc_msg_buffer_clear_req & request) {
    LOG_DBG("[%s] remote_ptr: %" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    ggml_backend_buffer_clear(buffer, request.value);
    return true;
}

ggml_tensor * rpc_server::deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor) {
    // Validate tensor type before using it
    if (tensor->type >= GGML_TYPE_COUNT) {
        GGML_LOG_ERROR("[%s] invalid tensor type received: %u\n", __func__, tensor->type);
        return nullptr;
    }

    // Fix: Prevent division by zero if blck_size is 0 (e.g., deprecated types)
    if (ggml_blck_size((enum ggml_type)tensor->type) == 0) {
        GGML_LOG_ERROR("[%s] invalid tensor type received (blck_size is 0): %u\n", __func__, tensor->type);
        return nullptr;
    }

    ggml_tensor * result = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

    // ggml_new_tensor_4d might fail if dimensions are invalid, although less likely to crash than invalid type
    if (result == nullptr) {
        GGML_LOG_ERROR("[%s] ggml_new_tensor_4d failed for type %u\n", __func__, tensor->type);
        return nullptr;
    }

    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    if (result->buffer && buffers.find(result->buffer) == buffers.end()) {
        result->buffer = nullptr;
    }

    if (result->buffer) {
        uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(result->buffer);
        uint64_t buffer_size = (uint64_t) ggml_backend_buffer_get_size(result->buffer);
        if (ggml_backend_buffer_is_meta(result->buffer)) {
            // no bounds validation possible: for tensor-parallel meta buffers the data
            // field is a logical address - buffers are packed in per-device slice space
            // and views carry full-tensor-space offsets. The meta layer re-derives the
            // physical per-device placement itself and never dereferences this pointer.
        } else {
            // require that the tensor data does not go beyond the buffer end
            uint64_t tensor_size = (uint64_t) ggml_nbytes(result);
            GGML_ASSERT(tensor->data + tensor_size >= tensor->data); // check for overflow
            GGML_ASSERT(tensor->data >= buffer_start && tensor->data + tensor_size <= buffer_start + buffer_size);
        }
    }

    result->op = (ggml_op) tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data = reinterpret_cast<void *>(tensor->data);
    ggml_set_name(result, tensor->name);
    return result;
}


// opt-in cache size cap (GGML_RPC_CACHE_LIMIT_MIB): after a save, evict entries
// oldest-mtime-first until the dir fits. Served entries refresh their mtime, so a
// hot working set survives while superseded model generations age out.
static void rpc_cache_enforce_limit(const char * cache_dir) {
    static const long long limit_mib = []() {
        const char * env = getenv("GGML_RPC_CACHE_LIMIT_MIB");
        return env != nullptr ? atoll(env) : 0LL;
    }();
    if (limit_mib <= 0) {
        return;
    }
    const uint64_t limit = (uint64_t) limit_mib * 1024ull * 1024ull;
    struct cache_entry {
        fs::path           path;
        uint64_t           size;
        fs::file_time_type mtime;
    };
    std::vector<cache_entry> entries;
    uint64_t total = 0;
    std::error_code ec;
    for (auto it = fs::directory_iterator(cache_dir, fs::directory_options::skip_permission_denied, ec);
         !ec && it != fs::directory_iterator(); it.increment(ec)) {
        if (!it->is_regular_file(ec)) {
            continue;
        }
        cache_entry e { it->path(), it->file_size(ec), it->last_write_time(ec) };
        if (ec) {
            continue;
        }
        total += e.size;
        entries.push_back(std::move(e));
    }
    if (total <= limit) {
        return;
    }
    std::sort(entries.begin(), entries.end(), [](const cache_entry & a, const cache_entry & b) {
        return a.mtime < b.mtime;
    });
    size_t   n_evicted = 0;
    uint64_t freed     = 0;
    for (const cache_entry & e : entries) {
        if (total <= limit) {
            break;
        }
        if (fs::remove(e.path, ec) && !ec) {
            total -= e.size;
            freed += e.size;
            n_evicted++;
        }
    }
    if (n_evicted > 0) {
        GGML_LOG_INFO("[rpc cache] evicted %zu entries (%.1f MiB) to fit GGML_RPC_CACHE_LIMIT_MIB=%lld\n",
                      n_evicted, freed / (1024.0*1024.0), limit_mib);
    }
}

bool rpc_server::set_tensor(const std::vector<uint8_t> & input) {
    // serialization format: | rpc_tensor | offset (8 bytes) | data (size bytes) |
    if (input.size() < sizeof(rpc_tensor) + sizeof(uint64_t)) {
        return false;
    }
    const rpc_tensor * in_tensor = (const rpc_tensor *)input.data();
    uint64_t offset;
    memcpy(&offset, input.data() + sizeof(rpc_tensor), sizeof(offset));
    const size_t size = input.size() - sizeof(rpc_tensor) - sizeof(offset);

    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %zu\n", __func__, (void*)tensor->buffer, tensor->data, offset, size);

    // sanitize tensor->data (skip for tensor-parallel meta buffers: their data field is a
    // logical address in split-tensor space and the meta layer bounds physical placement)
    if (!ggml_backend_buffer_is_meta(tensor->buffer)) {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 || size > (p1 - in_tensor->data - offset)) {
            GGML_LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu) out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, in_tensor->data, offset, size, p0, p1);
            return false;
        }
    }

    const void * data = input.data() + sizeof(rpc_tensor) + sizeof(offset);
    if (cache_dir && size > HASH_THRESHOLD) {
        uint64_t hash = fnv_hash((const uint8_t*)data, size);
        char hash_str[17];
        snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, hash);
        // save to cache_dir/hash_str via a temp file: a partial write (killed worker,
        // full disk) must never land under the final name - an unverified truncated
        // entry used to poison every later load of the tensor
        fs::path cache_file = fs::path(cache_dir) / hash_str;
        fs::path tmp_file   = fs::path(cache_dir) / (std::string(hash_str) + ".tmp");
        std::ofstream ofs(tmp_file, std::ios::binary);
        if (!ofs.is_open()) {
            // the cache dir can disappear at runtime (manual cleanup) - without this
            // every later save fails silently and loads re-stream forever
            std::error_code ec_mk;
            fs::create_directories(fs::path(cache_dir), ec_mk);
            ofs.open(tmp_file, std::ios::binary);
        }
        ofs.write((const char *)data, size);
        ofs.close();
        std::error_code ec;
        if (ofs.good()) {
            fs::rename(tmp_file, cache_file, ec);
        }
        if (!ofs.good() || ec) {
            GGML_LOG_ERROR("[%s] failed to save '%s'\n", __func__, cache_file.string().c_str());
            fs::remove(tmp_file, ec);
        } else {
#ifndef _WIN32
            // flush + drop the pages now: a model-sized load otherwise accumulates
            // model-sized DIRTY page cache on top of the weights themselves, enough
            // to OOM a worker whose RAM share is sized near its capacity
            int fd = open(cache_file.string().c_str(), O_RDONLY);
            if (fd >= 0) {
                fsync(fd);
                posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
                close(fd);
            }
#endif
            GGML_LOG_INFO("[%s] saved to '%s'\n", __func__, cache_file.string().c_str());
            rpc_cache_enforce_limit(cache_dir);
        }
    }
    ggml_backend_tensor_set(tensor, data, offset, size);
    return true;
}

bool rpc_server::get_cached_file(uint64_t hash, std::vector<uint8_t> & data) {
    if (!cache_dir) {
        return false;
    }
    char hash_str[17];
    snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, hash);
    fs::path cache_file = fs::path(cache_dir) / hash_str;
    std::error_code ec;
    if (!fs::exists(cache_file, ec)) {
        return false;
    }
    std::ifstream ifs(cache_file, std::ios::binary);
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    data.resize(size);
    ifs.read((char *)data.data(), size);
    // refresh mtime: the size-capped cache evicts by LRU and a served entry is hot
    fs::last_write_time(cache_file, fs::file_time_type::clock::now(), ec);
    return true;
}

// worker-local model sourcing (TASKS.md #26)
//
// The index maps fnv_hash(tensor bytes) -> (file, offset, size) for every tensor of every
// GGUF under --model-dir that is large enough for the coordinator's SET_TENSOR_HASH path
// (> HASH_THRESHOLD). Hashing reads each file once (~seconds/10 GB on NVMe), so the result
// is persisted in the cache dir keyed by (path, mtime, size) and reloaded instantly while
// the file is unchanged. A stale or different file simply hash-misses at lookup time and
// the coordinator streams as before — correct by construction, no version-skew failure mode.

static uint64_t local_index_file_key(const std::string & path, uint64_t mtime, uint64_t size) {
    std::string ident = path + "|" + std::to_string(mtime) + "|" + std::to_string(size);
    return fnv_hash((const uint8_t *) ident.data(), ident.size());
}

void rpc_server::build_local_index(const char * model_dir) {
    std::error_code ec;
    if (model_dir == nullptr || !fs::is_directory(model_dir, ec)) {
        GGML_LOG_ERROR("model dir '%s' is not a directory, skipping local index\n", model_dir ? model_dir : "");
        return;
    }
    size_t n_files = 0;
    for (auto it = fs::recursive_directory_iterator(model_dir, fs::directory_options::skip_permission_denied, ec);
         it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (ec) {
            break;
        }
        if (!it->is_regular_file(ec)) {
            continue;
        }
        const fs::path path = it->path();
        // GGUF files sometimes carry suffixes (e.g. "model.gguf?download=true") — match anywhere
        if (path.filename().string().find(".gguf") == std::string::npos) {
            continue;
        }
        const uint64_t f_size  = (uint64_t) fs::file_size(path, ec);
        const uint64_t f_mtime = (uint64_t) fs::last_write_time(path, ec).time_since_epoch().count();
        const std::string path_str = path.string();

        // try the persisted index first
        fs::path idx_file;
        if (cache_dir) {
            char key_str[17];
            snprintf(key_str, sizeof(key_str), "%016" PRIx64, local_index_file_key(path_str, f_mtime, f_size));
            idx_file = fs::path(cache_dir) / (std::string("modelidx-") + key_str);
            std::ifstream ifs(idx_file);
            if (ifs) {
                std::string magic, file_line;
                std::getline(ifs, magic);
                // v1 files (no tensor names) are re-hashed once to upgrade
                if (magic == "rpc-model-index v2") {
                    size_t n_loaded = 0;
                    uint64_t hash, offset, size;
                    std::string name;
                    while (ifs >> std::hex >> hash >> std::dec >> offset >> size >> name) {
                        local_index[hash] = { path_str, offset, size };
                        local_name_index[name].push_back({ path_str, offset, size });
                        n_loaded++;
                    }
                    GGML_LOG_INFO("local index: %s (%zu tensors, from cached index)\n", path_str.c_str(), n_loaded);
                    n_files++;
                    continue;
                }
            }
        }

        // parse + hash the GGUF (reads every large tensor once)
        struct gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
        struct gguf_context * gctx = gguf_init_from_file(path_str.c_str(), params);
        if (gctx == nullptr) {
            GGML_LOG_ERROR("local index: failed to parse %s, skipping\n", path_str.c_str());
            continue;
        }
        std::ifstream f(path_str, std::ios::binary);
        if (!f) {
            GGML_LOG_ERROR("local index: failed to open %s, skipping\n", path_str.c_str());
            gguf_free(gctx);
            continue;
        }
        GGML_LOG_INFO("local index: hashing %s (%.1f GiB) ...\n", path_str.c_str(), f_size / (1024.0*1024.0*1024.0));
        const size_t data_offset = gguf_get_data_offset(gctx);
        std::vector<uint8_t> buf;
        std::vector<std::tuple<uint64_t, uint64_t, uint64_t, std::string>> entries; // hash, offset, size, name
        for (int64_t i = 0; i < gguf_get_n_tensors(gctx); i++) {
            const size_t size = gguf_get_tensor_size(gctx, i);
            if (size <= HASH_THRESHOLD) {
                continue; // the coordinator only hashes tensors above the threshold
            }
            const uint64_t offset = data_offset + gguf_get_tensor_offset(gctx, i);
            buf.resize(size);
            f.seekg(offset);
            if (!f.read((char *) buf.data(), size)) {
                GGML_LOG_ERROR("local index: short read in %s at offset %" PRIu64 ", skipping file\n", path_str.c_str(), offset);
                entries.clear();
                break;
            }
            entries.emplace_back(fnv_hash(buf.data(), size), offset, size, gguf_get_tensor_name(gctx, i));
        }
        gguf_free(gctx);
        for (const auto & [hash, offset, size, name] : entries) {
            local_index[hash] = { path_str, offset, size };
            local_name_index[name].push_back({ path_str, offset, size });
        }
        GGML_LOG_INFO("local index: %s -> %zu tensors indexed\n", path_str.c_str(), entries.size());
        if (!entries.empty()) {
            n_files++;
        }

        // persist for the next start
        if (cache_dir && !entries.empty()) {
            std::ofstream ofs(idx_file);
            if (ofs) {
                ofs << "rpc-model-index v2\n";
                for (const auto & [hash, offset, size, name] : entries) {
                    char hash_str[17];
                    snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, hash);
                    ofs << hash_str << " " << offset << " " << size << " " << name << "\n";
                }
            } else {
                GGML_LOG_ERROR("local index: failed to persist %s (re-hashing next start)\n", idx_file.string().c_str());
            }
        }
    }
    GGML_LOG_INFO("local index: %zu files, %zu tensors total%s\n", n_files, local_index.size(),
                  cache_dir ? "" : " (no cache dir — the index is re-hashed every start)");
}

bool rpc_server::get_local_tensor(uint64_t hash, std::vector<uint8_t> & data) {
    const auto it = local_index.find(hash);
    if (it == local_index.end()) {
        return false;
    }
    const local_tensor_ref & ref = it->second;
    std::ifstream f(ref.file, std::ios::binary);
    if (!f) {
        return false;
    }
    data.resize(ref.size);
    f.seekg(ref.offset);
    if (!f.read((char *) data.data(), ref.size)) {
        return false;
    }
    // the file may have changed since indexing — verify before serving
    if (fnv_hash(data.data(), data.size()) != hash) {
        GGML_LOG_ERROR("local index: %s changed on disk (hash mismatch at offset %" PRIu64 "), falling back to streaming\n",
                       ref.file.c_str(), ref.offset);
        return false;
    }
    LOG_DBG("[%s] hash %016" PRIx64 " served from %s (%" PRIu64 " bytes)\n", __func__, hash, ref.file.c_str(), ref.size);
    return true;
}

bool rpc_server::set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response)
{
    std::vector<uint8_t> cached_file;
    if (!get_cached_file(request.hash, cached_file) && !get_local_tensor(request.hash, cached_file)) {
        response.result = 0;
        return true;
    }
    return set_tensor_hash_apply(request, cached_file, response);
}

bool rpc_server::set_tensor_hash2(const rpc_msg_set_tensor_hash2_req & request, rpc_msg_set_tensor_hash_rsp & response)
{
    std::vector<uint8_t> data;
    if (!get_cached_file(request.hash, data) && !get_local_tensor(request.hash, data) && !get_local_slice(request, data)) {
        response.result = 0;
        return true;
    }
    rpc_msg_set_tensor_hash_req base;
    base.tensor = request.tensor;
    base.offset = request.offset;
    base.hash   = request.hash;
    return set_tensor_hash_apply(base, data, response);
}

bool rpc_server::get_local_slice(const rpc_msg_set_tensor_hash2_req & request, std::vector<uint8_t> & data) {
    char name[GGML_MAX_NAME];
    memcpy(name, request.src_name, GGML_MAX_NAME);
    name[GGML_MAX_NAME - 1] = '\0';
    const auto it = local_name_index.find(name);
    if (it == local_name_index.end()) {
        return false;
    }
    const uint64_t n_rows   = request.src_n_rows;
    const uint64_t row_size = request.src_row_size;
    const uint64_t stride   = request.src_row_stride;
    if (n_rows == 0 || row_size == 0) {
        return false;
    }
    const uint64_t total = row_size * n_rows;
    if (total / row_size != n_rows || total > (1ull << 31)) {
        return false;
    }
    const uint64_t span_end = request.src_offset + (n_rows - 1) * stride + row_size;
    if (span_end < request.src_offset) {
        return false;
    }
    for (const local_tensor_ref & ref : it->second) {
        if (span_end > ref.size) {
            continue;
        }
        std::ifstream f(ref.file, std::ios::binary);
        if (!f) {
            continue;
        }
        data.resize(total);
        bool ok = true;
        for (uint64_t i = 0; i < n_rows && ok; i++) {
            f.seekg((std::streamoff) (ref.offset + request.src_offset + i*stride));
            ok = (bool) f.read((char *) data.data() + i*row_size, row_size);
        }
        // same-named tensors across files (or a changed file) disambiguate here
        if (ok && fnv_hash(data.data(), data.size()) == request.hash) {
            LOG_DBG("[%s] slice of '%s' served from %s (%" PRIu64 " bytes, %" PRIu64 " rows)\n",
                    __func__, name, ref.file.c_str(), total, n_rows);
            return true;
        }
    }
    return false;
}

bool rpc_server::set_tensor_hash_apply(const rpc_msg_set_tensor_hash_req & request, std::vector<uint8_t> & cached_file, rpc_msg_set_tensor_hash_rsp & response)
{
    // verify before serving: a truncated or stale entry (interrupted cache write,
    // changed local file) must fall back to streaming, not corrupt the tensor.
    // A bad cache file is deleted so the fallback SET_TENSOR rewrites it.
    if (fnv_hash(cached_file.data(), cached_file.size()) != request.hash) {
        GGML_LOG_ERROR("[%s] cached data for hash 0x%" PRIx64 " fails verification (%zu bytes) - falling back to streaming\n",
                       __func__, request.hash, cached_file.size());
        if (cache_dir) {
            char hash_str[17];
            snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, request.hash);
            std::error_code ec;
            fs::remove(fs::path(cache_dir) / hash_str, ec);
        }
        response.result = 0;
        return true;
    }
    size_t size = cached_file.size();
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %zu, hash: %" PRIx64 "\n",
            __func__, (void*)tensor->buffer, tensor->data, request.offset, size, request.hash);

    // sanitize tensor->data (skip for tensor-parallel meta buffers: their data field is a
    // logical address in split-tensor space and the meta layer bounds physical placement)
    if (!ggml_backend_buffer_is_meta(tensor->buffer)) {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data + request.offset < p0
         || request.tensor.data + request.offset >= p1
         || size > (p1 - request.tensor.data - request.offset)) {
            GGML_LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu, hash=0x%" PRIx64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, request.tensor.data, request.offset, size, request.hash, p0, p1);
            return false;
        }
    }
    ggml_backend_tensor_set(tensor, cached_file.data(), request.offset, size);
    response.result = 1;
    return true;
}

bool rpc_server::init_tensor(const rpc_msg_init_tensor_req & request) {
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        GGML_LOG_ERROR("Null tensor pointer passed to server init_tensor function.\n");
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p\n", __func__, (void*)tensor->buffer, tensor->data);
    // Call the backend's buffer_init_tensor function
    ggml_backend_buffer_t buffer = tensor->buffer;
    if (buffer && buffer->iface.init_tensor) {
        buffer->iface.init_tensor(buffer, tensor);
    } else {
        if (!buffer) {
            GGML_LOG_ERROR("Tensor with null buffer passed to init_tensor function\n");
        }
    }

    if (tensor->extra != nullptr) {
        // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
        // Currently unimplemented.
        GGML_LOG_ERROR("tensor->extra populated by the backend, this is currently unsupported.\n");
        return false;
    }

    return true;
}

bool rpc_server::get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response) {
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %" PRIu64 "\n", __func__, (void*)tensor->buffer, tensor->data, request.offset, request.size);

    // sanitize tensor->data (skip for tensor-parallel meta buffers: their data field is a
    // logical address in split-tensor space and the meta layer bounds physical placement)
    if (!ggml_backend_buffer_is_meta(tensor->buffer)) {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data + request.offset < p0 ||
            request.tensor.data + request.offset >= p1 ||
            request.size > (p1 - request.tensor.data - request.offset)) {
                GGML_LOG_ERROR("[%s] requested tensor region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%" PRIu64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                               __func__, request.tensor.data, request.offset, request.size, p0, p1);
                return false;
        }
    }

    response.resize(request.size, 0);
    ggml_backend_tensor_get(tensor, response.data(), request.offset, request.size);
    return true;
}

bool rpc_server::copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response) {
    struct ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * src = deserialize_tensor(ctx, &request.src);
    ggml_tensor * dst = deserialize_tensor(ctx, &request.dst);
    if (src == nullptr || dst == nullptr || src->buffer == nullptr || dst->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensors\n", __func__);
        return false;
    }

    uint64_t src_size   = (uint64_t) ggml_nbytes(src);
    uint64_t dst_data   = (uint64_t) dst->data;
    uint64_t dst_base   = (uint64_t) ggml_backend_buffer_get_base(dst->buffer);
    uint64_t dst_buf_sz = (uint64_t) ggml_backend_buffer_get_size(dst->buffer);

    if (dst_data + src_size > dst_base + dst_buf_sz) {
        GGML_LOG_ERROR("[%s] out-of-bounds write in rpc_server::copy_tensor:\n"
                         "    write range : [0x%" PRIx64 ", 0x%" PRIx64 "]\n"
                         "    buffer base: [0x%" PRIx64 ", 0x%" PRIx64 "]\n",
                         __func__,
                         dst_data,
                         dst_data + src_size,
                         dst_base,
                         dst_base + dst_buf_sz);
        return false;
    }

    LOG_DBG("[%s] src->buffer: %p, dst->buffer: %p\n",
            __func__, (void*) src->buffer, (void*) dst->buffer);

    response.result = ggml_backend_buffer_copy_tensor(src, dst);
    return true;
}

ggml_tensor * rpc_server::create_node(uint64_t id,
                                      struct ggml_context * ctx,
                                      const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                                      std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map) {
    if (tensor_map.find(id) != tensor_map.end()) {
        return tensor_map[id];
    }
    // Safely find the tensor pointer
    auto it_ptr = tensor_ptrs.find(id);
    if (it_ptr == tensor_ptrs.end()) {
        return nullptr;
    }
    const rpc_tensor * tensor = it_ptr->second;

    struct ggml_tensor * result = deserialize_tensor(ctx, tensor);
    if (result == nullptr) {
        return nullptr;
    }
    if (result->buffer == nullptr && result->data != nullptr) {
        GGML_LOG_ERROR("[%s] invalid data ptr: tensor '%s' (op %s) references unknown buffer 0x%" PRIx64 ", data 0x%" PRIx64 "\n",
                       __func__, tensor->name, ggml_op_name((ggml_op) tensor->op), tensor->buffer, tensor->data);
        return nullptr;
    }
    tensor_map[id] = result;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        // Check if the source ID is 0 before calling create_node recursively
        if (tensor->src[i] == 0) {
            result->src[i] = nullptr;
        } else {
            result->src[i] = create_node(tensor->src[i], ctx, tensor_ptrs, tensor_map);
            // If the recursive call failed for a non-zero ID, propagate the error
            if (result->src[i] == nullptr) {
                GGML_LOG_ERROR("[%s] failed to create source node %d (src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                               __func__, i, tensor->src[i], id);
                // Must return nullptr to signal failure up the call stack
                return nullptr;
            }
        }
    }

    // Handle view_src similarly
    if (tensor->view_src == 0) {
        result->view_src = nullptr;
    } else {
        result->view_src = create_node(tensor->view_src, ctx, tensor_ptrs, tensor_map);
        // If the recursive call failed for a non-zero ID, propagate the error
        if (result->view_src == nullptr) {
            GGML_LOG_ERROR("[%s] failed to create view_src node (view_src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                           __func__, tensor->view_src, id);
            // Must return nullptr to signal failure up the call stack
            return nullptr;
        }
    }
    result->view_offs = tensor->view_offs;
    return result;
}

bool rpc_server::deserialize_compute(uint32_t device, const uint8_t * src, size_t size, stored_graph & sg, const char * what) {
    // payload tail format:
    // | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    if (size < sizeof(uint32_t)) {
        return false;
    }
    uint32_t n_nodes;
    memcpy(&n_nodes, src, sizeof(n_nodes));
    src += sizeof(n_nodes);
    if (size < sizeof(uint32_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t)) {
        return false;
    }
    const uint64_t * nodes = (const uint64_t *)src;
    src += n_nodes*sizeof(uint64_t);
    uint32_t n_tensors;
    memcpy(&n_tensors, src, sizeof(n_tensors));
    src += sizeof(n_tensors);
    if (size < sizeof(uint32_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t) + n_tensors*sizeof(rpc_tensor)) {
        return false;
    }
    const rpc_tensor * tensors = (const rpc_tensor *)src;
    LOG_DBG("[%s] device: %u, n_nodes: %u, n_tensors: %u\n", what, device, n_nodes, n_tensors);

    size_t buf_size = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, false);
    if (sg.buffer.size() < buf_size) {
        sg.buffer.resize(buf_size);
    }
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ sg.buffer.data(),
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, false);
    graph->n_nodes = n_nodes;
    std::unordered_map<uint64_t, const rpc_tensor*> tensor_ptrs;
    tensor_ptrs.reserve(n_tensors);
    for (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs.emplace(tensors[i].id, &tensors[i]);
    }
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
    tensor_map.reserve(n_nodes);
    for (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        memcpy(&id, &nodes[i], sizeof(id));
        graph->nodes[i] = create_node(id, ctx, tensor_ptrs, tensor_map);

        // Check if create_node failed for a *non-zero* ID.
        // If id was 0, create_node returning nullptr is expected.
        // If id was non-zero and create_node returned nullptr, it indicates a deserialization error.
        if (graph->nodes[i] == nullptr && id != 0) {
            GGML_LOG_ERROR("[%s] failed to create graph node %d (id=%" PRId64 ")\n", what, i, id);
            return false;
        }
    }
    if (!compute_contained(device, graph, what)) {
        sg.graph = nullptr; // never RECOMPUTE a graph that failed
        return false;
    }
    sg.graph = graph;
    sg.owner_conn = current_conn;
    return true;
}

bool rpc_server::graph_compute(const std::vector<uint8_t> & input) {
    // serialization format:
    // | device (4 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    if (input.size() < sizeof(uint32_t)) {
        return false;
    }
    uint32_t device;
    memcpy(&device, input.data(), sizeof(device));
    if (device >= backends.size()) {
        return false;
    }
    return deserialize_compute(device, input.data() + sizeof(uint32_t), input.size() - sizeof(uint32_t),
                               stored_graphs[device], __func__);
}

bool rpc_server::graph_compute_uid(const std::vector<uint8_t> & input) {
    // serialization format:
    // | device (4 bytes) | pad (4 bytes) | uid (8 bytes) | n_nodes | nodes | n_tensors | tensors |
    const size_t header = 2*sizeof(uint32_t) + sizeof(uint64_t);
    if (input.size() < header) {
        return false;
    }
    uint32_t device;
    uint64_t uid;
    memcpy(&device, input.data(), sizeof(device));
    memcpy(&uid, input.data() + 2*sizeof(uint32_t), sizeof(uid));
    if (device >= backends.size() || uid == 0) {
        return false;
    }
    stored_graph_cache & cache = stored_graph_caches[device];
    auto it = cache.by_uid.find(uid);
    if (it == cache.by_uid.end()) {
        while (cache.order.size() >= RPC_GRAPH_UID_CACHE_CAP) {
            cache.by_uid.erase(cache.order.front());
            cache.order.pop_front();
        }
        it = cache.by_uid.emplace(uid, stored_graph{}).first;
        cache.order.push_back(uid);
    }
    return deserialize_compute(device, input.data() + header, input.size() - header, it->second, __func__);
}

bool rpc_server::graph_recompute_uid(const rpc_msg_graph_recompute_uid_req & request) {
    uint32_t device = request.device;
    if (device >= backends.size()) {
        return false;
    }
    stored_graph_cache & cache = stored_graph_caches[device];
    auto it = cache.by_uid.find(request.uid);
    if (it == cache.by_uid.end() || it->second.graph == nullptr) {
        GGML_LOG_ERROR("[%s] no cached graph for device %u uid %" PRIu64 "\n", __func__, device, request.uid);
        return false;
    }
    if (it->second.owner_conn != current_conn) {
        // another connection replaced this uid's stored graph - recomputing it
        // would silently run the wrong graph (see graph_recompute)
        GGML_LOG_ERROR("[%s] stored graph for device %u uid %" PRIu64 " belongs to another connection\n",
                       __func__, device, request.uid);
        return false;
    }
    if (!compute_contained(device, it->second.graph, __func__)) {
        it->second.graph = nullptr; // never RECOMPUTE a graph that failed
        return false;
    }
    return true;
}

// fused boundary command (proto 4.5):
// input:    | device u32 | flags u32
//           | if flags&1 (SET):   rpc_tensor | u64 offset | u64 size | data[size]
//           | if flags&2 (GRAPH): u64 uid  (recompute of a uid-cached graph)
//           | if flags&4 (FETCH): rpc_tensor | u64 offset | u64 size |
// response: | data[fetch size] |  (empty when no FETCH)
// Failure semantics match the unfused commands: any error drops the connection.
bool rpc_server::graph_fused(const std::vector<uint8_t> & input, std::vector<uint8_t> & response) {
    size_t pos = 0;
    auto take = [&](void * dst, size_t n) -> bool {
        if (pos + n > input.size()) {
            return false;
        }
        memcpy(dst, input.data() + pos, n);
        pos += n;
        return true;
    };
    uint32_t device, flags;
    if (!take(&device, sizeof(device)) || !take(&flags, sizeof(flags)) || device >= backends.size()) {
        return false;
    }

    struct ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    auto bounds_ok = [&](const rpc_tensor * rt, ggml_tensor * t, uint64_t offset, uint64_t size) -> bool {
        if (t == nullptr || t->buffer == nullptr) {
            return false;
        }
        if (ggml_backend_buffer_is_meta(t->buffer)) {
            return true; // logical split-space address, bounded by the meta layer
        }
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(t->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(t->buffer);
        return rt->data + offset >= p0 && rt->data + offset < p1 && size <= (p1 - rt->data - offset);
    };

    if (flags & 1) { // SET
        rpc_tensor rt;
        uint64_t offset, size;
        if (!take(&rt, sizeof(rt)) || !take(&offset, sizeof(offset)) || !take(&size, sizeof(size))) {
            return false;
        }
        if (pos + size > input.size()) {
            return false;
        }
        ggml_tensor * tensor = deserialize_tensor(ctx, &rt);
        if (!bounds_ok(&rt, tensor, offset, size)) {
            GGML_LOG_ERROR("[%s] invalid SET segment\n", __func__);
            return false;
        }
        ggml_backend_tensor_set(tensor, input.data() + pos, offset, size);
        pos += size;
    }
    if (flags & 2) { // GRAPH chain (uid recomputes, in order)
        uint32_t n_uids;
        if (!take(&n_uids, sizeof(n_uids)) || n_uids == 0 || n_uids > 4096) {
            return false;
        }
        for (uint32_t k = 0; k < n_uids; k++) {
            uint64_t uid;
            if (!take(&uid, sizeof(uid))) {
                return false;
            }
            rpc_msg_graph_recompute_uid_req req = { device, 0, uid };
            if (!graph_recompute_uid(req)) {
                return false;
            }
        }
    }
    if (flags & 4) { // FETCH
        rpc_tensor rt;
        uint64_t offset, size;
        if (!take(&rt, sizeof(rt)) || !take(&offset, sizeof(offset)) || !take(&size, sizeof(size))) {
            return false;
        }
        ggml_tensor * tensor = deserialize_tensor(ctx, &rt);
        if (!bounds_ok(&rt, tensor, offset, size)) {
            GGML_LOG_ERROR("[%s] invalid FETCH segment\n", __func__);
            return false;
        }
        response.resize(size, 0);
        ggml_backend_tensor_get(tensor, response.data(), offset, size);
    }
    return true;
}

bool rpc_server::compute_contained(uint32_t device, ggml_cgraph * graph, const char * what) {
    ggml_status status = GGML_STATUS_FAILED;
    try {
        status = ggml_backend_graph_compute(backends[device], graph);
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("[%s] backend exception on device %u: %s\n", what, device, e.what());
    }
    if (status != GGML_STATUS_SUCCESS) {
        const int failures = consecutive_compute_failures.fetch_add(1) + 1;
        GGML_LOG_ERROR("[%s] graph computation failed on device %u (status %d) — "
                       "dropping this connection, the worker stays up (failure %d/3)\n",
                       what, device, (int) status, failures);
        if (failures >= 3) {
            GGML_LOG_ERROR("%d consecutive compute failures — the backend is likely unusable, "
                           "exiting so the restart policy brings up a clean process\n", failures);
            fflush(stderr);
            std::_Exit(1); // no atexit/CUDA teardown — the context may be poisoned and hang
        }
        return false;
    }
    consecutive_compute_failures.store(0);
    return true;
}

bool rpc_server::graph_recompute(const rpc_msg_graph_recompute_req & request) {
    uint32_t device = request.device;
    if (device >= backends.size()) {
        return false;
    }
    if (stored_graphs[device].graph == nullptr) {
        return false;
    }
    if (stored_graphs[device].owner_conn != current_conn) {
        // another connection replaced this device's stored graph - recomputing it
        // would silently run the wrong graph. Two coordinators sharing one device
        // is unsupported; fail loudly.
        GGML_LOG_ERROR("[%s] stored graph for device %u belongs to another connection\n", __func__, device);
        return false;
    }
    ggml_cgraph * graph = stored_graphs[device].graph;
    LOG_DBG("[%s] device: %u\n", __func__, device);
    if (!compute_contained(device, graph, __func__)) {
        stored_graphs[device].graph = nullptr; // never RECOMPUTE a graph that failed
        return false;
    }
    return true;
}

bool rpc_server::get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    size_t free, total;
    ggml_backend_dev_t dev = ggml_backend_get_device(backends[dev_id]);
    ggml_backend_dev_memory(dev, &free, &total);
    response.free_mem = free;
    response.total_mem = total;
    LOG_DBG("[%s] device: %u, free_mem: %" PRIu64 ", total_mem: %" PRIu64 "\n", __func__, dev_id, response.free_mem, response.total_mem);
    return true;
}

bool rpc_server::buffer_set_usage(const rpc_msg_buffer_set_usage_req & request) {
    LOG_DBG("[%s] remote_ptr: %" PRIx64 ", usage: %u\n", __func__, request.remote_ptr, request.usage);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    ggml_backend_buffer_set_usage(buffer, (enum ggml_backend_buffer_usage) request.usage);
    return true;
}

bool rpc_server::get_device_desc(const rpc_msg_get_device_desc_req & request, rpc_msg_get_device_desc_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backends[dev_id]);
    // a "CPU|" prefix tells the coordinator this device is worker RAM, not a GPU —
    // used by the KV-annex placement (TASKS.md #30); same pattern as "Meta[N](...)"
    const bool is_cpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    snprintf(response.desc, sizeof(response.desc), "%s%s",
             is_cpu ? "CPU|" : "", ggml_backend_dev_description(dev));
    return true;
}

rpc_server::~rpc_server() {
    for (auto buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
}

// GGML_RPC_TIMING (worker-side): per-command latency breakdown so the coordinator's
// end-to-end RTT (rpc_ep_stat ewma) can be decomposed into what the WORKER spends -
// lock-wait (cross-connection exec-mutex contention) vs exec+send (the handler,
// dominated by graph_compute for the EP boundary commands) - vs the residual
// (network RTT + coordinator-side). Answers "is the ~4-5ms boundary turnaround
// compute, contention, or wire?" (TASKS.md #28 attribution). Off by default.
static bool rpc_timing_enabled() {
    static const bool on = getenv("GGML_RPC_TIMING") != nullptr;
    return on;
}
static std::mutex g_rpc_timing_mutex;
struct rpc_cmd_timing { uint64_t count = 0, lock_us = 0, exec_us = 0, exec_max_us = 0; };
static rpc_cmd_timing g_rpc_timings[RPC_CMD_COUNT];
static uint64_t g_rpc_timing_total = 0;
static const char * rpc_cmd_str(int cmd) {
    switch (cmd) {
        case RPC_CMD_GRAPH_COMPUTE:        return "GRAPH_COMPUTE";
        case RPC_CMD_GRAPH_RECOMPUTE:      return "GRAPH_RECOMPUTE";
        case RPC_CMD_GRAPH_COMPUTE_UID:    return "GRAPH_COMPUTE_UID";
        case RPC_CMD_GRAPH_RECOMPUTE_UID:  return "GRAPH_RECOMPUTE_UID"; // meta per-boundary compute
        case RPC_CMD_GRAPH_FUSED:          return "GRAPH_FUSED";         // fused EP boundary
        case RPC_CMD_GET_TENSOR:           return "GET_TENSOR";          // star-reduce partial read
        case RPC_CMD_SET_TENSOR:           return "SET_TENSOR";
        case RPC_CMD_SET_TENSOR_HASH:      return "SET_TENSOR_HASH";
        case RPC_CMD_SET_TENSOR_HASH2:     return "SET_TENSOR_HASH2";
        case RPC_CMD_COPY_FROM_REMOTE:     return "COPY_FROM_REMOTE";    // W2W pull (butterfly)
        case RPC_CMD_GET_TENSOR_FENCED:    return "GET_TENSOR_FENCED";
        case RPC_CMD_PING:                 return "PING";
        default:                           return "other";
    }
}
static void rpc_timing_record(int cmd, int64_t t_recv, int64_t t_lock, int64_t t_done) {
    if (cmd < 0 || cmd >= RPC_CMD_COUNT) {
        return;
    }
    const uint64_t ex = (uint64_t) std::max<int64_t>(0, t_done - t_lock);
    std::lock_guard<std::mutex> lock(g_rpc_timing_mutex);
    rpc_cmd_timing & e = g_rpc_timings[cmd];
    e.count++;
    e.lock_us += (uint64_t) std::max<int64_t>(0, t_lock - t_recv);
    e.exec_us += ex;
    e.exec_max_us = std::max(e.exec_max_us, ex);
    // periodic dump so a long-running serve loop reports without needing to exit
    if (++g_rpc_timing_total % 20000 == 0) {
        for (int c = 0; c < RPC_CMD_COUNT; c++) {
            const rpc_cmd_timing & t = g_rpc_timings[c];
            if (t.count == 0) {
                continue;
            }
            GGML_LOG_INFO("[rpc-timing] %-20s n=%-7llu lock avg %.1f us  exec avg %.1f us  exec max %llu us\n",
                          rpc_cmd_str(c), (unsigned long long) t.count,
                          (double) t.lock_us / t.count, (double) t.exec_us / t.count,
                          (unsigned long long) t.exec_max_us);
        }
    }
}

static void rpc_serve_client_loop(rpc_server & server, socket_ptr sock, uint64_t conn_id) {
    uint8_t cmd;
    if (!sock->recv_data(&cmd, 1)) {
        return;
    }
    if (cmd != RPC_CMD_HELLO) {
        GGML_LOG_ERROR("Expected HELLO command, update client\n");
        return;
    }

    // Read input_size and validate protocol version
    uint64_t hello_input_size;
    if (!sock->recv_data(&hello_input_size, sizeof(hello_input_size))) {
        return;
    }

    if (hello_input_size != sizeof(rpc_msg_hello_req)) {
        GGML_LOG_ERROR("HELLO request size mismatch (%zu vs %zu) — client needs upgrade to protocol v%d.x\n",
                       (size_t)hello_input_size, sizeof(rpc_msg_hello_req), RPC_PROTO_MAJOR_VERSION);
        return;
    }

    rpc_msg_hello_req req = {};
    if (!sock->recv_data(&req, sizeof(req))) {
        return;
    }

    rpc_msg_hello_rsp rsp = {};
    server.hello(rsp);
    // Advertise server transport capabilities based on client's caps
    sock->get_caps(rsp.conn_caps);
    if (!send_msg(sock, &rsp, sizeof(rsp))) {
        return;
    }

    // Activate transport upgrade using client's caps
    sock->update_caps(req.conn_caps);
    server.conn_started(conn_id);
    while (true) {
        // idle wait for the next command happens without the execution lock so
        // other connections keep making progress
        if (!sock->recv_data(&cmd, 1)) {
            break;
        }
        const int64_t t_recv = rpc_timing_enabled() ? ggml_time_us() : 0; // GGML_RPC_TIMING
        if (cmd >= RPC_CMD_COUNT) {
            // fail fast if the command is invalid
            GGML_LOG_ERROR("Unknown command: %d\n", cmd);
            break;
        }
        if (cmd == RPC_CMD_GET_TENSOR_FENCED) {
            // the fence wait must happen WITHOUT the execution lock: the commands
            // it waits for need that lock to execute
            rpc_msg_get_tensor_fenced_req request;
            if (!recv_msg(sock, &request, sizeof(request))) {
                break;
            }
            if (!server.wait_fence(request.fence_conn, request.fence_seq)) {
                break;
            }
            std::lock_guard<std::mutex> exec_lock(server.exec_mutex);
            server.current_conn = conn_id;
            rpc_msg_get_tensor_req greq = { request.tensor, request.offset, request.size };
            std::vector<uint8_t> response;
            if (!server.get_tensor(greq, response)) {
                break;
            }
            if (!send_msg(sock, response.data(), response.size())) {
                break;
            }
            server.mark_executed(conn_id);
            continue;
        }
        if (cmd == RPC_CMD_COPY_FROM_REMOTE) {
            // the remote fetch must happen WITHOUT the execution lock: with two
            // crossing pulls in flight (batched reduce), each server's serving
            // thread needs the OTHER server's lock for the fenced read - holding
            // it across the fetch is an ABBA deadlock. copy_from_remote takes the
            // lock itself, only around the local write.
            std::vector<uint8_t> input;
            if (!recv_msg(sock, input)) {
                break;
            }
            rpc_msg_copy_from_remote_rsp response;
            if (!server.copy_from_remote(input, response)) {
                break;
            }
            if (!send_msg(sock, &response, sizeof(response))) {
                break;
            }
            server.mark_executed(conn_id);
            continue;
        }
        // proto 4.7 fleet commands run WITHOUT the execution lock: they only touch
        // their own mutex-guarded globals, and they must respond while a graph is
        // computing (a hung worker must still be inspectable and restartable)
        if (cmd == RPC_CMD_GET_LOG) {
            rpc_msg_get_log_req request;
            if (!recv_msg(sock, &request, sizeof(request))) {
                break;
            }
            std::string tail = rpc_log_tail(std::min<uint64_t>(request.max_bytes, RPC_LOG_RING_CAP));
            if (!send_msg(sock, tail.data(), tail.size())) {
                break;
            }
            server.mark_executed(conn_id);
            continue;
        }
        if (cmd == RPC_CMD_GET_SCORE) {
            if (!recv_msg(sock, nullptr, 0)) {
                break;
            }
            rpc_msg_get_score_rsp response = {};
            response.valid     = g_worker_score_valid.load() ? 1 : 0;
            response.bw_gbps   = g_worker_score_bw.load();
            response.mm_gflops = g_worker_score_fl.load();
            if (!send_msg(sock, &response, sizeof(response))) {
                break;
            }
            server.mark_executed(conn_id);
            continue;
        }
        if (cmd == RPC_CMD_SHUTDOWN) {
            rpc_msg_shutdown_req request;
            if (!recv_msg(sock, &request, sizeof(request))) {
                break;
            }
            rpc_msg_shutdown_rsp response = {};
            // must carry the magic AND the worker must have opted in (--allow-shutdown):
            // RPC is unauthenticated, so this stays off unless a supervisor will restart us
            const bool allowed = request.magic == RPC_SHUTDOWN_MAGIC && g_worker_shutdown_enabled.load();
            response.ok = allowed ? 1 : 0;
            if (!allowed && request.magic == RPC_SHUTDOWN_MAGIC) {
                GGML_LOG_WARN("%s", "refused an RPC shutdown request - start rpc-server with --allow-shutdown to permit it\n");
            }
            if (!send_msg(sock, &response, sizeof(response))) {
                break;
            }
            server.mark_executed(conn_id);
            if (allowed) {
                GGML_LOG_INFO("shutdown requested over RPC (conn %" PRIu64 ") - exiting for supervisor restart\n", conn_id);
                fflush(stdout);
                fflush(stderr);
                exit(0);
            }
            continue;
        }
        // one command executes at a time across all connections; the payload recv
        // inside each case is safe to hold the lock over - the client sends the
        // header and payload back-to-back
        std::lock_guard<std::mutex> exec_lock(server.exec_mutex);
        const int64_t t_lock = rpc_timing_enabled() ? ggml_time_us() : 0; // GGML_RPC_TIMING
        server.current_conn = conn_id;
        switch (cmd) {
            case RPC_CMD_HELLO: {
                // HELLO command is handled above
                return;
            }
            case RPC_CMD_DEVICE_COUNT: {
                if (!recv_msg(sock, nullptr, 0)) {
                    return;
                }
                rpc_msg_device_count_rsp response;
                response.device_count = server.device_count();
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_ALLOC_BUFFER: {
                rpc_msg_alloc_buffer_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_alloc_buffer_rsp response;
                if (!server.alloc_buffer(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_ALLOC_SIZE: {
                rpc_msg_get_alloc_size_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_alloc_size_rsp response;
                if (!server.get_alloc_size(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_ALIGNMENT: {
                rpc_msg_get_alignment_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_alignment_rsp response;
                if (!server.get_alignment(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_MAX_SIZE: {
                rpc_msg_get_max_size_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_max_size_rsp response;
                if (!server.get_max_size(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_BUFFER_GET_BASE: {
                rpc_msg_buffer_get_base_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_buffer_get_base_rsp response;
                if (!server.buffer_get_base(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_FREE_BUFFER: {
                rpc_msg_free_buffer_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.free_buffer(request)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_BUFFER_CLEAR: {
                rpc_msg_buffer_clear_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.buffer_clear(request)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.set_tensor(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR_HASH: {
                rpc_msg_set_tensor_hash_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_set_tensor_hash_rsp response;
                if (!server.set_tensor_hash(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR_HASH2: {
                rpc_msg_set_tensor_hash2_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_set_tensor_hash_rsp response;
                if (!server.set_tensor_hash2(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_INIT_TENSOR: {
                rpc_msg_init_tensor_req request;
                if (!recv_msg(sock, &request,sizeof(request))) {
                    return;
                }
                if (!server.init_tensor(request)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_PING: {
                // completion marker: commands execute in order, so by the time this
                // response is sent every previously received command has finished
                if (!recv_msg(sock, nullptr, 0)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_CONN_ID: {
                if (!recv_msg(sock, nullptr, 0)) {
                    return;
                }
                rpc_msg_get_conn_id_rsp response = { conn_id };
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_COPY_FROM_REMOTE: {
                // handled above, without the execution lock
                return;
            }
            case RPC_CMD_GET_TENSOR: {
                rpc_msg_get_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                std::vector<uint8_t> response;
                if (!server.get_tensor(request, response)) {
                    return;
                }
                if (!send_msg(sock, response.data(), response.size())) {
                    return;
                }
                break;
            }
            case RPC_CMD_COPY_TENSOR: {
                rpc_msg_copy_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_copy_tensor_rsp response;
                if (!server.copy_tensor(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_COMPUTE: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.graph_compute(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_RECOMPUTE: {
                rpc_msg_graph_recompute_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.graph_recompute(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_COMPUTE_UID: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.graph_compute_uid(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_RECOMPUTE_UID: {
                rpc_msg_graph_recompute_uid_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.graph_recompute_uid(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_FUSED: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                std::vector<uint8_t> response;
                if (!server.graph_fused(input, response)) {
                    return;
                }
                // a response exists only when the message carried a FETCH - a
                // SET+GRAPH message is fire-and-forget like its unfused parts
                if (!response.empty() && !send_msg(sock, response.data(), response.size())) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_DEVICE_MEMORY: {
                rpc_msg_get_device_memory_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_device_memory_rsp response;
                if (!server.get_device_memory(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_SPLIT_STATES: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!rpc_split_states_ingest(input.data(), input.size())) {
                    GGML_LOG_ERROR("failed to parse split states payload (%zu bytes)\n", input.size());
                    return;
                }
                break;
            }
            case RPC_CMD_BUFFER_SET_USAGE: {
                rpc_msg_buffer_set_usage_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.buffer_set_usage(request)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_DEVICE_DESC: {
                rpc_msg_get_device_desc_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_device_desc_rsp response;
                if (!server.get_device_desc(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            default: {
                GGML_LOG_ERROR("Unknown command: %d\n", cmd);
                return;
            }
        }
        // fence progress: in lockstep with the client's sent-command counter
        server.mark_executed(conn_id);
        if (rpc_timing_enabled()) { // GGML_RPC_TIMING: lock-wait + exec+send, per cmd
            rpc_timing_record(cmd, t_recv, t_lock, ggml_time_us());
        }
    }
}

static void rpc_timing_dump() {
    std::lock_guard<std::mutex> lock(g_rpc_timing_mutex);
    for (int c = 0; c < RPC_CMD_COUNT; c++) {
        const rpc_cmd_timing & t = g_rpc_timings[c];
        if (t.count == 0) {
            continue;
        }
        GGML_LOG_INFO("[rpc-timing] %-20s n=%-7llu lock avg %.1f us  exec avg %.1f us  exec max %llu us\n",
                      rpc_cmd_str(c), (unsigned long long) t.count,
                      (double) t.lock_us / t.count, (double) t.exec_us / t.count,
                      (unsigned long long) t.exec_max_us);
    }
}

static void rpc_serve_client(rpc_server & server, socket_ptr sock, uint64_t conn_id) {
    rpc_serve_client_loop(server, sock, conn_id);
    if (rpc_timing_enabled()) {
        rpc_timing_dump(); // per-connection close: short runs report without the 20k periodic
    }
    // runs regardless of how the loop exited (clean close or protocol error)
    server.conn_closed(conn_id); // wakes any fence waiting on this connection
    std::lock_guard<std::mutex> exec_lock(server.exec_mutex);
    server.free_owned(conn_id);
}

void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir, const char * model_dir,
                                   size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices) {
    if (n_devices == 0 || devices == nullptr) {
        fprintf(stderr, "Invalid arguments to ggml_backend_rpc_start_server\n");
        return;
    }
    std::vector<ggml_backend_t> backends;
    printf("Starting RPC server v%d.%d.%d\n",
        RPC_PROTO_MAJOR_VERSION,
        RPC_PROTO_MINOR_VERSION,
        RPC_PROTO_PATCH_VERSION);
    printf("  endpoint       : %s\n", endpoint);
    printf("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    printf("  local models   : %s\n", model_dir ? model_dir : "n/a");
    printf("Devices:\n");
    for (size_t i = 0; i < n_devices; i++) {
        auto dev = devices[i];
        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);
        printf("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
               total / 1024 / 1024, free / 1024 / 1024);
        auto backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            fprintf(stderr, "Failed to create backend for device %s\n", dev->iface.get_name(dev));
            return;
        }
        backends.push_back(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg) {
            auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (ggml_backend_set_n_threads_fn) {
                ggml_backend_set_n_threads_fn(backend, n_threads);
            }
            // GGML_RPC_THREADPOOL_POLL=N: attach a PERSISTENT threadpool (poll level
            // N, 0-100) to CPU-class backends. Without one, every graph_compute
            // creates and joins a disposable pool - thread create/join per subgraph
            // is a dominant share of the per-boundary turnaround for EP workers
            // (TASKS.md #28 attribution round 2). N>0 additionally busy-polls
            // between subgraphs, trading idle cores for wake latency.
            const char * poll_env = getenv("GGML_RPC_THREADPOOL_POLL");
            if (poll_env != nullptr && *poll_env != '\0') {
                typedef struct ggml_threadpool * (*threadpool_new_t)(struct ggml_threadpool_params *);
                typedef void (*set_threadpool_t)(ggml_backend_t, struct ggml_threadpool *);
                auto threadpool_new_fn = (threadpool_new_t) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
                auto set_threadpool_fn = (set_threadpool_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
                if (threadpool_new_fn && set_threadpool_fn) {
                    struct ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads);
                    tpp.poll = std::max(0, std::min(100, atoi(poll_env)));
                    struct ggml_threadpool * tp = threadpool_new_fn(&tpp);
                    if (tp != nullptr) {
                        set_threadpool_fn(backend, tp); // lives for the server's lifetime
                        printf("  %s: persistent threadpool attached (%zu threads, poll %d)\n",
                               ggml_backend_dev_name(dev), n_threads, tpp.poll);
                    }
                }
            }
        }
    }

    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        return;
    }

#ifdef GGML_RPC_RDMA
    printf("  transport      : TCP (RDMA auto-negotiate enabled)\n");
#else
    printf("  transport      : TCP\n");
#endif // GGML_RPC_RDMA
    if (!rpc_transport_init()) {
        fprintf(stderr, "Failed to initialize RPC transport\n");
        return;
    }
    auto server_socket = socket_t::create_server(host.c_str(), port);
    if (server_socket == nullptr) {
        fprintf(stderr, "Failed to create server socket\n");
        return;
    }
    // TASKS.md #45: the LRU cap otherwise only enforces on writes, so an idle
    // worker keeps an oversized cache (stale model generations) indefinitely
    if (cache_dir != nullptr) {
        rpc_cache_enforce_limit(cache_dir);
    }

    // one server instance shared by all connections (buffers, stored graphs);
    // each connection is served by its own thread, execution is serialized by
    // the server's exec_mutex
    rpc_server server(backends, cache_dir);
    if (model_dir != nullptr) {
        server.build_local_index(model_dir);
    }
    uint64_t next_conn_id = 1;
    int accept_fails = 0;
    while (true) {
        auto client_socket = server_socket->accept();
        if (client_socket == nullptr) {
            // transient (EMFILE under fd pressure, ECONNABORTED, ...) must not kill a
            // worker holding a model share - retry; a listener that never recovers
            // still exits for the supervisor restart
            accept_fails++;
            if (accept_fails > 300) {
                fprintf(stderr, "accept failed %d times in a row - giving up for supervisor restart\n", accept_fails);
                return;
            }
            if (accept_fails == 1 || accept_fails % 50 == 0) {
                GGML_LOG_ERROR("Failed to accept client connection (attempt %d, retrying)\n", accept_fails);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            continue;
        }
        accept_fails = 0;
        const uint64_t conn_id = next_conn_id++;
        // GGML_LOG so the lines reach the fleet log ring (rpc-server's log callback)
        GGML_LOG_INFO("Accepted client connection %" PRIu64 "\n", conn_id);
        std::thread([&server, client_socket, conn_id]() {
            rpc_serve_client(server, client_socket, conn_id);
            GGML_LOG_INFO("Client connection %" PRIu64 " closed\n", conn_id);
        }).detach();
    }
    rpc_transport_shutdown();
    for (auto backend : backends) {
        ggml_backend_free(backend);
    }
}

// LAN presence beacon + discovery (TASKS.md #29d). The payload is a single
// versioned text line; the coordinator derives the worker's IP from the
// datagram source address, so only the port needs to be advertised. The id
// token identifies the worker PROCESS: a multi-homed worker beacons on every
// interface and would otherwise be discovered once per interface.
static constexpr const char * RPC_ANNOUNCE_MAGIC = "llama.cpp-rpc/1";

bool ggml_backend_rpc_start_announcer(const char * endpoint, const char * group,
                                      size_t n_devices, ggml_backend_dev_t * devices,
                                      const char * cache_dir) {
    std::string host;
    int port = 0;
    if (!parse_endpoint(endpoint, host, port)) {
        GGML_LOG_ERROR("invalid endpoint '%s'\n", endpoint);
        return false;
    }
    const uint64_t instance_id = [] {
        std::random_device rd;
        return ((uint64_t) rd() << 32) | rd();
    }();
    std::vector<ggml_backend_dev_t> devs(devices, devices + n_devices);
    const std::string cache_dir_str = cache_dir ? cache_dir : "";
    auto make_payload = [port, instance_id, devs, cache_dir_str]() {
        size_t free_total = 0;
        for (auto dev : devs) {
            size_t free = 0, total = 0;
            ggml_backend_dev_memory(dev, &free, &total);
            free_total += free;
        }
        char buf[256];
        int len = snprintf(buf, sizeof(buf), "%s port=%d id=%016" PRIx64 " proto=%d.%d devs=%zu free_mib=%zu",
                 RPC_ANNOUNCE_MAGIC, port, instance_id, RPC_PROTO_MAJOR_VERSION, RPC_PROTO_MINOR_VERSION,
                 devs.size(), free_total / (1024 * 1024));
        // speed score (TASKS.md #35f), present once the startup benchmark published it;
        // extra key=value tokens are ignored by older coordinators
        if (g_worker_score_valid.load() && len > 0 && (size_t) len < sizeof(buf)) {
            len += snprintf(buf + len, sizeof(buf) - len, " bw_gbps=%.2f mm_gflops=%.1f",
                     g_worker_score_bw.load(), g_worker_score_fl.load());
        }
        // tensor-cache size (TASKS.md #45): the dir walk is cheap but not free, so
        // refresh at most every 30 s (beacons fire every ~2 s)
        if (!cache_dir_str.empty() && len > 0 && (size_t) len < sizeof(buf)) {
            static std::atomic<uint64_t> cached_mib{0};
            static std::atomic<int64_t>  cached_at_ms{-1};
            const int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            if (cached_at_ms.load() < 0 || now_ms - cached_at_ms.load() > 30000) {
                uint64_t total = 0;
                std::error_code ec;
                for (auto it = fs::directory_iterator(cache_dir_str, fs::directory_options::skip_permission_denied, ec);
                     !ec && it != fs::directory_iterator(); it.increment(ec)) {
                    if (it->is_regular_file(ec)) {
                        total += it->file_size(ec);
                    }
                }
                cached_mib.store(total / (1024 * 1024));
                cached_at_ms.store(now_ms);
            }
            snprintf(buf + len, sizeof(buf) - len, " cache_mib=%" PRIu64, cached_mib.load());
        }
        return std::string(buf);
    };
    return rpc_announce_start(group, host.c_str(), std::move(make_payload));
}

int ggml_backend_rpc_discover(const char * group, int timeout_ms,
                              void (*cb)(const char * endpoint, const char * payload, void * user_data),
                              void * user_data) {
    const auto found = rpc_discover_listen(group, timeout_ms);
    const std::string magic  = std::string(RPC_ANNOUNCE_MAGIC) + " ";

    // group candidate endpoints by worker instance (multi-homed workers beacon from
    // several source IPs); beacons without an id token (older workers) group by endpoint
    struct candidate_group {
        std::vector<std::string> endpoints;
        std::string              payload;
    };
    std::vector<std::string> order; // stable output order
    std::unordered_map<std::string, candidate_group> groups;
    for (const auto & [ip, payload] : found) {
        if (payload.compare(0, magic.size(), magic) != 0) {
            continue; // not a worker beacon
        }
        const size_t pos = payload.find(" port=");
        if (pos == std::string::npos) {
            continue;
        }
        const int port = atoi(payload.c_str() + pos + 6);
        if (port <= 0 || port > 65535) {
            continue;
        }
        const std::string endpoint = ip + ":" + std::to_string(port);
        const size_t id_pos = payload.find(" id=");
        const std::string key = id_pos != std::string::npos ? payload.substr(id_pos + 4, 16) : endpoint;
        auto [it, inserted] = groups.try_emplace(key);
        if (inserted) {
            order.push_back(key);
            it->second.payload = payload;
        }
        if (std::find(it->second.endpoints.begin(), it->second.endpoints.end(), endpoint) == it->second.endpoints.end()) {
            it->second.endpoints.push_back(endpoint);
        }
    }

    // probe each instance's candidate addresses with a short TCP connect and report the
    // first reachable one — a beaconing-but-firewalled worker must not stall the load
    // behind a minutes-long blocking connect later
    int n = 0;
    for (const auto & key : order) {
        const candidate_group & g = groups.at(key);
        bool reported = false;
        for (const std::string & endpoint : g.endpoints) {
            std::string host;
            int port = 0;
            if (!parse_endpoint(endpoint, host, port)) {
                continue;
            }
            if (!rpc_probe_endpoint(host.c_str(), port, 3000)) {
                GGML_LOG_WARN("discovered RPC worker %s does not accept connections "
                              "(firewall blocking TCP %d?), skipping this address\n", endpoint.c_str(), port);
                continue;
            }
            cb(endpoint.c_str(), g.payload.c_str(), user_data);
            n++;
            reported = true;
            break;
        }
        if (!reported && !g.endpoints.empty()) {
            GGML_LOG_WARN("discovered RPC worker (beacon '%s') is unreachable on all %zu advertised addresses\n",
                          g.payload.c_str(), g.endpoints.size());
        }
    }
    return n;
}

// fleet ops (proto 4.7, TASKS.md #35). Each uses a dedicated ephemeral connection:
// the cached compute socket may be busy inside a long graph_compute round trip and
// its stream must never carry interleaved responses.

bool ggml_backend_rpc_fetch_log(const char * endpoint, char * buf, size_t buf_size, size_t * out_len) {
    if (endpoint == nullptr || buf == nullptr || buf_size == 0 || out_len == nullptr) {
        return false;
    }
    *out_len = 0;
    auto sock = rpc_connect_ephemeral(endpoint);
    if (sock == nullptr) {
        return false;
    }
    rpc_ephemeral_guard guard{ sock.get() };
    ggml_backend_rpc_async_state & st = rpc_async_state(sock.get());
    if (st.server_minor < 7) {
        GGML_LOG_WARN("worker %s speaks proto 4.%d < 4.7, no log fetch\n", endpoint, st.server_minor);
        return false;
    }
    rpc_msg_get_log_req request = { buf_size };
    std::lock_guard<std::mutex> lock(st.mutex);
    if (!send_rpc_cmd_raw(sock, st, RPC_CMD_GET_LOG, &request, sizeof(request))) {
        return false;
    }
    uint64_t out_size;
    if (!sock->recv_data(&out_size, sizeof(out_size))) {
        return false;
    }
    if (out_size > buf_size) {
        return false; // server ignored max_bytes - do not trust the stream
    }
    if (out_size > 0 && !sock->recv_data(buf, out_size)) {
        return false;
    }
    *out_len = out_size;
    return true;
}

// free/total device memory over a THROWAWAY connection - never the cached compute
// socket. The fleet-status UI poll must not share the compute socket: a synchronous
// GET_DEVICE_MEMORY racing the decode path's in-flight async traffic desyncs the
// framing, and get_device_memory's own failure handler would then rpc_mark_failed()
// the endpoint - poisoning a HEALTHY worker mid-serve (TASKS.md #47). This path
// touches neither the socket cache nor the failed-endpoint set.
static const char * ggml_backend_rpc_device_get_name(ggml_backend_dev_t dev); // defined below

bool ggml_backend_rpc_dev_memory_ephemeral(ggml_backend_dev_t dev, int timeout_ms,
                                           size_t * free, size_t * total) {
    if (free)  { *free  = 0; }
    if (total) { *total = 0; }
    if (dev == nullptr || dev->iface.get_name != ggml_backend_rpc_device_get_name) {
        return false;
    }
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *) dev->context;
    const std::string & endpoint = ctx->endpoint;
    const uint32_t device = ctx->device;
    if (!ggml_backend_rpc_endpoint_reachable(endpoint.c_str(), timeout_ms)) {
        return false; // bounded probe first: never sink the caller into a dead connect
    }
    auto sock = rpc_connect_ephemeral(endpoint);
    if (sock == nullptr) {
        return false;
    }
    rpc_ephemeral_guard guard{ sock.get() };
    rpc_msg_get_device_memory_req request;
    request.device = device;
    rpc_msg_get_device_memory_rsp response;
    // send + read with the standard fixed-response framing (| size(8) | data |);
    // this is a fresh connection with no in-flight commands, so no ping drain needed
    if (!send_rpc_cmd(sock, RPC_CMD_GET_DEVICE_MEMORY, &request, sizeof(request), &response, sizeof(response))) {
        return false;
    }
    if (free)  { *free  = response.free_mem; }
    if (total) { *total = response.total_mem; }
    return true;
}

bool ggml_backend_rpc_shutdown_worker(const char * endpoint) {
    if (endpoint == nullptr) {
        return false;
    }
    auto sock = rpc_connect_ephemeral(endpoint);
    if (sock == nullptr) {
        return false;
    }
    rpc_ephemeral_guard guard{ sock.get() };
    if (rpc_async_state(sock.get()).server_minor < 7) {
        GGML_LOG_WARN("worker %s speaks proto 4.%d < 4.7, no remote shutdown\n",
                      endpoint, rpc_async_state(sock.get()).server_minor);
        return false;
    }
    rpc_msg_shutdown_req request  = { RPC_SHUTDOWN_MAGIC };
    rpc_msg_shutdown_rsp response = {};
    if (!send_rpc_cmd(sock, RPC_CMD_SHUTDOWN, &request, sizeof(request), &response, sizeof(response))) {
        return false;
    }
    return response.ok != 0;
}

bool ggml_backend_rpc_get_worker_score(const char * endpoint, float * bw_gbps, float * mm_gflops) {
    if (endpoint == nullptr || bw_gbps == nullptr || mm_gflops == nullptr) {
        return false;
    }
    auto sock = rpc_connect_ephemeral(endpoint);
    if (sock == nullptr) {
        return false;
    }
    rpc_ephemeral_guard guard{ sock.get() };
    if (rpc_async_state(sock.get()).server_minor < 7) {
        return false;
    }
    rpc_msg_get_score_rsp response = {};
    if (!send_rpc_cmd(sock, RPC_CMD_GET_SCORE, nullptr, 0, &response, sizeof(response))) {
        return false;
    }
    if (!response.valid) {
        return false;
    }
    *bw_gbps   = response.bw_gbps;
    *mm_gflops = response.mm_gflops;
    return true;
}

// device speed benchmark (TASKS.md #31/#35f): a chain of large f32 matvecs. A
// matvec streams the whole matrix once per multiply, so tokens/s here IS the
// device's effective memory bandwidth - the same quantity that bounds decode -
// and the measurement is identical in kind for CPU and GPU devices.
bool ggml_backend_rpc_benchmark_device(ggml_backend_dev_t dev, float * bw_gbps, float * mm_gflops) {
    if (dev == nullptr || bw_gbps == nullptr || mm_gflops == nullptr) {
        return false;
    }
    size_t free_mem = 0, total_mem = 0;
    ggml_backend_dev_memory(dev, &free_mem, &total_mem);
    // 4096^2 f32 = 64 MiB; fall back to 2048^2 (16 MiB) on tiny devices
    const int64_t n     = (free_mem == 0 || free_mem > 96 * 1024 * 1024) ? 4096 : 2048;
    const int     chain = 8;

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (backend == nullptr) {
        return false;
    }
    // CPU-class devices: bench with all cores, like real serving
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg != nullptr) {
        auto set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (set_n_threads_fn != nullptr) {
            set_n_threads_fn(backend, std::max(1u, std::thread::hardware_concurrency()));
        }
    }

    ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (chain + 4) + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(ip) };
    ggml_context * ctx = ctx_ptr.get();
    if (ctx == nullptr) {
        ggml_backend_free(backend);
        return false;
    }
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n);
    ggml_tensor * cur = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    for (int i = 0; i < chain; i++) {
        cur = ggml_mul_mat(ctx, w, cur);
        ggml_build_forward_expand(gf, cur);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_dev_buffer_type(dev));
    if (buf == nullptr) {
        ggml_backend_free(backend);
        return false;
    }
    // fill with 0x3c bytes: 0x3c3c3c3c is a small normal f32 (~0.011), avoiding
    // both denormal stalls and all-zero fast paths
    ggml_backend_buffer_clear(buf, 0x3c);

    bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS; // warmup
    int  reps = 0;
    const auto t0 = std::chrono::steady_clock::now();
    double elapsed_s = 0.0;
    while (ok && reps < 64) {
        ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
        reps++;
        elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (elapsed_s > 0.25) {
            break;
        }
    }
    if (ok && reps > 0 && elapsed_s > 0.0) {
        const double bytes = (double) n * n * sizeof(float) * chain * reps;
        const double flops = 2.0 * n * n * chain * reps;
        *bw_gbps   = (float) (bytes / elapsed_s / 1e9);
        *mm_gflops = (float) (flops / elapsed_s / 1e9);
    } else {
        ok = false;
    }
    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    return ok;
}

static const char * ggml_backend_rpc_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ctx->name.c_str();
}

static const char * ggml_backend_rpc_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ctx->description.c_str();
}

static void ggml_backend_rpc_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ggml_backend_rpc_get_device_memory(ctx->endpoint.c_str(), ctx->device, free, total);
}

static enum ggml_backend_dev_type ggml_backend_rpc_device_get_type(ggml_backend_dev_t dev) {
    // TODO: obtain value from the server
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_rpc_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rpc_device_get_name(dev);
    props->description = ggml_backend_rpc_device_get_description(dev);
    props->type        = ggml_backend_rpc_device_get_type(dev);
    ggml_backend_rpc_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ true,
    };
}

static ggml_backend_event_t ggml_backend_rpc_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    ggml_backend_rpc_event_context * ectx = new ggml_backend_rpc_event_context {
        /* .endpoint = */ ctx->endpoint,
        /* .seq      = */ 0,
    };
    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ ectx,
    };
}

static void ggml_backend_rpc_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    delete (ggml_backend_rpc_event_context *)event->context;
    delete event;
    GGML_UNUSED(dev);
}

static void ggml_backend_rpc_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_event_context * ectx = (ggml_backend_rpc_event_context *)event->context;
    if (ectx->seq == 0) {
        return;
    }
    auto sock = get_socket(ectx->endpoint);
    if (sock == nullptr) {
        rpc_mark_failed(ectx->endpoint, __func__);
        return;
    }
    rpc_sync_pings(sock, ectx->seq);
    GGML_UNUSED(dev);
}

static ggml_backend_t ggml_backend_rpc_device_init(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ggml_backend_rpc_init(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_rpc_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ggml_backend_rpc_buffer_type(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(dev);
}

static bool ggml_backend_rpc_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    //TODO: call the remote backend and cache the results
    return true;
}

static bool ggml_backend_rpc_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (!buft || buft->iface.get_name != ggml_backend_rpc_buffer_type_name) {
        return false;
    }
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *)dev->context;
    return buft_ctx->endpoint == dev_ctx->endpoint && buft_ctx->device == dev_ctx->device;
}

const char * ggml_backend_rpc_dev_endpoint(ggml_backend_dev_t dev) {
    if (dev == nullptr || dev->iface.get_name != ggml_backend_rpc_device_get_name) {
        return nullptr;
    }
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *) dev->context;
    return ctx->endpoint.c_str();
}

// true when this RPC device is backed by the WORKER's CPU (its RAM) rather than a GPU.
// The device still reports GGML_BACKEND_DEVICE_TYPE_GPU so llama schedules layers onto
// pure-CPU workers exactly as before — this side channel exists for placement policies
// that need the truth (TASKS.md #30 KV annex).
bool ggml_backend_rpc_dev_worker_is_cpu(ggml_backend_dev_t dev) {
    if (dev == nullptr || dev->iface.get_name != ggml_backend_rpc_device_get_name) {
        return false;
    }
    return ((ggml_backend_rpc_device_context *) dev->context)->worker_is_cpu;
}

// forget past endpoint failures so an in-process reload can reconnect (TASKS.md #29c).
// Sockets are held only by buffer contexts, so once the model is destroyed the cached
// connections expire and the next use dials fresh.
void ggml_backend_rpc_reset_failed_endpoints(void) {
    std::lock_guard<std::mutex> lock(g_rpc_failed_mutex);
    if (!g_rpc_failed_endpoints.empty()) {
        GGML_LOG_INFO("RPC: clearing %zu failed endpoint mark(s) for reload\n", g_rpc_failed_endpoints.size());
        g_rpc_failed_endpoints.clear();
    }
}

// probe: can this RPC device's endpoint be (re)connected right now?
bool ggml_backend_rpc_dev_reachable(ggml_backend_dev_t dev) {
    if (dev == nullptr || dev->iface.get_name != ggml_backend_rpc_device_get_name) {
        return true; // not an RPC device: nothing to probe
    }
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *) dev->context;
    // dial FRESH: the cached socket can be a dead one still held alive by old
    // buffers - a restarted-and-healthy worker then reads as unreachable, drops
    // its -ts share, and the split collapses onto the survivors (TASKS.md #43b).
    // A successful fresh dial also replaces the cache entry the reload then uses.
    return get_socket_impl(ctx->endpoint, /*fresh =*/ true) != nullptr;
}

// true when this RPC device's endpoint has failed this session
bool ggml_backend_rpc_dev_failed(ggml_backend_dev_t dev) {
    if (dev == nullptr || dev->iface.get_name != ggml_backend_rpc_device_get_name) {
        return false;
    }
    return rpc_endpoint_failed(((ggml_backend_rpc_device_context *) dev->context)->endpoint);
}

// dial a FRESH connection (the cached socket may be a dead one still held by
// buffers); success replaces the cache entry, ready for re-provisioning
bool ggml_backend_rpc_endpoint_reprobe(const char * endpoint) {
    return endpoint != nullptr && get_socket_impl(endpoint, /*fresh =*/ true) != nullptr;
}

void ggml_backend_rpc_clear_failed_endpoint(const char * endpoint) {
    if (endpoint == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_rpc_failed_mutex);
    g_rpc_failed_endpoints.erase(endpoint);
}

// endpoint of an RPC buffer, null for buffers of any other backend
const char * ggml_backend_rpc_buffer_endpoint(ggml_backend_buffer_t buffer) {
    if (buffer == nullptr || !ggml_backend_buffer_is_rpc(buffer)) {
        return nullptr;
    }
    return ((ggml_backend_rpc_buffer_type_context *) buffer->buft->context)->endpoint.c_str();
}

// surgical re-provision (TASKS.md #29c refinement): re-create this buffer's remote
// allocation on a freshly reconnected endpoint (same address, restarted worker) and
// replay its weight journal - big writes via SET_TENSOR_HASH against the worker's
// local cache, small writes literally. The buffer context is rebased IN PLACE; the
// old and new bases are returned so the caller (the meta backend) can rebase its
// shadow registrations. Buffers without a journal (KV/compute members) are
// re-allocated empty. Returns false when anything cannot be restored.
bool ggml_backend_rpc_buffer_reprovision(ggml_backend_buffer_t buffer, void ** old_base, void ** new_base) {
    if (buffer == nullptr || !ggml_backend_buffer_is_rpc(buffer)) {
        return false;
    }
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *) buffer->context;
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *) buffer->buft->context;
    const std::string & endpoint = buft_ctx->endpoint;

    // reuse the cache entry when a reprobe already dialed a fresh connection;
    // dial one ourselves if the cache still holds this buffer's dead socket
    std::shared_ptr<socket_t> sock = get_socket(endpoint);
    if (sock == nullptr || sock.get() == ctx->sock.get()) {
        sock = get_socket_impl(endpoint, /*fresh =*/ true);
    }
    if (sock == nullptr) {
        return false;
    }

    rpc_msg_alloc_buffer_req alloc_req = { buft_ctx->device, ggml_backend_buffer_get_size(buffer) };
    rpc_msg_alloc_buffer_rsp alloc_rsp;
    if (!send_rpc_cmd(sock, RPC_CMD_ALLOC_BUFFER, &alloc_req, sizeof(alloc_req), &alloc_rsp, sizeof(alloc_rsp)) ||
        alloc_rsp.remote_ptr == 0) {
        return false;
    }
    rpc_msg_buffer_get_base_req base_req = { alloc_rsp.remote_ptr };
    rpc_msg_buffer_get_base_rsp base_rsp;
    if (!send_rpc_cmd(sock, RPC_CMD_BUFFER_GET_BASE, &base_req, sizeof(base_req), &base_rsp, sizeof(base_rsp))) {
        return false;
    }

    const std::shared_ptr<socket_t> old_sock  = ctx->sock;
    const uint64_t                  old_remote = ctx->remote_ptr;
    void *                          old_base_p = ctx->base_ptr;

    *old_base = ctx->base_ptr;
    *new_base = reinterpret_cast<void *>(base_rsp.base_ptr);
    ctx->sock       = sock;
    ctx->remote_ptr = alloc_rsp.remote_ptr;
    ctx->base_ptr   = *new_base;

    // a failed replay must FREE the fresh allocation and restore the old handles:
    // leaving it sits a partially-filled share on the worker until the fallback
    // reload's unload, doubling resident memory in the window (TASKS.md #43e)
    auto abandon = [&]() {
        rpc_msg_free_buffer_req freq = { ctx->remote_ptr };
        send_rpc_cmd(sock, RPC_CMD_FREE_BUFFER, &freq, sizeof(freq), nullptr, 0);
        ctx->sock       = old_sock;
        ctx->remote_ptr = old_remote;
        ctx->base_ptr   = old_base_p;
        return false;
    };

    rpc_msg_buffer_set_usage_req usage_req = { ctx->remote_ptr, (uint32_t) buffer->usage };
    if (!send_rpc_cmd(sock, RPC_CMD_BUFFER_SET_USAGE, &usage_req, sizeof(usage_req), nullptr, 0)) {
        return abandon();
    }

    std::lock_guard<std::mutex> lock(g_rpc_journal_mutex);
    auto it = g_rpc_journals.find(buffer);
    if (it == g_rpc_journals.end()) {
        return true; // stateless member buffer (KV slice / compute arena): realloc is enough
    }
    rpc_buffer_journal & j = it->second;
    if (j.disabled) {
        return abandon();
    }
    const uint64_t nb = (uint64_t)(uintptr_t) *new_base;
    if (j.cleared) {
        rpc_msg_buffer_clear_req creq = { ctx->remote_ptr, j.clear_value };
        if (!send_rpc_cmd(sock, RPC_CMD_BUFFER_CLEAR, &creq, sizeof(creq), nullptr, 0)) {
            return abandon();
        }
    }
    for (rpc_tensor jt : j.init_tensors) {
        rpc_msg_init_tensor_req req;
        req.tensor = jt;
        req.tensor.data  += nb;
        req.tensor.buffer = ctx->remote_ptr; // the journal recorded the DEAD allocation's id
        if (!send_rpc_cmd(sock, RPC_CMD_INIT_TENSOR, &req, sizeof(req), nullptr, 0)) {
            return abandon();
        }
    }
    size_t   n_hash  = 0;
    size_t   n_small = 0;
    uint64_t replayed_bytes = 0;
    uint64_t next_progress  = 4ULL * 1024 * 1024 * 1024;
    for (auto & kv : j.sets) {
        const rpc_buffer_journal::set_entry & e = kv.second;
        // a share-sized replay runs minutes (worker-disk + hash bound): make it
        // visible - a silent surgical looked like a coordinator hang (TASKS.md #43a)
        replayed_bytes += e.size;
        if (replayed_bytes >= next_progress) {
            GGML_LOG_INFO("[%s] %s: replay progress %.1f / %.1f GiB\n", __func__, endpoint.c_str(),
                          replayed_bytes / (1024.0*1024*1024),
                          ggml_backend_buffer_get_size(buffer) / (1024.0*1024*1024));
            next_progress += 4ULL * 1024 * 1024 * 1024;
        }
        rpc_tensor root = e.root;
        root.data  += nb;
        root.buffer = ctx->remote_ptr; // the journal recorded the DEAD allocation's id
        if (e.spill_off == UINT64_MAX) {
            rpc_msg_set_tensor_hash_rsp rsp;
            bool ok;
            if (e.src_name[0] != '\0' && rpc_async_state(sock.get()).server_minor >= 8) {
                // slice-sourced entries never touched the worker's disk cache -
                // replay through the provenance path (TASKS.md #44)
                rpc_msg_set_tensor_hash2_req req;
                memset(&req, 0, sizeof(req));
                req.tensor         = root;
                req.offset         = e.offset;
                req.hash           = e.hash;
                req.src_offset     = e.src_offset;
                req.src_row_size   = e.src_row_size;
                req.src_n_rows     = e.src_n_rows;
                req.src_row_stride = e.src_row_stride;
                memcpy(req.src_name, e.src_name, sizeof(req.src_name));
                ok = send_rpc_cmd(sock, RPC_CMD_SET_TENSOR_HASH2, &req, sizeof(req), &rsp, sizeof(rsp)) && rsp.result;
            } else {
                rpc_msg_set_tensor_hash_req req;
                req.tensor = root;
                req.offset = e.offset;
                req.hash   = e.hash;
                ok = send_rpc_cmd(sock, RPC_CMD_SET_TENSOR_HASH, &req, sizeof(req), &rsp, sizeof(rsp)) && rsp.result;
            }
            if (!ok) {
                // the restarted worker's cache does not hold these bytes - only a
                // full reload can restore them
                GGML_LOG_WARN("[%s] worker cache miss for hash 0x%" PRIx64 " - falling back to reload\n",
                              __func__, e.hash);
                return abandon();
            }
            n_hash++;
        } else {
            std::vector<uint8_t> input(sizeof(rpc_tensor) + sizeof(uint64_t) + e.size);
            memcpy(input.data(), &root, sizeof(root));
            memcpy(input.data() + sizeof(root), &e.offset, sizeof(e.offset));
            if (fseeko(j.spill, (off_t) e.spill_off, SEEK_SET) != 0 ||
                fread(input.data() + sizeof(root) + sizeof(e.offset), 1, e.size, j.spill) != e.size) {
                return abandon();
            }
            if (!send_rpc_cmd(sock, RPC_CMD_SET_TENSOR, input.data(), input.size())) {
                return abandon();
            }
            n_small++;
        }
    }
    // GGML_RPC_REPROVISION_VERIFY=1: read every replayed region back and compare
    // hashes - a diagnostic for the replay itself (adds a full read of the share)
    static const bool verify = getenv("GGML_RPC_REPROVISION_VERIFY") != nullptr;
    if (verify) {
        size_t n_bad = 0;
        std::vector<uint8_t> readback;
        for (auto & kv : j.sets) {
            const rpc_buffer_journal::set_entry & e = kv.second;
            rpc_tensor root = e.root;
            root.data  += nb;
            root.buffer = ctx->remote_ptr;
            readback.resize(e.size);
            rpc_msg_get_tensor_req greq;
            greq.tensor = root;
            greq.offset = e.offset;
            greq.size   = e.size;
            if (!send_rpc_cmd(sock, RPC_CMD_GET_TENSOR, &greq, sizeof(greq), readback.data(), e.size)) {
                return abandon();
            }
            uint64_t want = e.hash;
            if (e.spill_off != UINT64_MAX) {
                std::vector<uint8_t> spill_bytes(e.size);
                if (fseeko(j.spill, (off_t) e.spill_off, SEEK_SET) != 0 ||
                    fread(spill_bytes.data(), 1, e.size, j.spill) != e.size) {
                    return abandon();
                }
                want = fnv_hash(spill_bytes.data(), e.size);
            }
            if (fnv_hash(readback.data(), e.size) != want) {
                GGML_LOG_ERROR("[%s] VERIFY MISMATCH at offset %" PRIu64 " size %" PRIu64 "\n",
                               __func__, e.offset, e.size);
                n_bad++;
            }
        }
        GGML_LOG_INFO("[%s] verify: %zu regions, %zu mismatches\n", __func__, j.sets.size(), n_bad);
        if (n_bad > 0) {
            return abandon();
        }
    }
    // coverage is the tripwire for journal gaps: a weight buffer far below ~100%
    // means writes were lost (or bypassed the journal) and the replay left fresh
    // uninitialized bytes where weights should be
    uint64_t covered = 0;
    for (auto & kv : j.sets) {
        covered += kv.second.size;
    }
    GGML_LOG_INFO("[%s] %s: buffer %zu MiB restored (%zu cache-replayed, %zu literal writes, "
                  "journal covers %.1f%%)\n",
                  __func__, endpoint.c_str(), ggml_backend_buffer_get_size(buffer) / (1024*1024), n_hash, n_small,
                  100.0 * (double) covered / (double) ggml_backend_buffer_get_size(buffer));
    return true;
}

static const struct ggml_backend_device_i ggml_backend_rpc_device_i = {
    /* .get_name             = */ ggml_backend_rpc_device_get_name,
    /* .get_description      = */ ggml_backend_rpc_device_get_description,
    /* .get_memory           = */ ggml_backend_rpc_device_get_memory,
    /* .get_type             = */ ggml_backend_rpc_device_get_type,
    /* .get_props            = */ ggml_backend_rpc_device_get_props,
    /* .init_backend         = */ ggml_backend_rpc_device_init,
    /* .get_buffer_type      = */ ggml_backend_rpc_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_rpc_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rpc_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ ggml_backend_rpc_device_event_new,
    /* .event_free           = */ ggml_backend_rpc_device_event_free,
    /* .event_synchronize    = */ ggml_backend_rpc_device_event_synchronize,
};

// backend reg interface

struct ggml_backend_rpc_reg_context {
    std::string                     name;
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_rpc_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    return ctx ? ctx->name.c_str() : "RPC";
}

static size_t ggml_backend_rpc_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    return ctx ? ctx->devices.size() : 0;
}

static ggml_backend_dev_t ggml_backend_rpc_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    if (ctx == nullptr) {
        GGML_ABORT("The RPC backend does not have enumerated devices - use ggml_backend_rpc_add_server instead");
    } else {
        GGML_ASSERT(index < ctx->devices.size());
        return ctx->devices[index];
    }
}

static void * ggml_backend_rpc_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_rpc_add_server") == 0) {
        return (void *)ggml_backend_rpc_add_server;
    }
    if (std::strcmp(name, "ggml_backend_rpc_start_server") == 0) {
        return (void *)ggml_backend_rpc_start_server;
    }
    if (std::strcmp(name, "ggml_backend_rpc_dev_endpoint") == 0) {
        return (void *)ggml_backend_rpc_dev_endpoint;
    }
    if (std::strcmp(name, "ggml_backend_rpc_set_split_states") == 0) {
        return (void *)ggml_backend_rpc_set_split_states;
    }
    if (std::strcmp(name, "ggml_backend_rpc_split_state_lookup") == 0) {
        // rpc-server resolves this via the registry (a direct call would leave
        // the tool with an undefined symbol when GGML_BACKEND_DL=ON)
        return (void *)ggml_backend_rpc_split_state_lookup;
    }
    if (std::strcmp(name, "ggml_backend_rpc_start_announcer") == 0) {
        return (void *)ggml_backend_rpc_start_announcer;
    }
    if (std::strcmp(name, "ggml_backend_rpc_source_hint") == 0) {
        return (void *)ggml_backend_rpc_source_hint;
    }
    if (std::strcmp(name, "ggml_backend_rpc_discover") == 0) {
        return (void *)ggml_backend_rpc_discover;
    }
    if (std::strcmp(name, "ggml_backend_rpc_any_endpoint_failed") == 0) {
        return (void *)ggml_backend_rpc_any_endpoint_failed;
    }
    if (std::strcmp(name, "ggml_backend_rpc_dev_worker_is_cpu") == 0) {
        return (void *)ggml_backend_rpc_dev_worker_is_cpu;
    }
    if (std::strcmp(name, "ggml_backend_rpc_reset_failed_endpoints") == 0) {
        return (void *)ggml_backend_rpc_reset_failed_endpoints;
    }
    if (std::strcmp(name, "ggml_backend_rpc_dev_reachable") == 0) {
        return (void *)ggml_backend_rpc_dev_reachable;
    }
    if (std::strcmp(name, "ggml_backend_rpc_dev_memory_ephemeral") == 0) {
        return (void *)ggml_backend_rpc_dev_memory_ephemeral;
    }
    if (std::strcmp(name, "ggml_backend_rpc_dev_failed") == 0) {
        return (void *)ggml_backend_rpc_dev_failed;
    }
    if (std::strcmp(name, "ggml_backend_rpc_endpoint_reprobe") == 0) {
        return (void *)ggml_backend_rpc_endpoint_reprobe;
    }
    if (std::strcmp(name, "ggml_backend_rpc_clear_failed_endpoint") == 0) {
        return (void *)ggml_backend_rpc_clear_failed_endpoint;
    }
    if (std::strcmp(name, "ggml_backend_rpc_buffer_endpoint") == 0) {
        return (void *)ggml_backend_rpc_buffer_endpoint;
    }
    if (std::strcmp(name, "ggml_backend_rpc_buffer_reprovision") == 0) {
        return (void *)ggml_backend_rpc_buffer_reprovision;
    }
    if (std::strcmp(name, "ggml_backend_cpy_tensor_batch_async") == 0) {
        return (void *)ggml_backend_rpc_cpy_tensor_batch_async;
    }
    if (std::strcmp(name, "ggml_backend_get_tensor_batch") == 0) {
        return (void *)ggml_backend_rpc_get_tensor_batch;
    }
    if (std::strcmp(name, "ggml_backend_boundary_fused_send") == 0) {
        return (void *)ggml_backend_rpc_boundary_fused_send;
    }
    if (std::strcmp(name, "ggml_backend_boundary_fused_recv") == 0) {
        return (void *)ggml_backend_rpc_boundary_fused_recv;
    }
    // fleet introspection + worker ops (proto 4.7, TASKS.md #35)
    if (std::strcmp(name, "ggml_backend_rpc_log_append") == 0) {
        return (void *)ggml_backend_rpc_log_append;
    }
    if (std::strcmp(name, "ggml_backend_rpc_fetch_log") == 0) {
        return (void *)ggml_backend_rpc_fetch_log;
    }
    if (std::strcmp(name, "ggml_backend_rpc_shutdown_worker") == 0) {
        return (void *)ggml_backend_rpc_shutdown_worker;
    }
    if (std::strcmp(name, "ggml_backend_rpc_benchmark_device") == 0) {
        return (void *)ggml_backend_rpc_benchmark_device;
    }
    if (std::strcmp(name, "ggml_backend_rpc_set_worker_score") == 0) {
        return (void *)ggml_backend_rpc_set_worker_score;
    }
    if (std::strcmp(name, "ggml_backend_rpc_get_worker_score") == 0) {
        return (void *)ggml_backend_rpc_get_worker_score;
    }
    if (std::strcmp(name, "ggml_backend_rpc_endpoint_stats") == 0) {
        return (void *)ggml_backend_rpc_endpoint_stats;
    }
    if (std::strcmp(name, "ggml_backend_rpc_foreach_endpoint_stat") == 0) {
        return (void *)ggml_backend_rpc_foreach_endpoint_stat;
    }
    if (std::strcmp(name, "ggml_backend_rpc_endpoint_reachable") == 0) {
        return (void *)ggml_backend_rpc_endpoint_reachable;
    }
    if (std::strcmp(name, "ggml_backend_rpc_set_shutdown_enabled") == 0) {
        return (void *)ggml_backend_rpc_set_shutdown_enabled;
    }
    return NULL;

    GGML_UNUSED(reg);
}

static const struct ggml_backend_reg_i ggml_backend_rpc_reg_i = {
    /* .get_name         = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count = */ ggml_backend_rpc_reg_get_device_count,
    /* .get_device       = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_reg(void) {
    static struct ggml_backend_reg ggml_backend_rpc_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_rpc_reg;
}

static uint32_t ggml_backend_rpc_get_device_count(const char * endpoint) {
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        GGML_LOG_ERROR("Failed to connect to %s\n", endpoint);
        return 0;
    }
    rpc_msg_device_count_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_DEVICE_COUNT, nullptr, 0, &response, sizeof(response));
    if (!status) {
        GGML_LOG_ERROR("Failed to query device count from %s\n", endpoint);
        return 0;
    }
    return response.device_count;
}

static const ggml_backend_reg_i ggml_backend_rpc_reg_interface = {
    /* .get_name          = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count  = */ ggml_backend_rpc_reg_get_device_count,
    /* .get_device        = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint) {
    static std::unordered_map<std::string, ggml_backend_reg_t> reg_map;
    static std::mutex mutex;
    static uint32_t dev_id = 0;
    std::lock_guard<std::mutex> lock(mutex);
    if (reg_map.find(endpoint) != reg_map.end()) {
        return reg_map[endpoint];
    }
    uint32_t dev_count = ggml_backend_rpc_get_device_count(endpoint);
    if (dev_count == 0) {
        return nullptr;
    }
    ggml_backend_rpc_reg_context * ctx = new ggml_backend_rpc_reg_context;
    ctx->name = "RPC[" + std::string(endpoint) + "]";
    for (uint32_t ind = 0; ind < dev_count; ind++) {
        std::string dev_name = "RPC" + std::to_string(dev_id);
        std::string dev_desc = std::string(endpoint);
        // fetch the real device description - a worker-side TP island advertises
        // itself as "Meta[N](...)" which the coordinator uses to upload split states;
        // older workers do not know this command and keep the endpoint as description
        {
            auto sock = get_socket(endpoint);
            rpc_msg_get_device_desc_req request = { ind };
            rpc_msg_get_device_desc_rsp response;
            if (sock != nullptr && send_rpc_cmd(sock, RPC_CMD_GET_DEVICE_DESC, &request, sizeof(request), &response, sizeof(response))) {
                response.desc[sizeof(response.desc) - 1] = '\0';
                dev_desc = std::string(response.desc) + " @ " + endpoint;
            }
        }
        bool worker_is_cpu = false;
        if (dev_desc.rfind("CPU|", 0) == 0) {
            worker_is_cpu = true;
            dev_desc = dev_desc.substr(4);
        }
        ggml_backend_rpc_device_context * dev_ctx = new ggml_backend_rpc_device_context {
            /* .endpoint    = */    endpoint,
            /* .device      = */    ind,
            /* .name        = */    dev_name,
            /* .description = */    dev_desc,
            /* .last_graph_uid = */ 0,
            /* .worker_is_cpu  = */ worker_is_cpu,
        };

        ggml_backend_dev_t dev = new ggml_backend_device {
            /* .iface   = */ ggml_backend_rpc_device_i,
            /* .reg     = */ ggml_backend_rpc_reg(),
            /* .context = */ dev_ctx,
        };
        ctx->devices.push_back(dev);
        dev_id++;
    }
    ggml_backend_reg_t reg = new ggml_backend_reg {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_interface,
        /* .context     = */ ctx
    };
    reg_map[endpoint] = reg;
    return reg;
}


GGML_BACKEND_DL_IMPL(ggml_backend_rpc_reg)
