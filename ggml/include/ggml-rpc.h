#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// minor 11: upstream 6d5a910 merge added 4 ggml ops (lightning indexer,
// fused hyper-connections) - the op enum shifted on the wire, so pre-merge
// workers must be rejected. patch tracks upstream's op-enum fingerprint and
// is enforced at HELLO (a mismatch decodes graphs to the wrong ops).
#define RPC_PROTO_MAJOR_VERSION    4
#define RPC_PROTO_MINOR_VERSION    11
#define RPC_PROTO_PATCH_VERSION    3

#ifdef  __cplusplus
static_assert(GGML_OP_COUNT == 101, "GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION");
#endif

#define GGML_RPC_MAX_SERVERS       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device);
GGML_BACKEND_API bool ggml_backend_is_rpc(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device);

GGML_BACKEND_API void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total);

// source provenance for weight uploads (proto 4.8, TASKS.md #44): the caller announces
// that the data pointer of subsequent set_tensor(_2d) calls on this thread aliases the
// named GGUF tensor's bytes at base_offset. The RPC client forwards (name, offset, row
// geometry) in SET_TENSOR_HASH2 offers so a --model-dir worker can serve an arbitrary
// SLICE of a local GGUF by pread - split-boundary moves no longer cold-miss the whole
// cache. NULL name clears the hint. Every serve is hash-verified, so a wrong or stale
// hint degrades to streaming, never to corruption.
GGML_BACKEND_API void ggml_backend_rpc_source_hint(const char * name, uint64_t base_offset);

// model_dir (optional, TASKS.md #26): directory of local GGUF files indexed by tensor-content
// hash at startup; SET_TENSOR_HASH cache misses are then served from local disk instead of
// streaming from the coordinator. A stale local file hash-misses and streams as before.
GGML_BACKEND_API void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir, const char * model_dir,
                                                    size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_reg(void);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint);

// tensor-parallel islands (docs/distributed-inference-plan.md phase 1):
// the coordinator computes per-tensor split states and uploads them to a worker running
// with --tensor-parallel; the worker-side meta device resolves them by tensor name.
// blob layout, repeated per tensor: [uint32 name_len][name bytes][ggml_backend_meta_split_state]
GGML_BACKEND_API bool ggml_backend_rpc_set_split_states(const char * endpoint, const void * data, size_t size);

// worker-side lookup of an uploaded split state; returns false if the name is unknown
GGML_BACKEND_API bool ggml_backend_rpc_split_state_lookup(const char * name, struct ggml_backend_meta_split_state * out);

// returns the endpoint of an RPC device, or NULL if the device is not an RPC device.
// combined with a device description of the form "Meta[N](...)" this identifies a
// remote tensor-parallel island of N GPUs that needs split states uploaded.
GGML_BACKEND_API const char * ggml_backend_rpc_dev_endpoint(ggml_backend_dev_t dev);

// LAN presence beacon + discovery over UDP multicast (opt-in, trusted networks only —
// the beacon egress is pinned to the interface the RPC endpoint is bound to).
// group syntax "ADDR:PORT"; NULL selects the built-in default group.
// worker side: announce this rpc-server every ~2 s so coordinators can discover it.
// cache_dir (optional): the tensor-cache directory; its current size is published
// in the beacon (cache_mib=) so fleet UIs can show per-worker disk pressure.
GGML_BACKEND_API bool ggml_backend_rpc_start_announcer(const char * endpoint, const char * group,
                                                       size_t n_devices, ggml_backend_dev_t * devices,
                                                       const char * cache_dir);
// coordinator side: listen for worker beacons for timeout_ms; cb fires once per unique
// endpoint ("ip:port", with the beacon's payload line). returns the number of workers found.
GGML_BACKEND_API int ggml_backend_rpc_discover(const char * group, int timeout_ms,
                                               void (*cb)(const char * endpoint, const char * payload, void * user_data),
                                               void * user_data);

// true when this RPC device is backed by the worker's CPU (RAM) rather than a GPU —
// side channel for placement policies (the device still reports type GPU to llama).
GGML_BACKEND_API bool ggml_backend_rpc_dev_worker_is_cpu(ggml_backend_dev_t dev);

// true once any RPC worker connection has failed (worker crash / network loss). The
// model's layer assignment is unrecoverable in-process — the serving application should
// fail in-flight requests and either exit for its restart policy or unload the model and
// reload across the live workers (see the two calls below).
GGML_BACKEND_API bool ggml_backend_rpc_any_endpoint_failed(void);

// forget past endpoint failures so a reload (after the model is destroyed) reconnects
GGML_BACKEND_API void ggml_backend_rpc_reset_failed_endpoints(void);

// probe whether an RPC device's endpoint accepts connections right now
GGML_BACKEND_API bool ggml_backend_rpc_dev_reachable(ggml_backend_dev_t dev);

// surgical re-provision of a restarted worker (TASKS.md #29c refinement)
GGML_BACKEND_API bool         ggml_backend_rpc_dev_failed(ggml_backend_dev_t dev);
GGML_BACKEND_API bool         ggml_backend_rpc_endpoint_reprobe(const char * endpoint);
GGML_BACKEND_API void         ggml_backend_rpc_clear_failed_endpoint(const char * endpoint);
GGML_BACKEND_API const char * ggml_backend_rpc_buffer_endpoint(ggml_backend_buffer_t buffer);
GGML_BACKEND_API bool         ggml_backend_rpc_buffer_reprovision(ggml_backend_buffer_t buffer, void ** old_base, void ** new_base);

// fleet introspection + worker ops (proto 4.7, TASKS.md #35). The three commands
// behind these are handled OUTSIDE the worker's exec lock so they respond even
// while a graph is computing (a hung worker can still be shut down / inspected).

// worker side: append a line to the in-memory log ring served by RPC_CMD_GET_LOG
// (rpc-server wires its ggml log callback here; ~64 KiB retained)
GGML_BACKEND_API void ggml_backend_rpc_log_append(const char * text);

// coordinator side: fetch the tail of a worker's log ring over a dedicated
// connection (never queues behind compute traffic on the cached socket)
GGML_BACKEND_API bool ggml_backend_rpc_fetch_log(const char * endpoint, char * buf, size_t buf_size, size_t * out_len);

// coordinator side: ask a worker process to exit(0) so its supervisor (docker
// restart policy / systemd) restarts it; --rpc-reload then re-provisions it.
// The worker refuses unless it opted in via ggml_backend_rpc_set_shutdown_enabled.
GGML_BACKEND_API bool ggml_backend_rpc_shutdown_worker(const char * endpoint);
// worker side: allow remote shutdown (OFF by default - RPC is unauthenticated)
GGML_BACKEND_API void ggml_backend_rpc_set_shutdown_enabled(bool enabled);

// bounded TCP reachability probe (timeout_ms); safe to call from an HTTP thread,
// unlike ggml_backend_rpc_dev_reachable which can block on the cached socket
GGML_BACKEND_API bool ggml_backend_rpc_endpoint_reachable(const char * endpoint, int timeout_ms);

// worker speed score (TASKS.md #31/#35f): a short matvec benchmark measures the
// effective memory bandwidth of a device — decode t/s is bandwidth-bound, so
// this is the right weight for splitting work across unequal workers
GGML_BACKEND_API bool ggml_backend_rpc_benchmark_device(ggml_backend_dev_t dev, float * bw_gbps, float * mm_gflops);
// worker side: publish this worker's score (beacon field + RPC_CMD_GET_SCORE)
GGML_BACKEND_API void ggml_backend_rpc_set_worker_score(float bw_gbps, float mm_gflops);
// coordinator side: ask a worker to RE-benchmark now (proto 4.9). The worker
// refuses (returns false) while a compute is in flight - a bench racing a decode
// under-reads both. On success the worker's beacon carries the fresh score.
GGML_BACKEND_API bool ggml_backend_rpc_rescore_worker(const char * endpoint, float * bw_gbps, float * mm_gflops);
// coordinator side: query a worker's score; false if unscored or proto < 4.7
GGML_BACKEND_API bool ggml_backend_rpc_get_worker_score(const char * endpoint, float * bw_gbps, float * mm_gflops);

// client-side per-endpoint transfer counters (accumulated since process start)
struct ggml_backend_rpc_ep_stats {
    uint64_t bytes_sent;      // command payloads, headers included
    uint64_t bytes_recv;      // blocking-command responses
    uint64_t n_calls;         // commands sent
    uint64_t ewma_latency_us; // EWMA (alpha 1/8) of blocking round-trip latency
    // weight-upload provenance (loading view): bytes the worker already had
    // (tensor cache / --model-dir, served via SET_TENSOR_HASH) vs bytes that
    // had to stream over the wire
    uint64_t weights_cached_bytes;
    uint64_t weights_streamed_bytes;
};
GGML_BACKEND_API bool ggml_backend_rpc_endpoint_stats(const char * endpoint, struct ggml_backend_rpc_ep_stats * out);

// enumerate all endpoints the client has connected to, with live counters. Safe to
// call while a model is still loading (used for the Fleet UI's per-worker load view).
GGML_BACKEND_API void ggml_backend_rpc_foreach_endpoint_stat(
    void (*cb)(const char * endpoint, const struct ggml_backend_rpc_ep_stats * st, void * user_data),
    void * user_data);

#ifdef  __cplusplus
}
#endif
