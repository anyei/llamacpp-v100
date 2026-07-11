#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define RPC_PROTO_MAJOR_VERSION    4
#define RPC_PROTO_MINOR_VERSION    2
#define RPC_PROTO_PATCH_VERSION    1

#ifdef  __cplusplus
static_assert(GGML_OP_COUNT == 97, "GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION");
#endif

#define GGML_RPC_MAX_SERVERS       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device);
GGML_BACKEND_API bool ggml_backend_is_rpc(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device);

GGML_BACKEND_API void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total);

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
GGML_BACKEND_API bool ggml_backend_rpc_start_announcer(const char * endpoint, const char * group,
                                                       size_t n_devices, ggml_backend_dev_t * devices);
// coordinator side: listen for worker beacons for timeout_ms; cb fires once per unique
// endpoint ("ip:port", with the beacon's payload line). returns the number of workers found.
GGML_BACKEND_API int ggml_backend_rpc_discover(const char * group, int timeout_ms,
                                               void (*cb)(const char * endpoint, const char * payload, void * user_data),
                                               void * user_data);

// true once any RPC worker connection has failed (worker crash / network loss). The
// model's layer assignment is unrecoverable in-process — the serving application should
// fail in-flight requests and exit so its restart policy re-splits across live workers.
GGML_BACKEND_API bool ggml_backend_rpc_any_endpoint_failed(void);

#ifdef  __cplusplus
}
#endif
