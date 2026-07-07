#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define RPC_PROTO_MAJOR_VERSION    4
#define RPC_PROTO_MINOR_VERSION    1
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

GGML_BACKEND_API void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
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

#ifdef  __cplusplus
}
#endif
