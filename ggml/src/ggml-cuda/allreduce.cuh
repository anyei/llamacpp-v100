#pragma once

#include "common.cuh"
#include "ggml-backend-impl.h"

#include <cstddef>

// Opaque pipeline context -- owns all pinned buffers, streams, and events.
struct ggml_cuda_ar_pipeline;

// Allocate a pipeline for n_devices GPUs.
// devices[] holds the CUDA device IDs in rank order.
// Returns nullptr on allocation failure.
ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(
    const int * devices, size_t n_devices);

// Release all resources owned by the pipeline.
void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline * pipeline);

// Execute an in-place AllReduce (sum) across tensors[0..n_devices-1].
// tensors[i] must live on the device managed by backends[i] and be
// contiguous F32, F16, or BF16.
// Preconditions are checked by the CUDA comm dispatcher before calling this.
// Returns true once the reduction work has been enqueued successfully.
bool ggml_cuda_ar_allreduce(
    ggml_cuda_ar_pipeline * pipeline,
    ggml_backend_t        * backends,
    ggml_tensor           ** tensors);

// One-shot peer-to-peer AllReduce for exactly 2 devices with direct peer
// access (NVLink or PCIe P2P).  Targets the small per-layer reductions of
// tensor-parallel token generation, where per-call latency dominates.
struct ggml_cuda_ar_p2p;

// Returns nullptr unless n_devices == 2 and peer access can be enabled in
// both directions.  devices[] holds CUDA device IDs in backend rank order.
ggml_cuda_ar_p2p * ggml_cuda_ar_p2p_init(const int * devices, size_t n_devices);

void ggml_cuda_ar_p2p_free(ggml_cuda_ar_p2p * p2p);

// In-place sum across tensors[0..1] (contiguous F32 only).  Returns false if
// this call cannot be handled (type/size); the caller falls back.
bool ggml_cuda_ar_p2p_allreduce(
    ggml_cuda_ar_p2p * p2p,
    ggml_backend_t   * backends,
    ggml_tensor      ** tensors);

