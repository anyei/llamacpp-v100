// TASKS #154 item 3 (fork; docs/strata-port-plan.md section 7): the MoE doorbell handoff.
//
// The mailbox slot is pinned host memory (device-addressable through UVA). ring copies the FFN input rows, the miss
// ids and the gating weights into it and publishes the step; the host executor computes the missed experts while the
// GPU runs the cache hits; join spins on the slot's DONE word, then adds the host's weighted partial.
#include "moe-doorbell.cuh"

// a stuck host executor turns into an error after this long instead of a hang (the globaltimer is in ns)
#define MOE_JOIN_TIMEOUT_NS 10000000000ull

static __device__ __forceinline__ void moe_backoff() {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    __nanosleep(100);
#endif
}

static __device__ __forceinline__ uint64_t moe_now_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

static __global__ void k_moe_ring(
        const float * __restrict__ x, const int64_t x_s1, const int32_t * __restrict__ ids, const float * __restrict__ w,
        const int n_embd, const int n_used, const int n_tokens,
        char * slot, const int off_x, const int off_ids, const int off_w, const int32_t * step_in, int32_t * dst) {
    float   * hx   = (float   *) (slot + off_x);
    int32_t * hids = (int32_t *) (slot + off_ids);
    float   * hw   = (float   *) (slot + off_w);

    for (int t = 0; t < n_tokens; ++t) {
        for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
            hx[(int64_t) t*n_embd + i] = x[t*x_s1 + i];
        }
    }
    for (int i = threadIdx.x; i < n_used*n_tokens; i += blockDim.x) {
        hids[i] = ids[i];
        hw[i]   = w[i];
    }
    __threadfence_system();
    __syncthreads();

    if (threadIdx.x == 0) {
        const unsigned int step = (unsigned int) step_in[0]; // this graph's step (a graph input in device memory)
        *((volatile int *) (slot + GGML_MOE_SLOT_NTOK)) = n_tokens;
        __threadfence_system();
        *((volatile unsigned int *) (slot + GGML_MOE_SLOT_RING)) = step;
        dst[0] = (int32_t) step;
    }
}

static __global__ void k_moe_join(
        const float * __restrict__ a, const int64_t a_s1, float * __restrict__ dst, const int64_t d_s1,
        const int n_embd, const int n_tokens, const char * slot, const int off_partial, const int32_t * ring) {
    __shared__ int ok;
    if (threadIdx.x == 0) {
        const unsigned int step = (unsigned int) ring[0];
        const volatile unsigned int * done = (const volatile unsigned int *) (slot + GGML_MOE_SLOT_DONE);
        const uint64_t t0 = moe_now_ns();
        while (*done != step && moe_now_ns() - t0 < MOE_JOIN_TIMEOUT_NS) {
            moe_backoff();
        }
        ok = *done == step;
        __threadfence_system();
    }
    __syncthreads();
    if (!ok) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("moe doorbell: timeout waiting for the host executor (step %d) - aborting\n", ring[0]);
        }
        __trap();
    }

    const float * p = (const float *) (slot + off_partial);
    const int64_t n = (int64_t) n_embd*n_tokens;
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x*blockDim.x) {
        const int64_t t = i / n_embd;
        const int64_t c = i - t*n_embd;
        dst[t*d_s1 + c] = a[t*a_s1 + c] + __ldcv(p + i);
    }
}

void ggml_cuda_op_moe_ring(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x    = dst->src[0];
    const ggml_tensor * ids  = dst->src[1];
    const ggml_tensor * w    = dst->src[2];
    const ggml_tensor * step = dst->src[3];

    int64_t slot = 0;
    memcpy(&slot, dst->op_params + 0, sizeof(slot));

    const int n_embd   = (int) x->ne[0];
    const int n_tokens = (int) x->ne[1];
    const int n_used   = (int) ids->ne[0];

    k_moe_ring<<<1, 256, 0, ctx.stream()>>>(
        (const float *) x->data, x->nb[1]/sizeof(float), (const int32_t *) ids->data, (const float *) w->data,
        n_embd, n_used, n_tokens, (char *) slot, dst->op_params[2], dst->op_params[3], dst->op_params[4],
        (const int32_t *) step->data, (int32_t *) dst->data);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_moe_join(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a    = dst->src[0];
    const ggml_tensor * ring = dst->src[1];

    int64_t slot = 0;
    memcpy(&slot, dst->op_params + 0, sizeof(slot));

    const int n_embd   = (int) a->ne[0];
    const int n_tokens = (int) ggml_nrows(a);
    const int n_blocks = (int) std::min<int64_t>(16, ((int64_t) n_embd*n_tokens + 255)/256);

    k_moe_join<<<n_blocks, 256, 0, ctx.stream()>>>(
        (const float *) a->data, a->nb[1]/sizeof(float), (float *) dst->data, dst->nb[1]/sizeof(float),
        n_embd, n_tokens, (const char *) slot, dst->op_params[2], (const int32_t *) ring->data);
    CUDA_CHECK(cudaGetLastError());
}
