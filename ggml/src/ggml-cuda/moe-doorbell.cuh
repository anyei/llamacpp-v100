#pragma once

#include "common.cuh"

// TASKS #154 item 3 (fork; docs/strata-port-plan.md section 7): MoE doorbell handoff ops
void ggml_cuda_op_moe_ring(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_moe_join(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
