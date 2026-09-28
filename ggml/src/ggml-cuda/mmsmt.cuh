#pragma once

#include "common.cuh"

// small-T tensor-core GEMM for quantized weights on Volta (sm_70): TASKS #153 T1

#define MMSMT_MAX_BATCH_SIZE 8 // widths 9-16 stay on MMQ until the two-tile register budget is solved

bool ggml_cuda_should_use_mmsmt(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

void ggml_cuda_mul_mat_smt(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
