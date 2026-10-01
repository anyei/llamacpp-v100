// TASKS #154 item 3 (code review G2a): the doorbell executor's per-layer math against ggml's own CPU chain
// (mul_mat_id up / gate, the clamped SwiGLU of the architecture, mul_mat_id down, gating-weighted sum over the lanes)
// on identical inputs, including skipped (cached) lanes and the DeepSeek-V4 clamp order. A negative control checks the
// test is sensitive: the V4 case computed with the other clamp order must fail.
#include "ggml.h"
#include "ggml-cpu.h"

#include "llama-moe-doorbell-compute.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

struct test_case {
    const char * name;
    ggml_type    gate_up_type;
    ggml_type    down_type;
    int          T;
    float        limit;
    bool         ds4;
    int          nth;
};

static const int n_embd = 512, n_ff = 256, n_expert = 16, n_used = 6;

// returns the relative max error of the executor vs the ggml reference (max |diff| / max |ref|)
static double run_case(const test_case & tc, bool wrong_clamp_order, std::mt19937 & rng) {
    ggml_init_params ip = { 256u * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    std::uniform_real_distribution<float> uw(-0.05f, 0.05f), ux(-1.0f, 1.0f), up01(0.05f, 1.0f), u01(0.0f, 1.0f);

    auto make_q = [&](ggml_type type, int64_t k, int64_t rows, int64_t n3) {
        ggml_tensor * t = ggml_new_tensor_3d(ctx, type, k, rows, n3);
        std::vector<float> src((size_t) k * rows * n3);
        for (auto & v : src) v = uw(rng);
        ggml_quantize_chunk(type, src.data(), t->data, 0, rows * n3, k, nullptr);
        return t;
    };
    ggml_tensor * up   = make_q(tc.gate_up_type, n_embd, n_ff, n_expert);
    ggml_tensor * gate = make_q(tc.gate_up_type, n_embd, n_ff, n_expert);
    ggml_tensor * down = make_q(tc.down_type,    n_ff, n_embd, n_expert);

    ggml_tensor * x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, tc.T);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, tc.T);
    ggml_tensor * w   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_used, tc.T);
    for (int64_t i = 0; i < ggml_nelements(x); ++i) ((float *) x->data)[i] = ux(rng);
    for (int t = 0; t < tc.T; ++t) {
        std::vector<int> perm(n_expert);
        for (int e = 0; e < n_expert; ++e) perm[e] = e;
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int k = 0; k < n_used; ++k) {
            // ~40 % of the lanes are cache hits: the skip sentinel, computed by nobody here
            ((int32_t *) ids->data)[t*n_used + k] = u01(rng) < 0.4f ? -1 : perm[k];
            ((float   *) w->data)[t*n_used + k]   = up01(rng);
        }
    }

    // reference: ggml's CPU chain, exactly as build_moe_ffn builds it for the miss lanes
    ggml_tensor * cur    = ggml_reshape_3d(ctx, x, n_embd, 1, tc.T);
    ggml_tensor * up_o   = ggml_mul_mat_id(ctx, up,   cur, ids);
    ggml_tensor * gate_o = ggml_mul_mat_id(ctx, gate, cur, ids);
    ggml_tensor * act    = nullptr;
    if (tc.limit > 1e-6f) {
        up_o = ggml_clamp(ctx, up_o, -tc.limit, tc.limit);
        if (tc.ds4) {
            gate_o = ggml_clamp(ctx, gate_o, -INFINITY, tc.limit);
            act    = ggml_swiglu_split(ctx, gate_o, up_o);
        } else {
            ggml_tensor * ga = ggml_clamp(ctx, ggml_silu(ctx, gate_o), -INFINITY, tc.limit);
            act = ggml_mul(ctx, ga, up_o);
        }
    } else {
        act = ggml_swiglu_split(ctx, gate_o, up_o);
    }
    ggml_tensor * down_o = ggml_mul_mat_id(ctx, down, act, ids);
    ggml_tensor * wtd    = ggml_mul(ctx, down_o, w);
    ggml_tensor * out    = ggml_view_2d(ctx, wtd, n_embd, tc.T, wtd->nb[2], 0);
    for (int k = 1; k < n_used; ++k) {
        out = ggml_add(ctx, out, ggml_view_2d(ctx, wtd, n_embd, tc.T, wtd->nb[2], k*wtd->nb[1]));
    }
    out = ggml_cont(ctx, out);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    // the executor's math, run as a team of tc.nth threads would (phase by phase, every thread's share)
    llama_moe_db_layer L;
    L.up = up; L.gate = gate; L.down = down; L.limit = tc.limit;
    L.gate_clamp_first = wrong_clamp_order ? !tc.ds4 : tc.ds4;
    llama_moe_db_work W;
    W.reserve(n_embd, n_ff, 8, n_used,
              std::max(llama_moe_db_row_size(tc.gate_up_type, n_embd), llama_moe_db_row_size(tc.gate_up_type, n_embd)),
              llama_moe_db_row_size(tc.down_type, n_ff));
    if (!W.collect(L, (const int32_t *) ids->data, (const float *) w->data, tc.T)) {
        printf("  collect rejected a valid request\n");
        ggml_free(ctx);
        return 1e9;
    }
    std::vector<float> got((size_t) tc.T * n_embd, NAN);
    for (int i = 0; i < tc.nth; ++i) W.quantize(L, (const float *) x->data, tc.T, i, tc.nth);
    for (int i = 0; i < tc.nth; ++i) W.gate_up(L, i, tc.nth);
    for (int i = 0; i < tc.nth; ++i) W.act(L, i, tc.nth);
    for (int i = 0; i < tc.nth; ++i) W.down(L, tc.T, got.data(), i, tc.nth);

    double max_ref = 0, max_diff = 0;
    for (int t = 0; t < tc.T; ++t) {
        for (int r = 0; r < n_embd; ++r) {
            const double ref = ((const float *) ((const char *) out->data + t*out->nb[1]))[r];
            const double g   = got[(size_t) t*n_embd + r];
            max_ref  = std::max(max_ref, std::fabs(ref));
            max_diff = std::isfinite(g) ? std::max(max_diff, std::fabs(g - ref)) : 1e30;
        }
    }
    ggml_free(ctx);
    return max_diff / std::max(max_ref, 1e-30);
}

int main() {
    ggml_cpu_init();
    std::mt19937 rng(154);
    const test_case cases[] = {
        { "q3_K gate/up, q2_0 down (Flash-Next), T=4",  GGML_TYPE_Q3_K, GGML_TYPE_Q2_0, 4, 0.0f, false, 1 },
        { "q3_K / q2_0, T=1, 3 threads",                 GGML_TYPE_Q3_K, GGML_TYPE_Q2_0, 1, 0.0f, false, 3 },
        { "q2_K / q2_0, T=4, 5 threads",                 GGML_TYPE_Q2_K, GGML_TYPE_Q2_0, 4, 0.0f, false, 5 },
        { "q2_0 / q2_0, T=8, 4 threads",                 GGML_TYPE_Q2_0, GGML_TYPE_Q2_0, 8, 0.0f, false, 4 },
        { "q8_0 / q4_0, T=3, 2 threads",                 GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 3, 0.0f, false, 2 },
        { "clamped SwiGLU, T=4, 3 threads",              GGML_TYPE_Q3_K, GGML_TYPE_Q2_0, 4, 0.4f, false, 3 },
        { "DeepSeek-V4 clamp order, T=4, 3 threads",     GGML_TYPE_Q3_K, GGML_TYPE_Q2_0, 4, 0.4f, true,  3 },
        { "DeepSeek-V4 clamp order, q8_0, T=2",          GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 2, 0.4f, true,  1 },
    };
    // same quantized inputs and the same dot products; SiLU (libm expf vs ggml's vector exp) can move an activation's
    // q8_0 rounding by one step, so the bound is a tolerance, not bit equality
    const double tol = 2e-3;
    int n_fail = 0;
    for (const auto & tc : cases) {
        const double e = run_case(tc, false, rng);
        const bool ok = e <= tol;
        n_fail += !ok;
        printf("%-46s rel max err %.2e  %s\n", tc.name, e, ok ? "OK" : "FAIL");
    }
    // negative control: the V4 case with the other clamp order must be caught
    {
        const double e = run_case(cases[6], true, rng);
        const bool caught = e > 10*tol;
        n_fail += !caught;
        printf("%-46s rel max err %.2e  %s\n", "control: V4 computed with the wrong order", e, caught ? "CAUGHT (OK)" : "NOT CAUGHT (FAIL)");
    }
    printf("%s\n", n_fail == 0 ? "ALL OK" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}
