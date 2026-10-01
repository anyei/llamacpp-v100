// TASKS #154 item 3: the doorbell executor's per-layer math - see llama-moe-doorbell-compute.h
#include "llama-moe-doorbell-compute.h"

#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#define LLAMA_EXPERT_SLOT_SKIP_DB (-1) // the cache's skip sentinel (llama-expert-placement.h)

size_t llama_moe_db_row_size(ggml_type weight_type, int64_t n) {
    const auto * tr = ggml_get_type_traits_cpu(weight_type);
    if (tr == nullptr || tr->vec_dot == nullptr || ggml_get_type_traits_cpu(tr->vec_dot_type)->from_float == nullptr) {
        return 0;
    }
    return ggml_row_size(tr->vec_dot_type, n);
}

void llama_moe_db_work::reserve(int32_t n_embd_, int32_t n_ff_, int32_t max_tokens_, int32_t n_used_, size_t max_xq_rs, size_t max_aq_rs) {
    GGML_ASSERT(max_tokens_ >= 1 && max_tokens_ <= LLAMA_MOE_DB_MAX_TOKENS);
    n_embd     = n_embd_;
    n_ff       = n_ff_;
    max_tokens = max_tokens_;
    n_used     = n_used_;
    const size_t max_miss = (size_t) max_tokens * n_used;
    misses.reserve(max_miss);
    xq_rs = max_xq_rs;
    aq_rs = max_aq_rs;
    xq_gate.resize(xq_rs * max_tokens);
    xq_up.resize(xq_rs * max_tokens);
    aq.resize(aq_rs * max_miss);
    g.resize(max_miss * n_ff);
    u.resize(max_miss * n_ff);
}

bool llama_moe_db_work::collect(const llama_moe_db_layer & L, const int32_t * ids, const float * w, int32_t T) {
    misses.clear();
    if (T < 1 || T > max_tokens) {
        return false;
    }
    const int64_t n_expert = L.up->ne[2];
    for (int32_t t = 0; t < T; ++t) {
        for (int32_t k = 0; k < n_used; ++k) {
            const int32_t e = ids[t*n_used + k];
            if (e == LLAMA_EXPERT_SLOT_SKIP_DB) {
                continue;
            }
            if (e < 0 || e >= n_expert) {
                return false;
            }
            misses.push_back({ t, e, w[t*n_used + k] });
        }
    }
    return true;
}

void llama_moe_db_work::quantize(const llama_moe_db_layer & L, const float * x, int32_t T, int ith, int nth) {
    const auto * tr_gate = ggml_get_type_traits_cpu(L.gate->type);
    const auto * tr_up   = ggml_get_type_traits_cpu(L.up->type);
    const bool same = tr_gate->vec_dot_type == tr_up->vec_dot_type;
    for (int32_t t = ith; t < T; t += nth) {
        ggml_get_type_traits_cpu(tr_gate->vec_dot_type)->from_float(x + (size_t) t*n_embd, xq_gate.data() + t*xq_rs, n_embd);
        if (!same) {
            ggml_get_type_traits_cpu(tr_up->vec_dot_type)->from_float(x + (size_t) t*n_embd, xq_up.data() + t*xq_rs, n_embd);
        }
    }
}

void llama_moe_db_work::gate_up(const llama_moe_db_layer & L, int ith, int nth) {
    const auto * tr_gate = ggml_get_type_traits_cpu(L.gate->type);
    const auto * tr_up   = ggml_get_type_traits_cpu(L.up->type);
    const bool same = tr_gate->vec_dot_type == tr_up->vec_dot_type;
    // flattened (miss, row) space, contiguous chunks
    const int64_t n_items = (int64_t) misses.size() * n_ff;
    const int64_t i0 = n_items * ith / nth, i1 = n_items * (ith + 1) / nth;
    for (int64_t i = i0; i < i1; ++i) {
        const int32_t m = (int32_t) (i / n_ff), r = (int32_t) (i % n_ff);
        const llama_moe_db_miss & ms = misses[m];
        const char * wg = (const char *) L.gate->data + ms.e*L.gate->nb[2] + r*L.gate->nb[1];
        const char * wu = (const char *) L.up->data   + ms.e*L.up->nb[2]   + r*L.up->nb[1];
        const void * xg = xq_gate.data() + ms.t*xq_rs;
        const void * xu = same ? xg : (const void *) (xq_up.data() + ms.t*xq_rs);
        tr_gate->vec_dot(n_embd, &g[(size_t) m*n_ff + r], 0, wg, 0, xg, 0, 1);
        tr_up  ->vec_dot(n_embd, &u[(size_t) m*n_ff + r], 0, wu, 0, xu, 0, 1);
    }
}

void llama_moe_db_work::act(const llama_moe_db_layer & L, int ith, int nth) {
    const auto * tr_down = ggml_get_type_traits_cpu(L.down->type);
    const bool   clamp   = L.limit > 1e-6f;
    for (int32_t m = ith; m < (int32_t) misses.size(); m += nth) {
        float       * gm = &g[(size_t) m*n_ff];
        const float * um = &u[(size_t) m*n_ff];
        for (int32_t r = 0; r < n_ff; ++r) {
            float gate = gm[r];
            float up   = um[r];
            float a;
            if (!clamp) {
                a = gate / (1.0f + expf(-gate)) * up;
            } else {
                up = std::min(std::max(up, -L.limit), L.limit);
                if (L.gate_clamp_first) {
                    gate = std::min(gate, L.limit);                           // DeepSeek-V4: SwiGLU(clamp(gate), clamp(up))
                    a    = gate / (1.0f + expf(-gate)) * up;
                } else {
                    a = std::min(gate / (1.0f + expf(-gate)), L.limit) * up; // clamp(SiLU(gate)) * clamp(up)
                }
            }
            gm[r] = a;
        }
        ggml_get_type_traits_cpu(tr_down->vec_dot_type)->from_float(gm, aq.data() + m*aq_rs, n_ff);
    }
}

void llama_moe_db_work::down(const llama_moe_db_layer & L, int32_t T, float * out, int ith, int nth) {
    const auto * tr_down = ggml_get_type_traits_cpu(L.down->type);
    const int32_t r0 = (int32_t) ((int64_t) n_embd * ith / nth), r1 = (int32_t) ((int64_t) n_embd * (ith + 1) / nth);
    float acc[LLAMA_MOE_DB_MAX_TOKENS];
    for (int32_t r = r0; r < r1; ++r) {
        std::fill(acc, acc + T, 0.0f);
        for (size_t m = 0; m < misses.size(); ++m) {
            const llama_moe_db_miss & ms = misses[m];
            const char * wd = (const char *) L.down->data + ms.e*L.down->nb[2] + r*L.down->nb[1];
            float v;
            tr_down->vec_dot(n_ff, &v, 0, wd, 0, aq.data() + m*aq_rs, 0, 1);
            acc[ms.t] += ms.w * v;
        }
        for (int32_t t = 0; t < T; ++t) {
            out[(size_t) t*n_embd + r] = acc[t];
        }
    }
}
