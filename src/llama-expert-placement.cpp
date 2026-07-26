// TASKS #75: hot-expert placement artifact loading + validation.
// See llama-expert-placement.h and docs/expert-placement-plan.md.

#include "llama-expert-placement.h"

#include "llama-impl.h"

#include "ggml-backend.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <cstdlib>

using json = nlohmann::ordered_json;

std::unique_ptr<llama_expert_placement> llama_expert_placement_load(
        const std::string & path, uint32_t n_layer, uint32_t n_expert, size_t n_members) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error(format("expert placement: cannot open '%s'", path.c_str()));
    }

    json j;
    try {
        j = json::parse(f);
    } catch (const std::exception & e) {
        throw std::runtime_error(format("expert placement: '%s' is not valid JSON: %s", path.c_str(), e.what()));
    }

    auto placement = std::make_unique<llama_expert_placement>();

    try {
        placement->model_name = j.value("model", "");
        placement->n_layer    = j.at("n_layer").get<uint32_t>();
        placement->n_expert   = j.at("n_expert").get<uint32_t>();

        if (placement->n_layer != n_layer) {
            throw std::runtime_error(format("n_layer mismatch: artifact %u, model %u", placement->n_layer, n_layer));
        }
        if (placement->n_expert != n_expert) {
            throw std::runtime_error(format("n_expert mismatch: artifact %u, model %u", placement->n_expert, n_expert));
        }

        if (j.contains("member_shares")) {
            placement->member_shares = j["member_shares"].get<std::vector<double> >();
        }

        const json & j_perm = j.at("perm");
        const json & j_cnt  = j.at("counts_per_layer");
        if (j_perm.size() != n_layer || j_cnt.size() != n_layer) {
            throw std::runtime_error(format("perm/counts_per_layer must have n_layer=%u entries (got %zu/%zu)",
                                            n_layer, j_perm.size(), j_cnt.size()));
        }

        placement->perm.resize(n_layer);
        placement->counts.resize(n_layer);

        uint32_t n_placed = 0;
        for (uint32_t il = 0; il < n_layer; il++) {
            const json & jp = j_perm[il];
            const json & jc = j_cnt[il];
            if (jp.is_null() != jc.is_null()) {
                throw std::runtime_error(format("layer %u: perm and counts_per_layer must be null together", il));
            }
            if (jp.is_null()) {
                continue; // unplaced layer (dense / never routed)
            }
            n_placed++;

            std::vector<int32_t> perm = jp.get<std::vector<int32_t>>();
            if (perm.size() != n_expert) {
                throw std::runtime_error(format("layer %u: perm has %zu entries, want n_expert=%u", il, perm.size(), n_expert));
            }
            // bijection check
            std::vector<uint8_t> seen(n_expert, 0);
            for (int32_t e : perm) {
                if (e < 0 || (uint32_t) e >= n_expert || seen[e]) {
                    throw std::runtime_error(format("layer %u: perm is not a bijection over [0,%u) (expert %d)", il, n_expert, e));
                }
                seen[e] = 1;
            }

            std::vector<int32_t> cnt = jc.get<std::vector<int32_t>>();
            if (cnt.size() != n_members) {
                throw std::runtime_error(format("layer %u: counts_per_layer has %zu entries, want n_members=%zu "
                                                "(artifact generated for a different member count?)", il, cnt.size(), n_members));
            }
            int64_t sum = 0;
            for (size_t j = 0; j < cnt.size(); j++) {
                if (cnt[j] < 0) {
                    throw std::runtime_error(format("layer %u: negative member count %d", il, cnt[j]));
                }
                if (cnt[j] == 0) {
                    throw std::runtime_error(format("layer %u: member %zu owns 0 experts - "
                                                    "every member needs >= 1 expert per placed layer (v1 guard)", il, j));
                }
                sum += cnt[j];
            }
            if (sum != (int64_t) n_expert) {
                throw std::runtime_error(format("layer %u: member counts sum to %lld, want n_expert=%u",
                                                il, (long long) sum, n_expert));
            }

            placement->perm  [il].swap(perm);
            placement->counts[il].swap(cnt);
        }

        if (n_placed == 0) {
            throw std::runtime_error("no layer carries a placement (all perm entries null)");
        }

        LLAMA_LOG_INFO("%s: loaded expert placement '%s' (model '%s'): %u/%u layers placed, "
                       "%u experts across %zu members\n",
                       __func__, path.c_str(), placement->model_name.c_str(), n_placed, n_layer,
                       n_expert, n_members);
        if (j.contains("source_profile")) {
            const json & sp = j["source_profile"];
            LLAMA_LOG_INFO("%s: placement source profile: %s (tokens=%s)\n", __func__,
                           sp.value("path", "?").c_str(),
                           sp.contains("tokens") ? sp["tokens"].dump().c_str() : "?");
        }
    } catch (const json::exception & e) {
        throw std::runtime_error(format("expert placement: '%s' malformed: %s", path.c_str(), e.what()));
    }

    return placement;
}

int llama_expert_placement_layer_for(
        const llama_expert_placement * pl, const char * tensor_name) {
    if (pl == nullptr || tensor_name == nullptr) {
        return -1;
    }
    static const std::regex pattern_exps_weight("blk\\.(\\d+)\\.ffn_(gate|up|gate_up|down)_exps\\.weight");
    std::cmatch m;
    if (!std::regex_match(tensor_name, m, pattern_exps_weight)) {
        return -1;
    }
    const uint32_t il = (uint32_t) std::stoul(m[1]);
    if (!pl->layer_placed(il)) {
        return -1;
    }
    return (int) il;
}

const std::vector<int32_t> * llama_expert_placement_perm_for(
        const llama_expert_placement * pl, const char * tensor_name) {
    const int il = llama_expert_placement_layer_for(pl, tensor_name);
    return il >= 0 ? &pl->perm[il] : nullptr;
}

// gate 4 static audit (docs/expert-placement-plan.md section 4 item 4): verify
// member j's built ownership tables are a true partition - every expert owned
// by exactly one member, owned slots inside the artifact counts, non-owned
// experts remapped to the dummy slot 0. Loud load-time error naming the field,
// same style as the section 3 consistency check.
static void llama_expert_placement_audit_tables(
        uint32_t il, size_t j, const std::vector<int32_t> & perm, const std::vector<int32_t> & cnt,
        const std::vector<int32_t> & remap_j, const std::vector<float> & mask_j) {
    const uint32_t n_expert = (uint32_t) perm.size();
    const int32_t  cnt_j    = cnt[j];
    // original expert id -> owning member, from the perm+counts the tables were
    // built from (perm is a validated bijection, counts sum to n_expert)
    std::vector<int32_t> member_of(n_expert, -1);
    {
        int32_t pos = 0;
        for (size_t m = 0; m < cnt.size(); m++) {
            for (int32_t k = 0; k < cnt[m]; k++) {
                member_of[perm[pos + k]] = (int32_t) m;
            }
            pos += cnt[m];
        }
    }
    int64_t n_owned = 0;
    for (uint32_t e = 0; e < n_expert; e++) {
        if (member_of[e] < 0) {
            throw std::runtime_error(format("expert placement: layer %u: expert %d has no owning member "
                                            "(ownership is not a partition)", il, e));
        }
        const bool owned = member_of[e] == (int32_t) j;
        n_owned += owned;
        if (!owned && remap_j[e] != LLAMA_EXPERT_SLOT_SKIP) {
            throw std::runtime_error(format("expert placement: layer %u member %zu: remap maps non-owned expert %d "
                                            "to local slot %d, want the skip sentinel %d", il, j, e, remap_j[e],
                                            LLAMA_EXPERT_SLOT_SKIP));
        }
        if (owned && (remap_j[e] < 0 || remap_j[e] >= cnt_j)) {
            throw std::runtime_error(format("expert placement: layer %u member %zu: owned expert %d local slot %d "
                                            "outside [0,%d)", il, j, e, remap_j[e], cnt_j));
        }
        if (mask_j[e] != (owned ? 1.0f : 0.0f)) {
            throw std::runtime_error(format("expert placement: layer %u member %zu: mask for expert %d is %f, want %s",
                                            il, j, e, (double) mask_j[e], owned ? "1.0" : "0.0"));
        }
    }
    if (n_owned != cnt_j) {
        throw std::runtime_error(format("expert placement: layer %u: member %zu owns %lld experts, "
                                        "counts_per_layer says %d (ownership is not a partition)",
                                        il, j, (long long) n_owned, cnt_j));
    }
}

std::unique_ptr<llama_expert_placement_tables> llama_expert_placement_create_tables(
        const llama_expert_placement & pl, ggml_backend_buffer_type_t meta_buft, size_t n_members) {
    const uint32_t n_layer  = pl.n_layer;
    const uint32_t n_expert = pl.n_expert;

    uint32_t n_placed = 0;
    for (uint32_t il = 0; il < n_layer; il++) {
        n_placed += pl.layer_placed(il);
    }
    GGML_ASSERT(n_placed > 0);

    auto tables = std::make_unique<llama_expert_placement_tables>();
    ggml_init_params ip = {
        /*.mem_size   =*/ 2*n_placed*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    tables->ctx.reset(ggml_init(ip));
    GGML_ASSERT(tables->ctx);

    tables->remap.assign(n_layer, nullptr);
    tables->mask.assign(n_layer, nullptr);

    for (uint32_t il = 0; il < n_layer; il++) {
        if (!pl.layer_placed(il)) {
            continue;
        }
        ggml_tensor * remap = ggml_new_tensor_1d(tables->ctx.get(), GGML_TYPE_I32, n_expert);
        ggml_tensor * mask  = ggml_new_tensor_1d(tables->ctx.get(), GGML_TYPE_F32, n_expert);
        ggml_format_name(remap, "blk.%u.exp_remap", il);
        ggml_format_name(mask,  "blk.%u.exp_mask",  il);
        tables->remap[il] = remap;
        tables->mask [il] = mask;
    }

    tables->buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(tables->ctx.get(), meta_buft));
    if (!tables->buf) {
        throw std::runtime_error("expert placement: failed to allocate ownership tables on the meta buffer");
    }

    // per-member contents: member j's remap maps its owned experts to local slots
    // [0, cnt_j) (positions relative to its permuted range start); non-owned ids
    // map to local slot 0 (a valid slot - the mask zeroes their contribution).
    static const int meta_debug = []() {
        const char * d = getenv("GGML_META_DEBUG");
        return d != nullptr ? atoi(d) : 0;
    }();
    static bool selftest_done = false;
    std::vector<int32_t> remap_j(n_expert);
    std::vector<float>   mask_j(n_expert);
    for (uint32_t il = 0; il < n_layer; il++) {
        if (!pl.layer_placed(il)) {
            continue;
        }
        const std::vector<int32_t> & perm = pl.perm  [il];
        const std::vector<int32_t> & cnt  = pl.counts[il];
        for (size_t j = 0; j < n_members; j++) {
            int32_t first_pos = 0;
            for (size_t k = 0; k < j; k++) {
                first_pos += cnt[k];
            }
            // non-owned experts get the SKIP sentinel: mul_mat_id requires a token's
            // ids to be distinct (top-k selects distinct experts), and collapsing every
            // non-owned lane onto one local slot violated that - the CUDA id helper
            // records one lane per (token, expert) while counting all of them, which
            // walked its index arithmetic out of bounds. See docs/expert-placement-plan.md
            std::fill(remap_j.begin(), remap_j.end(), LLAMA_EXPERT_SLOT_SKIP);
            std::fill(mask_j.begin(),  mask_j.end(),  0.0f);
            for (int32_t k = first_pos; k < first_pos + cnt[j]; k++) {
                remap_j[perm[k]] = k - first_pos;
                mask_j [perm[k]] = 1.0f;
            }
            llama_expert_placement_audit_tables(il, j, perm, cnt, remap_j, mask_j);
            // GGML_META_DEBUG>1 negative control: the audit must catch a
            // hand-corrupted table (a non-owned expert mapped off the dummy slot)
            if (meta_debug > 1 && !selftest_done) {
                selftest_done = true;
                std::vector<int32_t> bad = remap_j;
                for (uint32_t k = 0; k < n_expert; k++) {
                    if ((int32_t) k < first_pos || (int32_t) k >= first_pos + cnt[j]) {
                        bad[perm[k]] = 1;
                        break;
                    }
                }
                try {
                    llama_expert_placement_audit_tables(il, j, perm, cnt, bad, mask_j);
                    LLAMA_LOG_ERROR("%s: SELFTEST: static audit FAILED to catch an injected table error\n", __func__);
                } catch (const std::runtime_error & e) {
                    LLAMA_LOG_INFO("%s: SELFTEST: static audit caught injected table error: %s\n", __func__, e.what());
                }
            }
            ggml_backend_meta_tensor_set_member(tables->remap[il], j, remap_j.data(), 0, n_expert*sizeof(int32_t));
            ggml_backend_meta_tensor_set_member(tables->mask [il], j, mask_j.data(),  0, n_expert*sizeof(float));
        }
    }

    return tables;
}
