#include "llama-moe-cache.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-expert-placement.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cstring>

static bool moe_cache_is_host_weight(const ggml_tensor * t) {
    return t != nullptr && t->buffer != nullptr && ggml_backend_buffer_is_host(t->buffer);
}

std::unique_ptr<llama_moe_cache> llama_moe_cache::create(const llama_model & model, const llama_moe_cache_params & params) {
    if (params.budget_mib == 0) {
        return nullptr;
    }
    const auto & hparams = model.hparams;
    const int64_t n_expert = hparams.n_expert;
    if (n_expert == 0) {
        LLAMA_LOG_INFO("%s: MoE cache off (not a MoE model)\n", __func__);
        return nullptr;
    }

    struct cand {
        int il;
        ggml_tensor * up;
        ggml_tensor * gate;
        ggml_tensor * down;
        ggml_backend_buffer_type_t buft;
    };
    std::vector<cand> cands;
    auto skip = [&](int il, const char * why) {
        if (params.debug) {
            LLAMA_LOG_INFO("%s: layer %d not cacheable: %s\n", __func__, il, why);
        }
    };
    for (int il = 0; il < (int) hparams.n_layer(); ++il) {
        const auto & layer = model.layers[il];
        ggml_tensor * up   = layer.ffn_up_exps;
        ggml_tensor * gate = layer.ffn_gate_exps;
        ggml_tensor * down = layer.ffn_down_exps;
        if (up == nullptr || gate == nullptr || down == nullptr || layer.ffn_gate_up_exps != nullptr) {
            skip(il, "no separate gate/up/down expert tensors");
            continue;
        }
        if (!moe_cache_is_host_weight(up) || !moe_cache_is_host_weight(gate) || !moe_cache_is_host_weight(down)) {
            skip(il, up->buffer ? ggml_backend_buffer_name(up->buffer) : "experts not on a host buffer");
            continue;
        }
        if (up->data == nullptr || gate->data == nullptr || down->data == nullptr) {
            // measure-only model (the fit probe loads without data): nothing to cache yet
            return nullptr;
        }
        if (!ggml_is_quantized(up->type) || !ggml_is_quantized(gate->type) || !ggml_is_quantized(down->type)) {
            skip(il, "expert type not quantized");
            continue;
        }
        if (up->ne[2] != n_expert || gate->ne[2] != n_expert || down->ne[2] != n_expert) {
            skip(il, "expert count mismatch");
            continue;
        }
        ggml_tensor * router = layer.ffn_gate_inp;
        if (router == nullptr || router->buffer == nullptr) {
            skip(il, "no router tensor");
            continue;
        }
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(router->buffer);
        if (ggml_backend_buft_is_host(buft)) {
            if (!params.force_cpu) {
                skip(il, "router on the host: no device to cache on");
                continue;
            }
            buft = ggml_backend_cpu_buffer_type();
        } else if (strstr(ggml_backend_buft_name(buft), "Split") != nullptr) {
            skip(il, "router in a split buffer");
            continue;
        }
        cands.push_back({ il, up, gate, down, buft });
    }
    if (cands.empty()) {
        LLAMA_LOG_INFO("%s: MoE cache off (no host-resident quantized expert layer with a device router)\n", __func__);
        return nullptr;
    }

    // budget: a fixed total is clamped to the device's free VRAM minus the reserve (the KV and
    // compute buffers are already allocated at this point; the reserve covers pool growth and
    // whatever loads after the cache, e.g. a draft model); auto = free VRAM minus the reserve
    size_t budget_bytes = params.budget_mib > 0 ? (size_t) params.budget_mib * 1024 * 1024 : 0;
    {
        const size_t reserve = (size_t) params.reserve_mib * 1024 * 1024;
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(cands[0].buft);
        size_t free_mem = 0, total_mem = 0;
        if (dev != nullptr && !params.force_cpu) {
            ggml_backend_dev_memory(dev, &free_mem, &total_mem);
        }
        if (dev != nullptr && !params.force_cpu) {
            const size_t avail = free_mem > reserve ? free_mem - reserve : 0;
            if (params.budget_mib < 0) {
                budget_bytes = avail;
                LLAMA_LOG_INFO("%s: MoE cache auto budget: free %zu MiB - reserve %zu MiB = %zu MiB\n", __func__,
                        free_mem / (1024*1024), reserve / (1024*1024), budget_bytes / (1024*1024));
            } else if (budget_bytes > avail) {
                LLAMA_LOG_WARN("%s: MoE cache budget %d MiB exceeds free %zu MiB - reserve %zu MiB: clamped to %zu MiB\n", __func__,
                        params.budget_mib, free_mem / (1024*1024), reserve / (1024*1024), avail / (1024*1024));
                budget_bytes = avail;
            }
        } else if (params.budget_mib < 0) {
            LLAMA_LOG_INFO("%s: MoE cache off (auto budget needs a device)\n", __func__);
            return nullptr;
        }
        if (budget_bytes == 0) {
            LLAMA_LOG_WARN("%s: MoE cache off: no VRAM left after the %zu MiB reserve\n", __func__, reserve / (1024*1024));
            return nullptr;
        }
    }
    const size_t per_layer = budget_bytes / cands.size();

    std::unique_ptr<llama_moe_cache> mc(new llama_moe_cache());
    mc->params = params;

    size_t total_alloc = 0;
    for (const cand & c : cands) {
        const size_t slab = (size_t) c.up->nb[2] + (size_t) c.gate->nb[2] + (size_t) c.down->nb[2];
        int64_t n_slots = slab ? (int64_t) (per_layer / slab) : 0;
        n_slots = std::min<int64_t>(n_slots, n_expert);
        if (n_slots < params.min_slots) {
            if (params.debug) {
                LLAMA_LOG_INFO("%s: layer %d skipped: %lld slots < %d\n", __func__, c.il, (long long) n_slots, params.min_slots);
            }
            continue;
        }
        auto l = std::make_unique<llama_moe_cache_layer>();
        l->il       = c.il;
        l->n_expert = n_expert;
        l->up_src   = c.up;
        l->gate_src = c.gate;
        l->down_src = c.down;
        if (!mc->alloc_layer(*l, c.buft, (int32_t) n_slots)) {
            LLAMA_LOG_WARN("%s: layer %d: pool allocation failed (%lld slots), layer stays on the CPU\n", __func__,
                    c.il, (long long) n_slots);
            continue;
        }
        // one upload backend per device
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(c.buft);
        for (const auto & b : mc->backends) {
            if (ggml_backend_get_device(b.get()) == dev) {
                l->fill_backend = b.get();
                break;
            }
        }
        if (l->fill_backend == nullptr) {
            ggml_backend_t b = dev ? ggml_backend_dev_init(dev, nullptr) : ggml_backend_cpu_init();
            if (b == nullptr) {
                LLAMA_LOG_WARN("%s: layer %d: no upload backend for %s, layer stays on the CPU\n", __func__,
                        c.il, ggml_backend_buft_name(c.buft));
                continue;
            }
            mc->backends.emplace_back(b);
            l->fill_backend = b;
        }
        total_alloc += ggml_backend_buffer_get_size(l->buf.get());
        const int32_t n_fill = std::min<int32_t>(params.static_fill, l->n_slots);
        for (int32_t e = 0; e < n_fill; ++e) {
            mc->fill_slot(*l, e, e);
        }
        mc->flush_tables(*l);
        if (params.debug) {
            LLAMA_LOG_INFO("%s: layer %d: %d slots x %zu KiB on %s, %d pre-filled\n", __func__, c.il, l->n_slots,
                    slab / 1024, ggml_backend_buft_name(c.buft), n_fill);
        }
        mc->by_up[l->up_src] = l.get();
        mc->by_il[l->il]     = l.get();
        mc->layers.push_back(std::move(l));
    }
    if (mc->layers.empty()) {
        LLAMA_LOG_INFO("%s: MoE cache off (budget too small for %d slots in any layer)\n", __func__, params.min_slots);
        return nullptr;
    }
    LLAMA_LOG_INFO("%s: MoE cache on: %zu layers, %zu MiB of slots, max batch %d, %d inserts/layer/step, %d MiB/step\n",
            __func__, mc->layers.size(), total_alloc / (1024*1024), params.max_batch, params.inserts, params.step_mib);

    mc->worker = std::thread(&llama_moe_cache::worker_loop, mc.get());
    return mc;
}

llama_moe_cache::~llama_moe_cache() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        stop = true;
    }
    cv.notify_all();
    if (worker.joinable()) {
        worker.join();
    }
    if (params.stats_every > 0 || params.debug) {
        log_stats();
    }
    // layers (pools) are freed before the upload backends (member order)
    layers.clear();
}

bool llama_moe_cache::alloc_layer(llama_moe_cache_layer & l, ggml_backend_buffer_type_t buft, int32_t n_slots) {
    ggml_init_params ip = {
        /*.mem_size   =*/ 8*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    l.ctx.reset(ggml_init(ip));
    if (!l.ctx) {
        return false;
    }
    ggml_context * ctx = l.ctx.get();
    // one padding slab past the last slot: the MMQ path may read a few bytes past the last expert
    l.up_c   = ggml_new_tensor_3d(ctx, l.up_src->type,   l.up_src->ne[0],   l.up_src->ne[1],   n_slots + 1);
    l.gate_c = ggml_new_tensor_3d(ctx, l.gate_src->type, l.gate_src->ne[0], l.gate_src->ne[1], n_slots + 1);
    l.down_c = ggml_new_tensor_3d(ctx, l.down_src->type, l.down_src->ne[0], l.down_src->ne[1], n_slots + 1);
    l.gpu_table = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, l.n_expert);
    l.cpu_table = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, l.n_expert);
    ggml_format_name(l.up_c,      "blk.%d.moe_cache_up",    l.il);
    ggml_format_name(l.gate_c,    "blk.%d.moe_cache_gate",  l.il);
    ggml_format_name(l.down_c,    "blk.%d.moe_cache_down",  l.il);
    ggml_format_name(l.gpu_table, "blk.%d.moe_cache_gpu_t", l.il);
    ggml_format_name(l.cpu_table, "blk.%d.moe_cache_cpu_t", l.il);

    l.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft));
    if (!l.buf) {
        return false;
    }
    // weights usage: the scheduler runs the slot mul_mat_id and the table get_rows on this device
    ggml_backend_buffer_set_usage(l.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    l.n_slots = n_slots;
    l.slot_expert.assign(n_slots, -1);
    l.slot_last_use.assign(n_slots, 0);
    l.slot_hits.assign(n_slots, 0);
    l.slot_in_flight.assign(n_slots, 0);
    l.expert_slot.assign(l.n_expert, -1);
    l.miss_count.assign(l.n_expert, 0);
    l.expert_pending.assign(l.n_expert, 0);
    l.obs.assign((size_t) params.max_batch * 64 * 8, 0); // room for 8 ubatches of max_batch x 64 routed rows
    l.obs_len = 0;

    // zero the padding slabs
    ggml_backend_tensor_memset(l.up_c,   0, (size_t) n_slots * l.up_c->nb[2],   l.up_c->nb[2]);
    ggml_backend_tensor_memset(l.gate_c, 0, (size_t) n_slots * l.gate_c->nb[2], l.gate_c->nb[2]);
    ggml_backend_tensor_memset(l.down_c, 0, (size_t) n_slots * l.down_c->nb[2], l.down_c->nb[2]);
    return true;
}

// synchronous create-time fill (the static gate instrument); residency bookkeeping included
void llama_moe_cache::fill_slot(llama_moe_cache_layer & l, int32_t expert, int32_t slot) {
    GGML_ASSERT(expert >= 0 && expert < l.n_expert && slot >= 0 && slot < l.n_slots);
    const ggml_tensor * srcs[3] = { l.up_src, l.gate_src, l.down_src };
    ggml_tensor *       dsts[3] = { l.up_c,   l.gate_c,   l.down_c   };
    for (int i = 0; i < 3; ++i) {
        const size_t nb = srcs[i]->nb[2];
        GGML_ASSERT(nb == dsts[i]->nb[2]);
        ggml_backend_tensor_set(dsts[i], (const char *) srcs[i]->data + (size_t) expert * nb, (size_t) slot * nb, nb);
    }
    if (l.slot_expert[slot] >= 0) {
        l.expert_slot[l.slot_expert[slot]] = -1;
    }
    l.slot_expert[slot]   = expert;
    l.expert_slot[expert] = slot;
}

// worker-side copy: async on the upload backend, no bookkeeping (step() publishes)
void llama_moe_cache::copy_slot(ggml_backend_t backend, llama_moe_cache_layer & l, int32_t expert, int32_t slot) {
    const ggml_tensor * srcs[3] = { l.up_src, l.gate_src, l.down_src };
    ggml_tensor *       dsts[3] = { l.up_c,   l.gate_c,   l.down_c   };
    for (int i = 0; i < 3; ++i) {
        const size_t nb = srcs[i]->nb[2];
        ggml_backend_tensor_set_async(backend, dsts[i], (const char *) srcs[i]->data + (size_t) expert * nb, (size_t) slot * nb, nb);
    }
}

void llama_moe_cache::flush_tables(llama_moe_cache_layer & l) {
    std::vector<int32_t> gpu(l.n_expert), cpu(l.n_expert);
    for (int64_t e = 0; e < l.n_expert; ++e) {
        const int32_t s = l.expert_slot[e];
        gpu[e] = s >= 0 ? s : LLAMA_EXPERT_SLOT_SKIP;
        cpu[e] = s >= 0 ? LLAMA_EXPERT_SLOT_SKIP : (int32_t) e;
    }
    ggml_backend_tensor_set(l.gpu_table, gpu.data(), 0, gpu.size()*sizeof(int32_t));
    ggml_backend_tensor_set(l.cpu_table, cpu.data(), 0, cpu.size()*sizeof(int32_t));
}

const llama_moe_cache_layer * llama_moe_cache::lookup(const ggml_tensor * up_exps) const {
    auto it = by_up.find(up_exps);
    return it == by_up.end() ? nullptr : it->second;
}

void llama_moe_cache::observe(ggml_backend_sched_t sched, const std::vector<std::pair<int, ggml_tensor *>> & ids) {
    for (const auto & [il, t] : ids) {
        auto it = by_il.find(il);
        if (it == by_il.end() || t == nullptr) {
            continue;
        }
        llama_moe_cache_layer & l = *it->second;
        const size_t n = (size_t) ggml_nelements(t);
        if (l.obs_len + n > l.obs.size()) {
            n_obs_dropped++;
            continue;
        }
        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, t);
        if (backend == nullptr) {
            n_obs_dropped++;
            continue;
        }
        ggml_backend_tensor_get_async(backend, t, l.obs.data() + l.obs_len, 0, n*sizeof(int32_t));
        l.obs_len += n;
    }
}

void llama_moe_cache::step() {
    n_steps++;

    // 1) publish the uploads the worker completed (their slots were unreferenced since eviction)
    std::vector<job> finished;
    {
        std::lock_guard<std::mutex> lock(mtx);
        finished.swap(done);
    }
    std::vector<bool> dirty(layers.size(), false);
    for (const job & j : finished) {
        llama_moe_cache_layer & l = *layers[j.layer];
        if (l.slot_expert[j.slot] >= 0) {
            l.expert_slot[l.slot_expert[j.slot]] = -1;
        }
        l.slot_expert[j.slot]      = j.expert;
        l.expert_slot[j.expert]    = j.slot;
        l.slot_in_flight[j.slot]   = 0;
        l.expert_pending[j.expert] = 0;
        l.slot_last_use[j.slot]    = n_steps;
        l.slot_hits[j.slot]        = 0;
        l.miss_count[j.expert]     = 0;
        dirty[j.layer] = true;
        n_fills++;
    }

    // 2) consume the observations: hits refresh recency, misses accumulate demand
    std::vector<std::vector<int32_t>> cands(layers.size());
    for (size_t li = 0; li < layers.size(); ++li) {
        llama_moe_cache_layer & l = *layers[li];
        auto & cand = cands[li];
        for (size_t i = 0; i < l.obs_len; ++i) {
            const int32_t e = l.obs[i];
            if (e < 0 || e >= l.n_expert) {
                continue;
            }
            const int32_t s = l.expert_slot[e];
            if (s >= 0) {
                l.slot_last_use[s] = n_steps;
                if (l.slot_hits[s] < UINT16_MAX) l.slot_hits[s]++;
                n_hits++;
            } else {
                if (l.miss_count[e] < UINT16_MAX) l.miss_count[e]++;
                n_misses++;
                if (!l.expert_pending[e] && std::find(cand.begin(), cand.end(), e) == cand.end()) {
                    cand.push_back(e);
                }
            }
        }
        l.obs_len = 0;
        if ((n_steps % 64) == 0) {
            for (auto & h : l.slot_hits) h >>= 1; // age the heat so a formerly hot expert can leave
        }
        // most-demanded first
        std::sort(cand.begin(), cand.end(), [&](int32_t a, int32_t b) { return l.miss_count[a] > l.miss_count[b]; });
    }

    // schedule uploads round-robin over the layers so the per-step byte cap is shared fairly
    size_t step_bytes = 0;
    const size_t step_cap = (size_t) params.step_mib * 1024 * 1024;
    std::vector<job> scheduled;
    std::vector<size_t> next(layers.size(), 0);
    for (int round = 0; round < params.inserts && step_bytes < step_cap; ++round) {
        for (size_t li = 0; li < layers.size() && step_bytes < step_cap; ++li) {
            llama_moe_cache_layer & l = *layers[li];
            auto & cand = cands[li];
            const size_t slab = (size_t) l.up_src->nb[2] + (size_t) l.gate_src->nb[2] + (size_t) l.down_src->nb[2];
            if (step_bytes + slab > step_cap) {
                break;
            }
            while (next[li] < cand.size()) {
                const int32_t e = cand[next[li]++];
                // a free slot admits on the first miss; a full pool needs a repeat miss
                int32_t victim = -1;
                for (int32_t s = 0; s < l.n_slots; ++s) {
                    if (l.slot_expert[s] < 0 && !l.slot_in_flight[s]) { victim = s; break; }
                }
                if (victim < 0) {
                    if (l.miss_count[e] < 2) {
                        continue;
                    }
                    // LRU among resident, not in flight, not hit this step; cold slots first
                    int32_t cold = -1, any = -1;
                    uint64_t cold_t = UINT64_MAX, any_t = UINT64_MAX;
                    for (int32_t s = 0; s < l.n_slots; ++s) {
                        if (l.slot_in_flight[s] || l.slot_last_use[s] == n_steps) {
                            continue;
                        }
                        if (l.slot_last_use[s] < any_t) { any_t = l.slot_last_use[s]; any = s; }
                        if (l.slot_hits[s] <= params.hot_uses && l.slot_last_use[s] < cold_t) { cold_t = l.slot_last_use[s]; cold = s; }
                    }
                    victim = cold >= 0 ? cold : any;
                    if (victim < 0) {
                        next[li] = cand.size(); // nothing evictable this step
                        break;
                    }
                    // evict now: the table entry is cleared before the next graph can run
                    l.expert_slot[l.slot_expert[victim]] = -1;
                    l.slot_expert[victim] = -1;
                    dirty[li] = true;
                    n_evict++;
                }
                l.slot_in_flight[victim] = 1;
                l.expert_pending[e]      = 1;
                scheduled.push_back({ li, e, victim });
                step_bytes += slab;
                break; // one upload per layer per round
            }
        }
    }

    // 3) one table write per changed layer, then hand the copies to the worker
    for (size_t li = 0; li < layers.size(); ++li) {
        if (dirty[li]) {
            flush_tables(*layers[li]);
        }
    }
    if (!scheduled.empty()) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            for (const job & j : scheduled) {
                todo.push_back(j);
            }
        }
        cv.notify_one();
    }

    if (params.stats_every > 0 && (n_steps % params.stats_every) == 0) {
        log_stats();
    }
}

void llama_moe_cache::worker_loop() {
    std::vector<job> batch;
    std::vector<ggml_backend_t> touched;
    while (true) {
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&] { return stop || !todo.empty(); });
            if (stop) {
                return;
            }
            batch.assign(todo.begin(), todo.end());
            todo.clear();
        }
        touched.clear();
        for (const job & j : batch) {
            llama_moe_cache_layer & l = *layers[j.layer];
            copy_slot(l.fill_backend, l, j.expert, j.slot);
            if (std::find(touched.begin(), touched.end(), l.fill_backend) == touched.end()) {
                touched.push_back(l.fill_backend);
            }
        }
        for (ggml_backend_t b : touched) {
            ggml_backend_synchronize(b);
        }
        {
            std::lock_guard<std::mutex> lock(mtx);
            done.insert(done.end(), batch.begin(), batch.end());
        }
    }
}

void llama_moe_cache::log_stats() {
    const uint64_t probes = n_hits + n_misses;
    size_t resident = 0, pending = 0;
    for (const auto & l : layers) {
        for (int32_t s = 0; s < l->n_slots; ++s) {
            resident += l->slot_expert[s] >= 0;
            pending  += l->slot_in_flight[s];
        }
    }
    LLAMA_LOG_INFO("moe-cache: step %llu hit %llu / %llu (%.1f%%) fills %llu evict %llu resident %zu in-flight %zu obs-dropped %llu\n",
            (unsigned long long) n_steps, (unsigned long long) n_hits, (unsigned long long) probes,
            probes ? 100.0 * (double) n_hits / (double) probes : 0.0,
            (unsigned long long) n_fills, (unsigned long long) n_evict, resident, pending, (unsigned long long) n_obs_dropped);
}
