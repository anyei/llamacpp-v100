#include "reasoning-budget.h"
#include "common.h"
#include "trie.h"
#include "unicode.h"

#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

struct token_matcher {
    std::vector<llama_tokens> seqs;
    common_aho_corasick ac;
    size_t state = 0;

    token_matcher(const std::vector<llama_tokens> & seqs) : seqs(collect(seqs)), ac(build_trie(this->seqs)) {}

    static std::vector<llama_tokens> collect(const std::vector<llama_tokens> & seqs) {
        std::vector<llama_tokens> res;
        for (const auto & seq : seqs) {
            if (!seq.empty() && std::find(res.begin(), res.end(), seq) == res.end()) {
                res.push_back(seq);
            }
        }
        return res;
    }

    static common_trie build_trie(const std::vector<llama_tokens> & seqs) {
        common_trie t;
        for (const auto & seq : seqs) {
            t.insert(std::vector<uint32_t>(seq.begin(), seq.end()));
        }
        return t;
    }

    // returns the index into seqs of the longest sequence ending at this token, or -1
    int32_t advance(llama_token token) {
        state = ac.next(state, (uint32_t) token);
        const int32_t p = ac.match_pattern(state);
        if (p >= 0) {
            state = 0;
        }
        return p;
    }

    void reset() { state = 0; }
};

struct common_reasoning_budget_ctx {
    const llama_vocab * vocab;

    token_matcher start_matcher;
    token_matcher end_matcher;
    llama_tokens forced_tokens;

    int32_t budget;           // maximum tokens in reasoning block
    int32_t remaining;        // tokens remaining in budget

    common_reasoning_budget_state state;

    // for forcing
    size_t force_pos;         // next position in forced_tokens to force

    int32_t end_match;        // index into end_matcher.seqs of the sequence that transitioned to DONE, -1 if none

    // #139 soft warning: injected while still COUNTING, then counting resumes
    llama_tokens warn_tokens; // sequence forced as the warning (empty = disabled)
    int32_t warn_at;          // fire when remaining <= warn_at (< 0 = disabled)
    size_t  warn_pos;         // next position in warn_tokens to force
    bool    warn_fired;       // already warned in this reasoning block
};

// the warning is armed only when it can still precede the hard cut
static bool common_reasoning_budget_warn_armed(const common_reasoning_budget_ctx * ctx) {
    return !ctx->warn_fired && ctx->warn_at >= 0 && !ctx->warn_tokens.empty();
}

static const char * common_reasoning_budget_name(const struct llama_sampler * /*smpl*/) {
    return "reasoning-budget";
}

static void common_reasoning_budget_accept(struct llama_sampler * smpl, llama_token token) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    switch (ctx->state) {
        case REASONING_BUDGET_IDLE:
        {
            if (ctx->start_matcher.advance(token) >= 0) {
                ctx->state = REASONING_BUDGET_COUNTING;
                ctx->remaining = ctx->budget;
                ctx->warn_pos = 0;
                ctx->warn_fired = false;
                COM_TRC("activated, budget=%d tokens\n", ctx->budget);

                if (ctx->remaining <= 0) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    COM_TRC("%s", "budget=0, forcing immediately\n");
                }
            }
            break;
        }
        case REASONING_BUDGET_COUNTING:
        case REASONING_BUDGET_WAITING_UTF8:
        {
            const int32_t match = ctx->end_matcher.advance(token);
            if (match >= 0) {
                ctx->state = REASONING_BUDGET_DONE;
                ctx->end_match = match;
                COM_TRC("%s", "deactivated (natural end)\n");
                break;
            }

            bool utf8_complete = true;
            if (ctx->vocab != nullptr) {
                const std::string piece = common_token_to_piece(ctx->vocab, token, false);
                utf8_complete = common_utf8_is_complete(piece);
            }

            if (ctx->state == REASONING_BUDGET_WAITING_UTF8) {
                if (utf8_complete) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    ctx->end_matcher.reset();
                    COM_TRC("%s", "UTF-8 complete, now forcing end sequence\n");
                }
            } else if (ctx->state == REASONING_BUDGET_COUNTING) {
                ctx->remaining--;
                if (ctx->remaining <= 0) {
                    if (utf8_complete) {
                        ctx->state = REASONING_BUDGET_FORCING;
                        ctx->force_pos = 0;
                        ctx->end_matcher.reset();
                        COM_TRC("%s", "budget exhausted, forcing end sequence\n");
                    } else {
                        ctx->state = REASONING_BUDGET_WAITING_UTF8;
                        ctx->end_matcher.reset();
                        COM_TRC("%s", "budget exhausted, waiting for UTF-8 completion\n");
                    }
                } else if (common_reasoning_budget_warn_armed(ctx) && ctx->remaining <= ctx->warn_at && utf8_complete) {
                    // soft warning (#139): inject the nudge, then resume counting so the
                    // model can close the block itself. Not UTF-8 complete? do nothing -
                    // the condition still holds next token, so this self-heals.
                    ctx->state = REASONING_BUDGET_WARNING;
                    ctx->warn_pos = 0;
                    ctx->warn_fired = true;
                    ctx->end_matcher.reset();
                    COM_TRC("budget low (%d left), injecting warning\n", ctx->remaining);
                }
            }
            break;
        }
        case REASONING_BUDGET_WARNING:
        {
            // forced warning tokens do not consume budget and must not be able to
            // complete an end sequence straddling the injection
            ctx->warn_pos++;
            if (ctx->warn_pos >= ctx->warn_tokens.size()) {
                ctx->state = REASONING_BUDGET_COUNTING;
                ctx->end_matcher.reset();
                COM_TRC("warning injected, resuming count (%d left)\n", ctx->remaining);
            }
            break;
        }
        case REASONING_BUDGET_FORCING:
        {
            // track the end sequence within forced_tokens so it is also reported on DONE
            const int32_t match = ctx->end_matcher.advance(token);
            ctx->force_pos++;
            if (ctx->force_pos >= ctx->forced_tokens.size()) {
                ctx->state = REASONING_BUDGET_DONE;
                ctx->end_match = match;
                COM_TRC("%s", "forced sequence complete, done\n");
            }
            break;
        }
        case REASONING_BUDGET_DONE:
            // Re-arm on a new start tag: some models emit multiple <think> blocks
            // per response, and each should get a fresh budget window.
            if (ctx->start_matcher.advance(token) >= 0) {
                ctx->state = REASONING_BUDGET_COUNTING;
                ctx->remaining = ctx->budget;
                ctx->end_matcher.reset();
                ctx->end_match = -1;
                ctx->warn_pos = 0;
                ctx->warn_fired = false;
                COM_TRC("re-activated on new start tag, budget=%d tokens\n", ctx->budget);

                if (ctx->remaining <= 0) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    COM_TRC("%s", "budget=0, forcing immediately\n");
                }
            }
            break;
    }
}

static void common_reasoning_budget_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    if (ctx->state != REASONING_BUDGET_FORCING && ctx->state != REASONING_BUDGET_WARNING) {
        // passthrough — don't modify logits
        return;
    }

    const bool warning = ctx->state == REASONING_BUDGET_WARNING;

    const llama_tokens & seq = warning ? ctx->warn_tokens  : ctx->forced_tokens;
    const size_t         pos = warning ? ctx->warn_pos     : ctx->force_pos;

    if (pos >= seq.size()) {
        return;
    }

    const llama_token forced = seq[pos];

    // set all logits to -inf except the forced token
    for (size_t i = 0; i < cur_p->size; i++) {
        if (cur_p->data[i].id != forced) {
            cur_p->data[i].logit = -INFINITY;
        }
    }
}

static void common_reasoning_budget_reset(struct llama_sampler * smpl) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;
    ctx->state = REASONING_BUDGET_IDLE;
    ctx->remaining = ctx->budget;
    ctx->start_matcher.reset();
    ctx->end_matcher.reset();
    ctx->force_pos = 0;
    ctx->end_match = -1;
    ctx->warn_pos = 0;
    ctx->warn_fired = false;
}

static struct llama_sampler * common_reasoning_budget_init_state(
        const struct llama_vocab * vocab, const std::vector<llama_tokens> & start_seqs,
        const std::vector<llama_tokens> & end_seqs, const llama_tokens & forced_tokens,
        int32_t budget, common_reasoning_budget_state initial_state,
        const llama_tokens & warn_tokens, int32_t warn_at);

static struct llama_sampler * common_reasoning_budget_clone(const struct llama_sampler * smpl);

static void common_reasoning_budget_free(struct llama_sampler * smpl) {
    delete (common_reasoning_budget_ctx *) smpl->ctx;
}

static struct llama_sampler_i common_reasoning_budget_i = {
    /* .name              = */ common_reasoning_budget_name,
    /* .accept            = */ common_reasoning_budget_accept,
    /* .apply             = */ common_reasoning_budget_apply,
    /* .reset             = */ common_reasoning_budget_reset,
    /* .clone             = */ common_reasoning_budget_clone,
    /* .free              = */ common_reasoning_budget_free,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
};

static struct llama_sampler * common_reasoning_budget_clone(const struct llama_sampler * smpl) {
    const auto * ctx = (const common_reasoning_budget_ctx *) smpl->ctx;

    return llama_sampler_init(
        /* .iface = */ &common_reasoning_budget_i,
        /* .ctx   = */ new common_reasoning_budget_ctx(*ctx)
    );
}

static struct llama_sampler * common_reasoning_budget_init_state(
        const struct llama_vocab        * vocab,
        const std::vector<llama_tokens> & start_seqs,
        const std::vector<llama_tokens> & end_seqs,
        const llama_tokens              & forced_tokens,
        int32_t                           budget,
        common_reasoning_budget_state     initial_state,
        const llama_tokens              & warn_tokens,
        int32_t                           warn_at) {
    // promote COUNTING with budget <= 0 to FORCING
    if (initial_state == REASONING_BUDGET_COUNTING && budget <= 0) {
        initial_state = REASONING_BUDGET_FORCING;
    }

    return llama_sampler_init(
        /* .iface = */ &common_reasoning_budget_i,
        /* .ctx   = */ new common_reasoning_budget_ctx {
            /* .vocab         = */ vocab,
            /* .start_matcher = */ token_matcher(start_seqs),
            /* .end_matcher   = */ token_matcher(end_seqs),
            /* .forced_tokens = */ forced_tokens,
            /* .budget        = */ budget,
            /* .remaining     = */ budget,
            /* .state         = */ initial_state,
            /* .force_pos     = */ 0,
            /* .end_match     = */ -1,
            /* .warn_tokens   = */ warn_tokens,
            /* .warn_at       = */ warn_at,
            /* .warn_pos      = */ 0,
            /* .warn_fired    = */ false,
        }
    );
}

struct llama_sampler * common_reasoning_budget_init(
        const struct llama_vocab        * vocab,
        const std::vector<llama_tokens> & start_seqs,
        const std::vector<llama_tokens> & end_seqs,
        const llama_tokens              & forced_tokens,
        int32_t                           budget,
        common_reasoning_budget_state     initial_state,
        const llama_tokens              & warn_tokens,
        int32_t                           warn_at) {
    return common_reasoning_budget_init_state(vocab, start_seqs, end_seqs, forced_tokens, budget, initial_state, warn_tokens, warn_at);
}

common_reasoning_budget_state common_reasoning_budget_get_state(const struct llama_sampler * smpl) {
    if (!smpl) {
        return REASONING_BUDGET_IDLE;
    }
    return ((const common_reasoning_budget_ctx *)smpl->ctx)->state;
}

const llama_tokens * common_reasoning_budget_get_end_match(const struct llama_sampler * smpl) {
    if (!smpl) {
        return nullptr;
    }

    const auto * ctx = (const common_reasoning_budget_ctx *) smpl->ctx;
    if (ctx->end_match < 0) {
        return nullptr;
    }

    return &ctx->end_matcher.seqs[ctx->end_match];
}

bool common_reasoning_budget_force(struct llama_sampler * smpl) {
    if (!smpl) {
        return false;
    }

    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    // only a sampler that is actively counting down the budget may be forced;
    // any other state (idle, already forcing/waiting, or done) is left untouched
    if (ctx->state != REASONING_BUDGET_COUNTING) {
        return false;
    }

    ctx->state = REASONING_BUDGET_FORCING;
    ctx->force_pos = 0;
    ctx->end_matcher.reset();
    COM_TRC("%s", "forced into forcing state (manual transition)\n");

    return true;
}
