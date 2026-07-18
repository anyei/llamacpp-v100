#!/usr/bin/env bash
# Hunyuan 3 fleet with MTP speculative decoding - thin wrapper so the model,
# image pin and fleet config live in ONE place (run-fleet-hy3-autoweight.sh).
#
# What MTP=1 adds there (TASKS.md #51/#52):
#   --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-min 1 --spec-draft-p-min 0.75
#   LLAMA_SPEC_DRAFT_NO_PAD=1  (p-min actually gates the draft chain - without it
#                               hy3's single-depth head force-proposes junk at
#                               depths 2-3: 32% acceptance, slower than plain)
#   LLAMA_SPEC_TIMING=1        (per-phase draft/verify timings in the log)
#
# Verify checklist for a run (#52): acceptance-per-proposed ~90%?, decode t/s
# above the 1.9 plain baseline?, output COHERENT (read it, not just t/s).
set -euo pipefail
cd "$(dirname "$0")"
MTP=1 exec ./run-fleet-hy3-autoweight.sh "$@"
