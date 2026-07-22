# DeepSeek-V4-Flash on one box: single-box -ncmoe vs the RPC fleet (TASKS.md #54)

Measured 2026-07-22 on the 2x Tesla V100-SXM2-32GB box (78 GiB RAM, model on
nvme0n1). Model: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf`
(86.7 GB, ~80 GB of routed experts across 43 MoE layers). All decode numbers are
server-reported `predicted_per_second` for 128-token temp-0 generations; disk
numbers are nvme0n1 sector deltas per generation. Reproduce with
`run-ncmoe-54-measure.sh` (fork) and `run-ncmoe-54-upstream-bench.sh` (upstream).

## Headline

| config | decode t/s | prefill t/s | NVMe/token (warm) |
|---|---|---|---|
| RPC layer-split fleet (coordinator + LAN workers) | 4.79 | 12.5 | 0 (--no-mmap) |
| single-box `-ngl 99 -ncmoe 99` (all experts CPU RAM) | 6.64 | 11.6 | 4 MB |
| single-box `-ngl 99 -ncmoe 37` (last 6 MoE layers' experts in VRAM) | **7.31** | 13.6 | 1-2 MB |

Single-box wins by +53%. Coherence-gated: chat-endpoint answers (Rayleigh
scattering, primes) correct and word-identical to the fleet's where comparable.

## Why the old numbers said the opposite

The task was filed because single-box measured 3.54 t/s vs the fleet's 4.6, with
a working theory of an unbeatable NVMe paging floor (~0.9 GB/token from disk).
Both the number and the theory fell:

1. **The 3.54 was RAM-starved by its own competitor.** It was measured while the
   fleet's local CPU RPC worker held 33 GiB of the box's RAM, leaving ~46 GiB.
   With the worker stopped, the page cache grows to 74 GiB and effectively all
   ~80 GiB of experts stay resident (routing skew covers the tail): NVMe reads
   fall from 184 MB/token (cold, 3.73 t/s) to 1-4 MB/token (warm, 6.6-7.3 t/s)
   within five 128-token generations.
2. **The disk-floor arithmetic never closed.** Measured NVMe ceiling (O_DIRECT
   random preads over the gguf): 0.67 GB/s at 256K QD1, 1.67 GB/s at 1M QD1,
   2.65 GB/s at 4M x4 threads. A 50% hit rate (0.9 GB/token from disk) would
   cost 340 ms/token at the ceiling - more than the whole 282 ms token observed
   back then. The old runs were already at ~85%+ effective hit rate; the fleet's
   edge was RAM starvation, not zero-disk vs disk.

## Placement law for `-ncmoe` on a 2-GPU layer split

`-ncmoe N` keeps the FIRST N MoE layers' experts on CPU; the remaining layers'
experts are placed in VRAM **with their layer**. In a 2-way layer split the tail
layers sit almost entirely on GPU1, so the VRAM expert budget is ONE card's free
memory, not the sum: `-ncmoe 20` (23 expert layers, ~43 GB) asked for a single
40 GiB CUDA1 buffer and OOM'd. With ~12 GiB free per card at 8k ctx the budget
is ~6 expert layers -> `-ncmoe 37`. Moving those 6 layers (14% of expert reads)
from RAM to VRAM bought +10% end-to-end (6.64 -> 7.31).

## Production recommendation (this box, V4)

- `-ngl 99 -ncmoe 37 -t 10 -c 8192 --no-warmup`, mmap ON (paging IS the loader),
  `-fit off`.
- Do NOT run the local CPU RPC worker on this box while serving V4 single-box -
  it eats the page cache that makes this fast.
- More context shrinks the VRAM expert budget; back off toward `-ncmoe 40` if
  the load OOMs.
- The fleet remains the answer only for models that exceed local RAM+VRAM
  (e.g. GLM-5.2 at ~240 GB).

## Upstream comparison (benchmark-only)

Same benchmark against an UNMODIFIED upstream llama.cpp checkout - build and
run isolated in `llama.cpp-upstream-bench/` + a CUDA 12.8 container (host CUDA
13 dropped sm70; the driver-API link needs `--gpus all` in the build container
so libcuda is injected), nothing in upstream is patched. Script:
`run-ncmoe-54-upstream-bench.sh`.

Upstream `ggml-org/llama.cpp` @ `6d5a910` (2026-07-22), identical config
(`-ngl 99 -ncmoe 37 -t 10 -c 8192 --no-warmup`, np1, same model file):

| gen | decode t/s | NVMe/token |
|---|---|---|
| 1 (re-warm) | 7.58 | 87 MB |
| 2 | 8.96 | 14 MB |
| 3 | 9.68 | 3 MB |
| 4 | 9.66 | 2 MB |
| 5 | **9.61** | 2 MB |

Prefill 14.4-15.4 t/s. Coherence gate passed - the chat Rayleigh answer is
word-identical to both the fork's and the fleet's.

**Upstream is ~31% faster than the fork on the identical single-box path**
(9.6 vs 7.31 t/s steady; prefill 15.4 vs 13.6). Closing that gap is TASKS.md
#65 - executed the same day, see below.

## #65 execution: where the 31% came from, and the merge that closed it

Attribution (same benchmark, three more builds):

- Upstream at the FORK'S MERGE-BASE `2da668617` (2026-07-05, only 17 days
  behind): **7.22-7.25 t/s** - identical to the fork's 7.31. The fork has NO
  regression; the entire gap is upstream work landed in those 17 days, and in
  this `-sm layer` single-box config none of the fork's V100 machinery (meta
  backend, NVLink allreduce, MTP) is active to compensate.
- The responsible upstream work (all V4/Volta-specific): fused hyper-connection
  ops (`0dc74e332`), `GGML_OP_LIGHTNING_INDEXER` CPU+CUDA (`00f5442cc`,
  `3b5321936`), fused `sqrt_softplus` topk-moe router (`846e991ec`), and CUDA
  graphs enabled on Volta (`3f08ef2c5`). The fork ran V4's indexer and
  hyper-connections as generic op compositions.
- Fix: merged upstream `6d5a910` into the fork (branch `merge-65-upstream`,
  17 conflict hunks in 14 files). Notable resolutions: upstream's `n_keep_tail`
  argument to `split_equal()` supersedes the fork's `full_seqs` (upstream
  adopted + refined the fork's #11 fix); fork's fleet/server/loading-page/DSA
  split-state code kept; checkpoint tail-bounding preserved into upstream's new
  struct shape. Plus one wire-protocol fix the merge FORCED: the 4 new ggml ops
  shift the RPC op enum, and the handshake only checked the major version -
  proto bumped to 4.11.3 and HELLO now rejects op-fingerprint (patch)
  mismatches, so pre-merge workers fail loudly instead of decoding wrong ops.

**Merged-fork result: 9.71-9.89 t/s decode / 16.0 t/s prefill** - matches and
slightly beats unmodified upstream. Coherence gate passed (word-identical
Rayleigh answer); the RPC/fleet path re-validated on a loopback worker (proto
4.11.3 handshake, layer split, coherent decode, #63 init times intact).

Final ledger for V4 on this box:

| serving path | decode t/s |
|---|---|
| fork RPC fleet | 4.79 |
| pre-merge fork single-box `-ncmoe 37` | 7.31 |
| upstream @ merge-base (Jul 5) single-box | 7.22-7.25 |
| upstream @ 6d5a910 (Jul 22) single-box | 9.61-9.68 |
| **merged fork (`merge-65-upstream`) single-box** | **9.71-9.89** |

Deployment note: landing the merge requires rebuilding ALL worker images from
the merged tree (proto 4.11.3 rejects pre-merge workers by design - the op
enum shifted).
