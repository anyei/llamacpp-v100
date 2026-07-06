# True Distributed Inference: Coordinator + N Workers

Status: analysis / design document. Nothing here is implemented yet beyond what is
marked "exists today". Target deployment: 1 coordinator docker instance + N worker
docker instances, each bound to its own GPU(s), on one or several hosts.
Reference hardware: Tesla V100-SXM2 (Volta, sm70, NVLink NV2 pairs), scaling from
2 to 4+ GPUs.

## 1. What exists today

### 1.1 RPC backend (worker side: `rpc-server`, `ggml/src/ggml-rpc/`)

- A worker runs `rpc-server` and exposes all its local accelerators as ggml
  devices over a hand-rolled binary TCP protocol (optional RDMA/RoCEv2).
- The coordinator is a normal `llama-server`/`llama-cli` started with
  `--rpc hostA:port,hostB:port,...`. Remote devices then behave like local ones
  for every split mode.
- Model weights and the KV cache for layers assigned to a worker are allocated
  and stay on the worker (`RPC_CMD_ALLOC_BUFFER`); only activations at split
  boundaries cross the wire. Weight upload can be skipped on restart via a
  worker-side file cache keyed by tensor hash (`SET_TENSOR_HASH`, `-c` flag).
- Graph shipping is optimized: the full graph is serialized once, and unchanged
  graphs re-run via a tiny `GRAPH_RECOMPUTE` message.

### 1.2 Split modes across devices

- `-sm layer` (pipeline): each device owns a contiguous layer slab + its KV.
  One activation handoff (n_embd x n_tokens) per stage boundary per microbatch.
  Works over RPC today and is the only mode that is *practical* over a network.
- `-sm tensor` (tensor parallel, meta backend `ggml-backend-meta.cpp`): each
  layer's weights are sharded across all devices; ~2 AllReduces per layer per
  step. On a single host this uses NCCL (or an internal CUDA AllReduce) and is
  the best mode for token-generation latency (measured on 2x V100: tg 47.8 t/s
  vs 33.2 for layer; pp 1576 vs 1096).

### 1.3 Hard limitations for distributed use (verified in source)

| Limitation | Where |
|---|---|
| NCCL init is single-process only (`ncclCommInitAll`) | `ggml-cuda.cu:1393-1414` |
| Meta-backend AllReduce over RPC falls back to a butterfly that stages every slice through the coordinator host (GET_TENSOR + host add + SET_TENSOR per step) | `ggml-backend-meta.cpp:2073-2217`, `ggml-backend.cpp:460-519` |
| RPC backend exposes no `comm_init`/`comm_allreduce` proc address, so the fast AllReduce path is never taken | `ggml-rpc.cpp:1883-1893` |
| Worker-to-worker copies refused (`COPY_TENSOR` only within one endpoint) | `ggml-rpc.cpp:501-521` |
| RPC is fully synchronous: no async ops, one in-flight command per socket | `ggml-rpc.cpp:652-655, 1798-1803` |
| No fault tolerance: any bad response aborts the coordinator | `ggml-rpc.cpp:30` |
| No auth/TLS ("never run on an open network") | `tools/rpc/README.md`, `rpc-server.cpp:301-309` |
| `supports_op` always true; capability negotiation absent | `ggml-rpc.cpp:1822-1827` |

## 2. Network budget: why mode choice matters

Per-token traffic for Qwen3.6-27B class models (n_embd = 5120, ~64 layers,
fp16/bf16 activations, single stream):

- Tensor parallel: 2 AllReduces per layer x 64 layers = 128 sync points per
  token. Payload per AllReduce at batch 1 is n_embd x 2 B = 10 KiB - trivial
  bandwidth, but each sync is a round trip. Over TCP with ~50-100 us RTT that
  is 128 x (RTT + software overhead) = 10-25 ms per token of pure latency,
  i.e. it would erase the entire 21 ms/token budget. Over NVLink/NCCL on one
  host the same syncs cost ~20-40 us each - that is why single-host TP works.
- Pipeline (layer split): N_stages - 1 handoffs per token (e.g. 3 for 4 GPUs),
  each n_embd x 2 B. Latency adds ~3 x RTT per token: sub-millisecond even on
  ordinary Ethernet. But batch-1 latency does not improve with more stages
  (each token still visits every layer sequentially); only throughput under
  concurrent load improves via pipelining.

Conclusion: **tensor parallel must stay inside a fast-interconnect domain
(NVLink/PCIe within a host); pipeline parallelism is the only sane cross-host
mode.** The right multi-host architecture is hierarchical: TP within each
worker, pipeline between workers.

## 3. Important caveat for the 4x V100 single-host plan

If all 4 V100s end up in ONE host, "distributed" (coordinator + worker
containers) is the wrong tool: a single llama-server process with
`-sm tensor -ts 0.25,0.25,0.25,0.25` and NCCL will beat any multi-container
split, because NCCL AllReduce over NVLink is 2-3 orders of magnitude lower
latency than loopback TCP RPC. Containers add nothing but isolation overhead
on one host. The coordinator/worker design pays off only when GPUs live in
different hosts.

Caveat inside the caveat: V100-SXM2 NVLink topology is usually pairwise (NV2
between neighbors, PCIe across pairs). `nvidia-smi topo -m` on the 4-GPU host
will tell whether all pairs are NVLinked; if only pairs are linked, expect
AllReduce over 4 GPUs to be bottlenecked by the PCIe hops (NCCL handles this
with ring construction, but per-sync latency rises). Measure tg with
`-sm tensor` on 4 GPUs vs 2 before committing.

## 4. Proposed architecture

```
                +--------------------+
                |    coordinator     |   llama-server (HTTP API, sampling,
                |  (may own GPUs 0,1)|   tokenizer, scheduler, spec decode)
                +---------+----------+
                          | RPC control channel (graph, activations)
          +---------------+----------------+
          |                                |
 +--------+---------+            +---------+--------+
 |     worker 1     |            |     worker 2     |
 | rpc-server       |            | rpc-server       |
 | GPUs 2,3 (TP via |            | GPUs 4,5 (TP via |
 | local NCCL)      |            | local NCCL)      |
 +------------------+            +------------------+

 pipeline split BETWEEN boxes, tensor split WITHIN each box
```

Key idea: make one RPC worker with K local GPUs look like ONE fast device to
the coordinator (a "TP island"), instead of K independent slow devices.

## 5. Implementation roadmap

### Phase 0 - works today, no code changes (baseline to measure)

- Coordinator: `llama-server --rpc worker1:50052,worker2:50052 -sm layer -ngl 99`
- Workers: `rpc-server -H 0.0.0.0 -p 50052 -c /cache` (one per host, all GPUs).
- Docker: `--network host` (RPC has no TLS; keep it on a private network or
  WireGuard overlay), `--gpus` per worker, model file only needed on the
  coordinator (weights stream once, then hash-cached on workers).
- Expected: correct, tolerable prefill, but generation latency gated by
  synchronous handoffs; no TP inside workers (each GPU is its own stage).

### Phase 1 - TP islands: meta backend inside the worker (medium effort)

Today `rpc-server` registers each GPU as a separate device. Add a worker flag
`--tensor-parallel` that wraps all local GPUs in a `ggml_backend_meta_device`
(same code path `-sm tensor` uses in-process) and exposes ONE device over RPC.
- Touchpoints: `tools/rpc/rpc-server.cpp` (device enumeration),
  `ggml/src/ggml-rpc/ggml-rpc.cpp` server init; meta backend is already a
  normal ggml device so most of it composes.
- Result: coordinator does `-sm layer` across workers; each worker internally
  runs NCCL TP across its GPUs. This is the hierarchical architecture with
  minimal new protocol work, and it is the highest-value phase.

### Phase 2 - async RPC + double-buffered pipeline (medium effort)

- Implement `set_tensor_async` / `graph_compute` overlap in the RPC client and
  a second in-flight command slot on the server, so stage N+1's input upload
  overlaps stage N's compute. `ggml-backend-sched` already double-buffers
  (`n_copies=4`) when backends report async capability; the RPC backend just
  needs to report and honor it (`ggml-rpc.cpp:1798-1803`).
- Add TCP_NODELAY + connection keepalive audit; optional RDMA where NICs allow.
- Result: multi-request throughput scales with number of workers (true
  pipelining), single-stream latency roughly unchanged.

### Phase 3 - cross-host tensor parallel, only if a fast fabric exists (large)

Only worth it with RDMA/RoCE or >= 25 GbE + kernel-bypass; otherwise skip.
- Replace `ncclCommInitAll` with `ncclCommInitRank` + bootstrap: coordinator
  generates `ncclUniqueId`, distributes it over the RPC control channel, each
  worker process joins the clique; NCCL then does allreduce worker-to-worker
  directly (NCCL supports sockets and IB transports natively).
- Touchpoints: `ggml-cuda.cu:1390-1460` (comm init), new RPC messages for
  bootstrap exchange, meta backend gains a "remote member" mode where
  subgraph dispatch goes over RPC but reductions go over NCCL.
- This also cleanly covers the "multi-container, single host" case (NCCL
  works across processes on one host via SHM/P2P transports, containers need
  `--ipc=host` and shared /dev/shm).

### Phase 4 - operational hardening (incremental, parallel to any phase)

- Worker health endpoint + coordinator-side reconnect/retry instead of
  `GGML_ABORT` (`ggml-rpc.cpp:30`); graceful degradation is model-fatal but
  should produce an HTTP 503, not a dead coordinator.
- Authentication token on the RPC handshake (HELLO message already exists,
  `ggml-rpc.cpp:330-347`) + optional TLS via the existing httplib/openssl dep.
- Docker compose reference: coordinator service + N worker services,
  healthchecks, `NCCL_SOCKET_IFNAME`, `--ipc=host` guidance, private overlay
  network. Worker image = current cuda image + `rpc-server` binary (already
  built; just not shipped in the server image target - add to
  `.devops/cuda.Dockerfile`).

## 6. What to measure at each phase (the numbers that matter)

- tg t/s single stream and pp t/s at 4k/32k, per phase, vs the single-process
  2-GPU baseline (tg 47.8, pp 1576 on this fork, build 2da668617).
- Per-token added latency of the RPC hop: run worker on same host over
  loopback first (isolates protocol cost from network cost).
- 4x V100 single host: `-sm tensor` 4-way vs 2 TP islands of 2 (`-sm layer`
  between islands) - the NVLink pair topology decides the winner.

## 7. Decision summary

| Scenario | Recommended mode | Needs code? |
|---|---|---|
| 2-4 GPUs, one host | single process, `-sm tensor` + NCCL | no |
| GPUs spread over hosts, 1-10 GbE | pipeline between hosts, TP inside each (Phase 1) | yes (Phase 1) |
| Multi-host + RDMA fabric | hierarchical or full TP (Phase 3) | yes (Phases 1-3) |
| Many independent requests, many hosts | N independent replicas + external LB (no llama.cpp changes; vLLM-style disaggregation is out of scope) | no |
