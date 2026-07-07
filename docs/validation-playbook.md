# Validation Playbook

The test scenarios used to validate this fork's tensor-parallel, distributed,
and state-handling work, with the actual commands. Everything runs in the dev
container (`llamacpp-dev`: repo at `/src`, incremental build at `/build`,
models at `/models`) unless stated otherwise. Rebuild after source edits:

```bash
docker exec llamacpp-dev bash -c 'cd /src && cmake --build /build -j18'
```

Models used: `Qwen3.6-27B-UD-Q4_K_XL-MTP.gguf` (production-scale checks),
`Qwen3-0.6B-BF16.gguf` (fast iteration — loads in seconds, fits beside a
busy GPU). Deterministic request used everywhere below:

```bash
PROMPT='{"prompt":"Write a detailed explanation of how a B-tree works, including insertion and deletion.","n_predict":96,"temperature":0,"seed":42}'
```

House rule: **every meta-backend or RPC change is gated on byte-identical
temp-0 output** against a reference produced by the simplest equivalent
config. `diff` on saved outputs, never eyeballing.

---

## 1. In-process regression gate (the gate before any meta commit)

27B, 2 GPUs, tensor mode, MTP spec — the production configuration. Byte-exact
against the previous build's output.

```bash
docker exec -d llamacpp-dev /build/bin/llama-server \
  -m /models/Qwen3.6-27B-UD-Q4_K_XL-MTP.gguf \
  -sm tensor -ts 0.5,0.5 -fa on -fit off -c 32768 \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-min 1 --spec-draft-p-min 0.75 \
  --host 127.0.0.1 --port 8099
# wait for /health, then:
docker exec llamacpp-dev bash -c \
  "curl -sf http://127.0.0.1:8099/completion -d '$PROMPT' | python3 -c \
   'import json,sys; open(\"/tmp/text_new.txt\",\"w\").write(json.load(sys.stdin)[\"content\"])'"
docker exec llamacpp-dev diff /tmp/text_ref.txt /tmp/text_new.txt   # must be empty
```

Expected: byte-identical; tg ~64-66 t/s (MTP, short ctx). Any diff = stop.

## 2. Single-GPU reference (canonical output + canonical state bytes)

Produces the reference that TP/island/pipeline outputs are diffed against,
and the canonical slot-save file for state-integrity checks.

```bash
docker exec -d llamacpp-dev bash -c "cd /build/bin && CUDA_VISIBLE_DEVICES=1 \
  ./llama-server -m /models/Qwen3-0.6B-BF16.gguf -ngl 99 -c 4096 -np 1 \
  --slot-save-path /tmp/slotsave --host 127.0.0.1 --port 8100"
# generate (writes /tmp/smoke06_ref.txt), then:
curl -sf -X POST 'http://127.0.0.1:8100/slots/0?action=save' -d '{"filename":"ref.bin"}'
```

## 3. TP island smoke (0.6B — fast, runs beside a busy GPU)

Worker exposes its GPUs as ONE tensor-parallel device; coordinator has zero
local GPUs. Exercises split-state upload, sliced allocation, RPC get/set.

```bash
docker exec -d llamacpp-dev bash -c "cd /build/bin && \
  ./ggml-rpc-server --tensor-parallel -H 127.0.0.1 -p 50081 > /tmp/w.log 2>&1"
docker exec -d llamacpp-dev bash -c "cd /build/bin && CUDA_VISIBLE_DEVICES=\"\" \
  ./llama-server -m /models/Qwen3-0.6B-BF16.gguf --rpc 127.0.0.1:50081 -ngl 99 \
  -c 4096 -np 1 --slot-save-path /tmp/slotsave --host 127.0.0.1 --port 8099"
# generate; diff output vs the single-GPU reference text
```

Checks: worker prints `tensor-parallel island: N devices exposed as one`;
output coherent; worker log free of `out of buffer bounds` / `ABORT`.
Note: `--tensor-parallel` with one visible GPU silently degrades to a plain
single device — confirm the island line is present.

## 4. State integrity — the three-config byte comparison

The strongest test in this playbook; it caught two silent corruption bugs
(views lost over RPC; mirrored fallback stomping packed slices). Save the
*same logical state* from three configs and compare bytes:

```bash
# same prompt + n_predict + seed on: (a) island (test 3), (b) in-process
# -sm tensor -ts 0.5,0.5, (c) single GPU (test 2); then on each:
curl -sf -X POST 'http://127.0.0.1:PORT/slots/0?action=save' -d '{"filename":"NAME.bin"}'

docker exec llamacpp-dev python3 - <<'EOF'
a = open('/tmp/slotsave/island.bin','rb').read()
b = open('/tmp/slotsave/tp.bin','rb').read()
c = open('/tmp/slotsave/ref.bin','rb').read()
n = min(len(a), len(b))
print('island vs tp :', sum(x!=y for x,y in zip(a,b)), 'diff bytes of', n)
print('tp     vs ref:', sum(x!=y for x,y in zip(b,c)), 'diff bytes of', n)
EOF
```

Expected pattern (deviation = placement bug):
- island == in-process TP: **0 diff bytes**
- TP vs single-GPU: **~40% of bytes differ** — low-mantissa AllReduce
  ordering noise, NOT a bug (round trips are exact, text identical)

Then the round trip: `action=restore` the saved file, regenerate with the
same request, and diff against the pre-save generation — must be
byte-identical. (Before the fix, the island round trip produced `111111...`.)

## 5. 27B island end-to-end (needs both GPUs — pause prod first)

```bash
docker stop llama-tq-mtp-Qwen3.6-27B-UD-Q4_K_XL-MTP.gguf   # restart after!
docker exec -d llamacpp-dev bash -c "cd /build/bin && \
  ./ggml-rpc-server --tensor-parallel -H 127.0.0.1 -p 50080 -c > /tmp/w27.log 2>&1"
docker exec -d llamacpp-dev bash -c "cd /build/bin && CUDA_VISIBLE_DEVICES=\"\" \
  ./llama-server -m /models/Qwen3.6-27B-UD-Q4_K_XL-MTP.gguf --rpc 127.0.0.1:50080 \
  -ngl 99 -c 8192 -np 1 --slot-save-path /tmp/slotsave --host 127.0.0.1 --port 8098"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader   # after load
```

Checks and measured expectations:
- load 40-50 s with a warm worker cache (~40 min first time)
- **9.2 GB per GPU**, equal on both = genuinely sharded (a full copy per GPU
  means split states didn't upload — grep coordinator log for `uploading`)
- tg 32-33 t/s short ctx / 29.2 at 5.2k ctx; coherent output
- hybrid checkpoint: send a >2048-token prompt (multi-batch prefill forces a
  context checkpoint = 149.6 MiB state read over RPC — the historical crash
  site); then a 501 MB slot save/restore, post-restore generation
  byte-identical

Trap: send long prompts from a request FILE (`curl --data @/tmp/req.json`).
`docker exec CTR python3 - <<EOF` without `-i` silently runs an empty script.

## 6. Shadow-rotation / alias-lifetime stress

Adopted aliases must not outlive their shadows. Force ring rotations past
the depth (default 8) with shape-varied prompts, then restore a state saved
BEFORE the rotations:

```bash
# 1. generate + slot-save (baseline)
# 2. 12 prompts of different lengths, cache_prompt=false (each = new graph
#    shape = rebuild = ring rotation)
# 3. action=restore the old save; regenerate; diff vs the baseline generation
```

Expected: byte-identical post-restore; no aborts; save bytes unchanged from
before the stress.

## 7. Async RPC pipeline A/B (proto 4.2)

Two single-GPU workers = two endpoints (honest 2-box simulation on loopback);
zero-GPU coordinator with a layer split.

```bash
docker exec -d llamacpp-dev bash -c "cd /build/bin && CUDA_VISIBLE_DEVICES=0 \
  ./ggml-rpc-server -H 127.0.0.1 -p 50082 > /tmp/w0.log 2>&1"
docker exec -d llamacpp-dev bash -c "cd /build/bin && CUDA_VISIBLE_DEVICES=1 \
  ./ggml-rpc-server -H 127.0.0.1 -p 50083 > /tmp/w1.log 2>&1"
docker exec -d llamacpp-dev bash -c "cd /build/bin && CUDA_VISIBLE_DEVICES=\"\" \
  ./llama-server -m /models/Qwen3-0.6B-BF16.gguf --rpc 127.0.0.1:50082,127.0.0.1:50083 \
  -sm layer -ngl 99 -c 8192 -np 4 -lv 5 --host 127.0.0.1 --port 8103"
grep 'pipeline parallelism enabled' coordinator.log   # must appear (needs -lv 5)
```

A/B trick — pipeline parallelism OFF on the same binary: add a no-match
tensor override (`-ot xxxnomatchxxx=CPU`), which trips the
`has_tensor_overrides` gate. Compare single-stream tg, 4-concurrent
aggregate, and pp at `-ub 256` (a 2k-token prompt = 8 ubatches; at the
default `-ub 2048` the whole prompt is ONE ubatch and nothing can overlap).

Expected: outputs byte-identical (ON == OFF == single-GPU reference);
loopback throughput parity at 0.6B scale.

## 8. Worker-to-worker transfer A/B (proto 4.2)

Same topology as test 7. Control = `GGML_RPC_NO_W2W=1` on the coordinator
(forces the coordinator-bridged copy).

```bash
# run A: GGML_RPC_NO_W2W=1 ...llama-server...   (bridged)
# run B: plain                                   (fenced direct pulls)
# diff both outputs vs each other and vs the single-GPU reference
```

Verifying the pull path actually ran (it soft-falls back silently):
```bash
# a live established connection from worker2 to worker1's port, while generating:
W2PID=$(docker exec llamacpp-dev pgrep -f 'p 50087' | head -1)
docker exec llamacpp-dev python3 -c "
port=0xC3B6  # 50086 hex; adjust to worker1's port
print(sum(1 for l in open('/proc/$W2PID/net/tcp').readlines()[1:]
          if int(l.split()[2].split(':')[1],16)==50086 and l.split()[3]=='01'))"
# worker logs must NOT contain 'cannot reach source worker' / 'fenced pull ... failed'
```

Caveat: `/proc/PID/net/tcp` shows the whole network namespace — count
connections, don't attribute blindly (coordinator holds one to each worker).

## 9. Multi-connection worker + buffer reclaim

```bash
# start worker + coordinator (any of the above), then while generating:
docker exec llamacpp-dev /build/bin/llama-server --list-devices --rpc 127.0.0.1:50085
# repeat a few times: read-only probe connections must serve concurrently
# and generation must stay byte-identical

# reclaim: kill the coordinator, watch worker-side VRAM
nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 1
docker exec llamacpp-dev pkill -9 -f coordinator-name
sleep 4; nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 1
```

Measured on the 0.6B: 1709 MiB -> 361 MiB, worker keeps serving others.

## 10. Fail-fast on unreachable workers

```bash
docker exec llamacpp-dev bash -c "cd /build/bin && CUDA_VISIBLE_DEVICES= \
  ./llama-server -m /models/Qwen3-0.6B-BF16.gguf --rpc 127.0.0.1:50099 -ngl 99 2>&1 \
  | grep -m1 'failed to connect'"
```

Expected: clear error + exit. (Pre-4.2 this silently degraded to CPU-only —
if a "GPU" run is mysteriously slow, check which device the buffers landed
on in the load log.)

## 11. Storage baselines (task 15 — memory-capped runs)

```bash
# raw NVMe read ceiling:
dd if=/path/to/model.gguf of=/dev/null bs=1M count=4096 iflag=direct

# CPU-resident reference (fits in RAM):
docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES= -v $MODELS_DIR:/models:ro \
  llamacpp-local-v100:TAG -m /models/MODEL.gguf -ngl 0 -c 2048 -t 10 ...

# mmap-thrash case: EVICT THE MODEL FROM PAGE CACHE FIRST, then cap memory:
python3 -c "import os; fd=os.open('/path/model.gguf',os.O_RDONLY); \
            os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)"
docker run --rm --gpus all --memory 14g --memory-swap 14g ...
```

Two traps that produce invalid numbers:
- **cgroup caps don't charge pre-cached pages**: without the fadvise
  eviction, a capped container reads the host page cache for free and shows
  no thrash at all (RSS can exceed the cap for file-backed pages).
- the cap must exceed the process's *anonymous* floor (~11.3 GB for the 27B
  CPU config: KV + compute + output buffers) or the OOM killer fires during
  load; weights are the only reclaimable part.

Measured (27B Q4, this box): 1.6 GB/s NVMe; 1.51 t/s resident; 0.21 t/s
thrash @14 GB cap.

## 12. Production image verification

```bash
docker run --rm --gpus all --entrypoint /app/llama-server IMAGE:TAG --version
# revision must match the intended commit
docker run --rm --gpus all --entrypoint /app/ggml-rpc-server IMAGE:TAG --help \
  | grep tensor-parallel
```

---

## General traps collected along the way

- Reference outputs depend on the exact request: same prompt, `n_predict`,
  `temperature:0`, `seed`, and the same server flags. A cached prompt
  (`cache_prompt:true` + repeat request) returns 1-token timings — vary the
  prompt or restart between timing samples.
- Two coordinators must not share one worker *device* (stored-graph replays
  fail loudly by design). Sharing a worker with distinct devices is fine.
- Perf numbers taken while the production server or another GPU tenant is
  running are contended: valid for A/B on identical conditions, not as
  absolute baselines.
- Long-running background jobs: `pkill` patterns must be scoped (named via
  `exec -a NAME`) or a cleanup trap will kill a concurrent test's workers.
