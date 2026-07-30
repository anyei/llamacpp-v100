#!/usr/bin/env python3
# Generate the launch wizard's environment-gate catalog (TASKS #78) from
# docs/env-gates.md and embed it into tools/ui/static/wizard.html between
# the GATES-CATALOG markers. Re-run after editing env-gates.md:
#
#   python3 scripts/gen-wizard-gates.py
#
# The catalog cannot drift from the doc: every gate table row becomes an
# entry; a gate missing from the class maps below still appears (class
# "diag", flagged unclassified) so new doc rows are never silently dropped.

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "env-gates.md"
WIZARD = ROOT / "tools" / "ui" / "static" / "wizard.html"

BEGIN = "// GATES-CATALOG-BEGIN"
END = "// GATES-CATALOG-END"

# Safety classes per TASKS #78:
#   serve  = safe to enable for serving
#   tune   = perf knob
#   diag   = instrumentation (instrumented legs are not baselines)
#   danger = never-serve / fault-injection (hard-confirm in the UI)
#   worker = worker-side (inert via the coordinator extra_env overlay)
CLASS = {
    "serve": [
        "GGML_META_BCAST_FUSE", "GGML_RPC_WIRE_F16", "GGML_RPC_WIRE_Q8",
        "GGML_META_EXPERT_DEFER", "GGML_META_EXPERT_DEFER_WAIT_US",
        "GGML_META_EXPERT_DEFER_SYNC_EDGE", "LLAMA_META_EP_ONLY",
        "LLAMA_META_ATTN_OWNER", "LLAMA_META_ALLOW_MULTI_LOCAL",
        "LLAMA_META_EXPERT_PLACEMENT", "LLAMA_META_LOCAL_DRAFT",
        "LLAMA_KV_WORKER_HOST", "LLAMA_SSD_STREAM_BUFFER",
        "LLAMA_SSD_STREAM_BUDGET", "LLAMA_SSD_STREAM_GPU",
        "LLAMA_SSD_STREAM_VRAM_BUDGET", "LLAMA_SSD_STREAM_SERIAL",
    ],
    "tune": [
        "LLAMA_SSD_STREAM_READ_THREADS", "LLAMA_SSD_STREAM_PREFETCH",
        "LLAMA_SSD_STREAM_SLRU", "LLAMA_SSD_STREAM_PROTECTED_PCT",
        "LLAMA_SSD_STREAM_NO_ODIRECT", "LLAMA_SSD_STREAM_GPU_NO_RECLAIM",
        "LLAMA_SSD_STREAM_VRAM_POOLS", "LLAMA_SSD_STREAM_GPU_SLRU",
        "LLAMA_SSD_STREAM_GPU_PROTECTED_PCT", "GGML_OP_OFFLOAD_MIN_BATCH",
        "LLAMA_SSD_STREAMING",
        "LLAMA_FLEET_CAPACITY_CHECK", "LLAMA_FLEET_KV_RESERVE_MB",
        "LLAMA_RPC_NO_SURGICAL", "LLAMA_RPC_AUTO_WEIGHT_RESERVE_MB",
        "GGML_RPC_NO_W2W", "GGML_RPC_NO_SRC_HINT", "GGML_META_MAX_GRAPHS",
        "GGML_META_SURGICAL_MAX_STATE_MIB", "GGML_META_FUSED_BCAST",
        "GGML_META_PARTIAL_MERGE", "LLAMA_LP_SYNC_EDGE",
        "GGML_CUDA_ALLREDUCE", "GGML_CUDA_AR_P2P_MAX_BYTES",
        "GGML_CUDA_AR_COPY_THRESHOLD", "GGML_CUDA_AR_COPY_CHUNK_BYTES",
        "GGML_CUDA_AR_BF16_THRESHOLD", "GGML_CUDA_DISABLE_GRAPHS",
        "GGML_CUDA_FORCE_GRAPHS", "LLAMA_ATTN_ROT_DISABLE",
        "LLAMA_DECODE_GRAPH_CACHE", "LLAMA_DECODE_GRAPH_CACHE_TOKENS",
        "LLAMA_SPEC_DRAFT_NO_PAD", "LLAMA_SPEC_ADAPTIVE",
    ],
    "diag": [
        "GGML_META_BOUNDARY_STATS", "GGML_META_TIMING", "GGML_META_DEBUG",
        "GGML_META_DEBUG_REDUCE", "GGML_META_NO_DELAY", "GGML_META_NO_STAR",
        "GGML_META_NO_FUSED", "GGML_RPC_STATS", "GGML_RPC_DEBUG",
        "LLAMA_EXPERT_PROFILE", "LLAMA_DECODE_TIMING", "LLAMA_BATCH_DEBUG",
        "LLAMA_DEBUG_DUMP_DIR", "LLAMA_DEBUG_DUMP_FILTER",
        "LLAMA_DSV4_COMPRESS_DEBUG", "GGML_SCHED_DEBUG",
        "GGML_CUDA_SYNC_NODES", "GGML_CUDA_CHECK_IDS",
        "GGML_CUDA_NO_CONCURRENT_STREAMS", "GGML_CUDA_MMID_NO_DST_ZERO",
        "GGML_SSD_STREAM_DEBUG", "LLAMA_SPEC_TIMING",
        "GGML_META_ZL_STATS", "GGML_META_ZL_DEBUG",
        "GGML_META_EXPERT_DEFER_VERIFY",
    ],
    "danger": [
        "GGML_META_PROBE_DEFER_GATHER", "GGML_META_EXPERT_DEFER_PREFILL",
        "LLAMA_TRUNC_ARR", "LLAMA_META_DUP_DEVICE",
        "GGML_CUDA_INJECT_COMPUTE_FAIL", "GGML_RPC_DEBUG_FAIL_ALLOC",
        "GGML_RPC_DEBUG_BUF", "GGML_CUDA_FA_MMA_FORCE_SMEM_FAIL",
        "LLAMA_LP_PAIRS",
    ],
    "worker": [
        "GGML_RPC_THREADPOOL_POLL", "GGML_RPC_CACHE_LIMIT_MIB",
        "GGML_RPC_SCORE", "GGML_RPC_ALLOW_SHUTDOWN", "GGML_RPC_TIMING",
        "GGML_CUDA_FA_NO_MMA", "GGML_CUDA_ERROR_CONTAIN",
        "GGML_RDMA_DEV", "GGML_RDMA_GID", "GGML_RPC_DEBUG_FUSED_DELAY_US",
    ],
}
NAME_CLASS = {n: c for c, names in CLASS.items() for n in names}

# gates whose danger lives in a specific VALUE, not in enabling them
DANGER_VALUES = {
    "GGML_META_EXPERT_DEFER": ["2"],  # v1 defer-all: coherence fail, never serve
}

# editor prefill when the generic rule below picks a poor enabling value
VAL_OVERRIDE = {
    "LLAMA_META_ATTN_OWNER": "0",
    "GGML_META_EXPERT_DEFER_WAIT_US": "3000",
    "GGML_META_BCAST_FUSE": "2",
    "LLAMA_SSD_STREAM_PREFETCH": "2",
    "LLAMA_SSD_STREAM_READ_THREADS": "4",
    "GGML_CUDA_ALLREDUCE": "p2p",
}

UNIT_TYPES = {"us", "mib", "count", "bytes", "width", "int"}


def strip_md(s):
    s = s.replace("\\|", "|")
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = s.replace("`", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def first_sentence(s, limit=170):
    # cut at the first sentence end that isn't inside parentheses
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "." and depth == 0 and (i + 1 == len(s) or s[i + 1] == " "):
            s = s[:i]
            break
    return s[: limit - 1] + "…" if len(s) > limit else s


def default_value(name, type_s, default_s):
    # what the value editor pre-fills when the gate is enabled: the value that
    # meaningfully turns the feature ON (a default that IS "off" is skipped)
    if name in VAL_OVERRIDE:
        return VAL_OVERRIDE[name]
    if "bool" in type_s:
        return "1"
    tl = type_s.lower()
    if "path" in tl or "str" in tl or ":" in type_s:
        return ""
    if "off" not in default_s.lower():
        m = re.search(r"-?\d+", default_s)
        if m and m.group(0) != "0":
            return m.group(0)
    m = re.search(r"[1-9]\d*", type_s)
    if m:
        return m.group(0)
    if re.fullmatch(r"[a-z0-9]+", tl) and tl not in UNIT_TYPES:
        return type_s
    return "1"


def parse(doc_text):
    gates = []
    cat = ""
    seen = set()
    for line in doc_text.splitlines():
        m = re.match(r"^##+\s+(.*)$", line)
        if m:
            cat = re.sub(r"^\d+b?\.\s*", "", m.group(1)).strip()
            cat = re.sub(r"\s*[-—(].*$", "", cat).strip()
            continue
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.replace("\\|", "\x00").split("|")[1:-1]]
        cells = [c.replace("\x00", "\\|") for c in cells]
        if len(cells) < 4:
            continue
        names = re.findall(r"`([A-Z][A-Z0-9_]+)`", cells[0])
        if not names:
            continue
        type_s, default_s = strip_md(cells[1]), strip_md(cells[2])
        desc = first_sentence(strip_md(cells[3]))
        for name in names:
            if name.startswith("LLAMA_ARG_"):  # CLI-alias envs: flags box, not gates
                continue
            if name in seen:
                continue
            seen.add(name)
            cls = NAME_CLASS.get(name)
            g = {
                "name": name,
                "cat": cat,
                "type": type_s,
                "default": default_s,
                "val": default_value(name, type_s, default_s),
                "cls": cls or "diag",
                "note": desc,
            }
            if cls is None:
                g["unclassified"] = True
            if name in DANGER_VALUES:
                g["dangerVals"] = DANGER_VALUES[name]
            gates.append(g)
    return gates


def main():
    gates = parse(DOC.read_text())
    if len(gates) < 60:
        sys.exit(f"parsed only {len(gates)} gates from {DOC} - table format changed?")
    uncls = [g["name"] for g in gates if g.get("unclassified")]
    if uncls:
        print(f"WARN: unclassified gates (add to CLASS in {Path(__file__).name}): {', '.join(uncls)}")

    payload = "const GATES = " + json.dumps(gates, indent=None, separators=(",", ":")) + ";"
    block = f"{BEGIN} (generated by scripts/gen-wizard-gates.py from docs/env-gates.md - do not hand-edit)\n{payload}\n{END}"

    html = WIZARD.read_text()
    if BEGIN in html:
        html = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), lambda _: block, html, flags=re.S)
    else:
        anchor = "// ---------- data layer: real router APIs ----------"
        if anchor not in html:
            sys.exit("wizard.html anchor not found")
        html = html.replace(anchor, block + "\n" + anchor)
    WIZARD.write_text(html)
    print(f"embedded {len(gates)} gates into {WIZARD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
