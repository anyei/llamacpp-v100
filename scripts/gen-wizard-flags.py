#!/usr/bin/env python3
"""Generate the wizard's FLAGS catalog from common/arg.cpp (TASKS #96).

Parses every add_opt() in common/arg.cpp, keeps the llama-server-applicable
options, marks fork-added flags by diffing against upstream's arg.cpp, and
embeds the catalog into tools/ui/static/wizard.html between the
FLAGS-CATALOG markers. Re-run after adding/changing any flag:

    python3 scripts/gen-wizard-flags.py             # regenerate + embed
    python3 scripts/gen-wizard-flags.py --check     # drift check (exit 1 on stale)

Fork detection uses `git show <ref>:common/arg.cpp` (default ref
upstream/master; override with GEN_WIZARD_UPSTREAM_REF). Without the ref the
catalog is still generated, with every flag conservatively marked upstream.

Exclusions are EXPLICIT and documented here - never silent (see EXCLUDE).
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "common" / "arg.cpp"
WIZARD = ROOT / "tools" / "ui" / "static" / "wizard.html"
BEGIN = "// FLAGS-CATALOG-BEGIN"
END = "// FLAGS-CATALOG-END"
UPSTREAM_REF = os.environ.get("GEN_WIZARD_UPSTREAM_REF", "upstream/master")

# flags deliberately NOT offered by the wizard, with the reason (TASKS #96:
# exclusions must be explicit). The launch path enforces the first two server-side.
EXCLUDE = {
    "--host": "router-owned child binding (rejected by /models/load)",
    "--port": "router-owned child binding (rejected by /models/load)",
    "--model": "the wizard owns the model pick",
    "--model-url": "the wizard owns the model pick",
    "--hf-repo": "the wizard owns the model pick",
    "--hf-repo-draft": "wizard spec panel owns draft model selection",
    "--hf-file": "the wizard owns the model pick",
    "--hf-repo-v": "the wizard owns the model pick",
    "--hf-file-v": "the wizard owns the model pick",
    "--hf-token": "credential - belongs in the router environment, not a launch",
    "--alias": "the router sets it from the preset name",
    "--help": "CLI-only",
    "--usage": "CLI-only",
    "--version": "CLI-only",
    "--completion-bash": "CLI-only",
    "--cache-list": "CLI-only",
    "--list-devices": "CLI-only",
    "--models-dir": "router-level (wizard sources drawer manages it)",
    "--models-preset": "router-level",
    "--models-max": "router-level",
    "--models-auto-unload": "router-level",
    "--api-key-file": "credential file - router deployment concern",
    "--ssl-key-file": "router deployment concern",
    "--ssl-cert-file": "router deployment concern",
    # user-approved matrix calls 2026-08-03 (research/wizard-flag-matrix.md addendum)
    "--fim-qwen-30b-default": "internet weight download conflicts with local-models-first wizard",
    "--fim-qwen-1.5b-default": "internet weight download conflicts with local-models-first wizard",
    "--fim-qwen-7b-spec": "internet weight download conflicts with local-models-first wizard",
    "--fim-qwen-14b-spec": "internet weight download conflicts with local-models-first wizard",
    "--fim-qwen-3b-default": "internet weight download conflicts with local-models-first wizard",
    "--fim-qwen-7b-default": "internet weight download conflicts with local-models-first wizard",
    "--gpt-oss-120b-default": "internet weight download conflicts with local-models-first wizard",
    "--gpt-oss-20b-default": "internet weight download conflicts with local-models-first wizard",
    "--embd-gemma-default": "internet weight download conflicts with local-models-first wizard",
    "--lookup-cache-dynamic": "lookup decoding unused on the fork; spec stack owns speculation",
    "--lookup-cache-static": "lookup decoding unused on the fork; spec stack owns speculation",
    "--reuse-port": "router owns port lifecycle",
    "--reverse-prompt": "interactive-mode flag, meaningless under the router",
    "--tts-use-guide-tokens": "no TTS vehicles in the zoo",
}

CATS = [
    (r'draft|spec-|speculative|mtp|ngram|lookup-cache', "Speculative decoding"),
    (r'samp|top-k|top-p|min-p|typical|--temp|mirostat|penal|--dry|xtc|sigma|logit-bias|grammar|json-schema|--seed|adaptive|dynatemp', "Sampling defaults"),
    (r'rope|yarn', "RoPE / YaRN"),
    (r'cache-type|kv-|checkpoint|cache-reuse|swa|defrag|cache-ram|cache-idle|context-shift|cache-prompt|slot-save|slot-prompt', "KV & prompt cache"),
    (r'rpc|fleet|tensor-split|split-mode|--device|gpu-layers|cpu-moe|override-tensor|main-gpu|ssd-stream|--fit|op-offload|repack', "Placement & fleet"),
    (r'batch|ubatch|--parallel|cont-batch|thread|--poll|--prio|numa|mlock|mmap|load-mode|direct-io|cpu-mask|cpu-range|cpu-strict|--warmup|kleidiai', "Performance & system"),
    (r'lora|control-vector', "Adapters"),
    (r'embedding|rerank|pool', "Embeddings & rerank"),
    (r'mmproj|--image|--audio|media-path', "Multimodal"),
    (r'chat-template|jinja|reasoning|--tool|think|prefill|--special|escape|antiprompt|reverse-prompt', "Chat & templates"),
    (r'api-key|ssl|cors|timeout|--slots|webui|--ui\b|props|metric|--log|verbos|endpoint|middleware|--path\b|sleep|heartbeat|api-prefix|--check\b|infill|mcp|sse-|reuse-port|no-host|--tags|--perf', "Server & ops"),
    (r'ctx-size|--predict|--keep|override-kv|check-tensors|--offline|swa-full|--chunks', "Model & context"),
]

def category(names, env, help_):
    hay = (" ".join(names) + " " + env + " " + help_[:90]).lower()
    for pat, c in CATS:
        if re.search(pat, hay):
            return c
    return "Other"


def balanced(text, start):
    depth = 0
    i = start
    in_str = in_line = in_block = False
    prev = ""
    while i < len(text):
        c = text[i]
        if in_line:
            if c == "\n":
                in_line = False
        elif in_block:
            if prev == "*" and c == "/":
                in_block = False
        elif in_str:
            if c == '"' and prev != "\\":
                in_str = False
            elif prev == "\\" and c == "\\":
                c = ""
        else:
            if c == '"':
                in_str = True
            elif c == "/" and text[i + 1:i + 2] == "/":
                in_line = True
            elif c == "/" and text[i + 1:i + 2] == "*":
                in_block = True
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    return i
        prev = c
        i += 1
    return -1


def strings_in(seg):
    return [m.group(1) for m in re.finditer(r'"((?:[^"\\]|\\.)*)"', seg)]


def parse_flags(text):
    out = []
    for m in re.finditer(r'\badd_opt\(', text):
        op = m.end() - 1
        end = balanced(text, op)
        if end < 0:
            continue
        body = text[op:end + 1]
        # chained setters run to the statement terminator
        semi = text.find(";", end)
        chain = text[end:semi if semi > 0 else end + 300]
        braces = re.findall(r'\{([^{}]*)\}', body[:1200])
        names, neg = [], []
        if braces:
            names = [s for s in strings_in(braces[0]) if s.startswith("-")]
            if len(braces) > 1:
                neg = [s for s in strings_in(braces[1]) if s.startswith("-")]
        if not names:
            continue
        after = body[body.find('}') + 1:]
        hint_m = re.match(r'\s*,\s*"([A-Z0-9_+\[\]\.<>|/ -]{1,40})"\s*,', after)
        hint = hint_m.group(1) if hint_m else ""
        lam = re.search(r'\[\]\(|\[\&\]\(', body)
        helpseg = body[:lam.start()] if lam else body
        drop = set(names + neg + ([hint] if hint else []))
        help_ = " ".join(h for h in strings_in(helpseg) if h not in drop)
        help_ = help_.replace("\\n", " ").replace("\\\"", "'").strip()
        envm = re.search(r'\.set_env\("([^"]+)"\)', chain)
        exm = re.search(r'\.set_examples\(\{([^}]*)\}\)', chain)
        examples = re.findall(r'LLAMA_EXAMPLE_(\w+)', exm.group(1)) if exm else []
        lamseg = body[lam.start():lam.start() + 200] if lam else ""
        if re.search(r'params\s*,\s*bool', lamseg):
            kind = "boolpair"
        elif re.search(r'params\s*,\s*int', lamseg):
            kind = "number"
        elif re.search(r'std::vector<std::string>', lamseg):
            kind = "list"
        elif re.search(r'const std::string', lamseg):
            kind = "text"
        elif re.search(r'params\s*\)', lamseg):
            kind = "switch"
        else:
            kind = "switch"
        out.append({
            "names": names, "neg": neg, "hint": hint, "kind": kind,
            "env": envm.group(1) if envm else "",
            "examples": examples, "help": help_,
        })
    return out


def control_of(f):
    if f["kind"] in ("switch", "boolpair"):
        return "check"
    if "|" in f["hint"] or re.search(r'\[[^]]*\|[^]]*\]', f["help"][:120]):
        return "select"
    if f["kind"] == "number" or re.match(r'^N\b|^<0|SECONDS|MS|SIZE|COUNT|PCT', f["hint"]):
        return "number"
    return "text"


def enum_vals(f):
    # best-effort option list for select controls, from the hint or help
    src = f["hint"] if "|" in f["hint"] else ""
    if not src:
        m = re.search(r'\[([^]]*\|[^]]*)\]', f["help"][:150])
        src = m.group(1) if m else ""
    vals = [v.strip() for v in src.strip("[]<>").split("|") if v.strip() and len(v.strip()) < 24]
    return vals[:8]


def upstream_names():
    try:
        text = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{UPSTREAM_REF}:common/arg.cpp"],
            capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError:
        print(f"WARN: cannot read {UPSTREAM_REF}:common/arg.cpp - marking everything upstream")
        return None
    names = set()
    for f in parse_flags(text):
        names.update(f["names"] + f["neg"])
    return names


def build_catalog():
    flags = parse_flags(SRC.read_text())
    up = upstream_names()
    cat = []
    seen = set()
    for f in flags:
        canonical = f["names"][-1]
        if canonical in seen:
            continue
        seen.add(canonical)
        if not (not f["examples"] or "COMMON" in f["examples"] or "SERVER" in f["examples"]):
            continue  # not a llama-server flag
        if any(n in EXCLUDE for n in f["names"]):
            continue  # documented exclusion
        entry = {
            "flag": canonical,
            "cat": category(f["names"] + f["neg"], f["env"], f["help"]),
            "ctl": control_of(f),
            "hint": f["hint"],
            "note": f["help"][:220],
        }
        if len(f["names"]) > 1:
            entry["aka"] = " ".join(n for n in f["names"] if n != canonical)
        if f["neg"]:
            entry["neg"] = f["neg"][-1]
        if f["env"]:
            entry["env"] = f["env"]
        if entry["ctl"] == "select":
            vals = enum_vals(f)
            if vals:
                entry["vals"] = vals
            else:
                entry["ctl"] = "text"
        if up is None or any(n not in up for n in [canonical]):
            if up is not None:
                entry["fork"] = True
        cat.append(entry)
    order = [c for _, c in CATS] + ["Other"]
    cat.sort(key=lambda e: (order.index(e["cat"]), e["flag"]))
    return cat


def main():
    check = "--check" in sys.argv
    catalog = build_catalog()
    if len(catalog) < 150:
        sys.exit(f"parsed only {len(catalog)} server flags - arg.cpp format changed?")
    payload = "const FLAGS = " + json.dumps(catalog, indent=None, separators=(",", ":")) + ";"
    block = f"{BEGIN} (generated by scripts/gen-wizard-flags.py from common/arg.cpp - do not hand-edit)\n{payload}\n{END}"
    html = WIZARD.read_text()
    if BEGIN not in html:
        sys.exit("wizard.html FLAGS-CATALOG markers not found")
    new_html = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), lambda _: block, html, flags=re.S)
    if check:
        if new_html != html:
            sys.exit(f"DRIFT: wizard.html FLAGS catalog is stale vs common/arg.cpp "
                     f"({len(catalog)} flags now) - run scripts/gen-wizard-flags.py")
        print(f"FLAGS catalog up to date ({len(catalog)} flags)")
        return
    WIZARD.write_text(new_html)
    n_fork = sum(1 for e in catalog if e.get("fork"))
    print(f"embedded {len(catalog)} flags ({n_fork} fork-added) into {WIZARD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
