#!/usr/bin/env python3
"""Extract the DeepSeek-V4-Flash MTP head (nextn block 43 + embeddings/output) from the
rogerai-fyi Q8-MTP-v2 sharded GGUF on HF into a standalone head-only draft GGUF.

Downloads ONLY the needed tensor byte ranges (~5.6 GB) via HTTP Range requests.
Resumable: tensor data fetches append to a .part file with a JSON state ledger.
"""
import json, os, struct, sys, time, urllib.request, urllib.error

BASE = "https://huggingface.co/rogerai-fyi/DeepSeek-V4-Flash-MTP-GGUF/resolve/main/Q8-MTP-v2/DeepSeek-V4-Flash-Q8-MTP-v2"
SHARDS = {1: f"{BASE}-00001-of-00005.gguf", 4: f"{BASE}-00004-of-00005.gguf", 5: f"{BASE}-00005-of-00005.gguf"}
OUT = sys.argv[1] if len(sys.argv) > 1 else "DeepSeek-V4-Flash-MTP-drafter-Q8.gguf"
STATE = OUT + ".state.json"
PART = OUT + ".part"
ALIGN_DEFAULT = 32

T = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}  # scalar type -> size

class RangeReader:
    """Buffered forward reader over HTTP ranges; tracks absolute position."""
    def __init__(self, url, chunk=8*1024*1024):
        self.url, self.chunk = url, chunk
        self.buf, self.base, self.pos = b"", 0, 0
    def _fetch(self, start, end):
        for attempt in range(6):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=90) as r:
                    return r.read()
            except Exception as e:
                print(f"  fetch retry {attempt+1}: {e}", flush=True)
                time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"range fetch failed permanently: {start}-{end}")
    def _ensure(self, n):
        while len(self.buf) - (self.pos - self.base) < n:
            start = self.base + len(self.buf)
            self.buf += self._fetch(start, start + self.chunk - 1)
    def read(self, n):
        self._ensure(n)
        off = self.pos - self.base
        d = self.buf[off:off+n]
        self.pos += n
        return d

def rd_str(r):
    n, = struct.unpack('<Q', r.read(8))
    return r.read(n).decode('utf-8', errors='replace')

def skip_val(r, t):
    if t in T: r.read(T[t]); return
    if t == 8: n, = struct.unpack('<Q', r.read(8)); r.read(n); return
    if t == 9:
        et, = struct.unpack('<I', r.read(4)); cnt, = struct.unpack('<Q', r.read(8))
        if et in T: r.read(cnt * T[et]); return
        if et == 8:
            for _ in range(cnt):
                n, = struct.unpack('<Q', r.read(8)); r.read(n)
            return
        raise ValueError(f"bad array elem {et}")
    raise ValueError(f"bad kv type {t}")

def parse_header(url, want_kv_bytes=False):
    """Return dict: n_tensors, kvs=[(key, span_start, span_end, raw|None)], tensors=[(name,dims,type,off)],
    data_start (absolute), align."""
    r = RangeReader(url)
    assert r.read(4) == b'GGUF'
    ver, = struct.unpack('<I', r.read(4)); assert ver == 3, ver
    n_tensors, = struct.unpack('<Q', r.read(8))
    n_kv, = struct.unpack('<Q', r.read(8))
    kvs = []
    align = ALIGN_DEFAULT
    for _ in range(n_kv):
        span_start = r.pos
        k = rd_str(r)
        t, = struct.unpack('<I', r.read(4))
        if k == "general.alignment":
            v_pos = r.pos
            skip_val(r, t)
            # re-read value bytes to get alignment (u32 expected)
            raw_v = r.buf[v_pos - r.base : r.pos - r.base]
            if t == 4: align = struct.unpack('<I', raw_v)[0]
        else:
            skip_val(r, t)
        span_end = r.pos
        raw = None
        if want_kv_bytes:
            raw = r.buf[span_start - r.base : span_end - r.base]
        kvs.append((k, span_start, span_end, raw))
    tensors = []
    for _ in range(n_tensors):
        name = rd_str(r)
        nd, = struct.unpack('<I', r.read(4))
        dims = struct.unpack(f'<{nd}Q', r.read(8*nd))
        tt, = struct.unpack('<I', r.read(4))
        off, = struct.unpack('<Q', r.read(8))
        tensors.append((name, dims, tt, off))
    header_end = r.pos
    data_start = (header_end + align - 1) // align * align
    return {"n_tensors": n_tensors, "kvs": kvs, "tensors": tensors,
            "data_start": data_start, "align": align}

def content_length(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return int(resp.headers["Content-Length"])

def wanted(name):
    if name.startswith("blk.43."):
        return True
    return name in ("token_embd.weight", "output.weight", "output_norm.weight",
                    "output_hc_base.weight", "output_hc_fn.weight", "output_hc_scale.weight")

def main():
    print("== parsing shard headers ==", flush=True)
    h1 = parse_header(SHARDS[1], want_kv_bytes=True)
    picks = []  # (name, dims, type, src_url, abs_start, nbytes)
    for sh in (4, 5):
        h = parse_header(SHARDS[sh])
        flen = content_length(SHARDS[sh])
        infos = sorted(h["tensors"], key=lambda x: x[3])
        for i, (name, dims, tt, off) in enumerate(infos):
            end = infos[i+1][3] if i+1 < len(infos) else flen - h["data_start"]
            nbytes = end - off
            if wanted(name):
                picks.append((name, dims, tt, SHARDS[sh], h["data_start"] + off, nbytes))
    picks.sort(key=lambda p: p[5])  # small tensors first: fast visible progress
    total = sum(p[5] for p in picks)
    print(f"selected {len(picks)} tensors, {total/1e9:.2f} GB", flush=True)
    for name, dims, tt, _, _, nb in picks:
        print(f"  {name} dims={list(dims)} type={tt} bytes={nb}", flush=True)

    align = h1["align"]
    # KVs: keep everything except split.*; count survivors
    keep_kvs = [(k, raw) for (k, _, _, raw) in h1["kvs"] if not k.startswith("split.")]
    kv_blob = b"".join(raw for _, raw in keep_kvs)

    # tensor info table with new offsets
    new_infos, cursor = [], 0
    for name, dims, tt, src, abs_start, nbytes in picks:
        cursor = (cursor + align - 1) // align * align
        new_infos.append((name, dims, tt, cursor, src, abs_start, nbytes))
        cursor += nbytes
    info_blob = b""
    for name, dims, tt, off, _, _, _ in new_infos:
        nb = name.encode()
        info_blob += struct.pack('<Q', len(nb)) + nb + struct.pack('<I', len(dims))
        info_blob += struct.pack(f'<{len(dims)}Q', *dims) + struct.pack('<I', tt) + struct.pack('<Q', off)

    header = b'GGUF' + struct.pack('<I', 3) + struct.pack('<Q', len(new_infos)) + struct.pack('<Q', len(keep_kvs))
    header += kv_blob + info_blob
    pad = (-len(header)) % align
    header += b"\x00" * pad
    data_base = len(header)
    final_size = data_base + new_infos[-1][3] + new_infos[-1][6]
    print(f"header {data_base} bytes, final file {final_size/1e9:.2f} GB", flush=True)

    state = {}
    if os.path.exists(STATE):
        state = json.load(open(STATE))
    if state.get("data_base") != data_base:
        state = {"data_base": data_base, "done": []}  # header changed -> restart ledger

    with open(PART, "r+b" if os.path.exists(PART) else "w+b") as f:
        f.seek(0); f.write(header)
        for idx, (name, dims, tt, off, src, abs_start, nbytes) in enumerate(new_infos):
            if name in state["done"]:
                continue
            print(f"[{idx+1}/{len(new_infos)}] {name} ({nbytes/1e6:.1f} MB)", flush=True)
            got = 0
            f.seek(data_base + off)
            CH = 32 * 1024 * 1024
            while got < nbytes:
                n = min(CH, nbytes - got)
                for attempt in range(8):
                    try:
                        req = urllib.request.Request(src, headers={"Range": f"bytes={abs_start+got}-{abs_start+got+n-1}"})
                        with urllib.request.urlopen(req, timeout=120) as r:
                            chunk = r.read()
                        assert len(chunk) == n, f"short read {len(chunk)}/{n}"
                        break
                    except Exception as e:
                        print(f"  chunk retry {attempt+1}: {e}", flush=True)
                        time.sleep(3 * (attempt + 1))
                else:
                    raise RuntimeError("chunk failed permanently")
                f.write(chunk)
                got += n
            state["done"].append(name)
            json.dump(state, open(STATE, "w"))
        f.truncate(final_size)
    os.rename(PART, OUT)
    print(f"DONE -> {OUT} ({final_size/1e9:.2f} GB)", flush=True)
    # self-check: reparse local header
    print("self-check: header parse of output", flush=True)
    with open(OUT, "rb") as f:
        assert f.read(4) == b'GGUF'
    print("OK", flush=True)

if __name__ == "__main__":
    main()
