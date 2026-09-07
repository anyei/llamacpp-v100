#!/usr/bin/env python3
"""Cut a prefix of transformer blocks out of a GGUF into a smaller standalone model.

A prefix is exact: block i only ever consumes the output of blocks < i, so the
kept blocks compute EXACTLY what they would in the full model for the same input
tokens. That makes the result a valid vehicle for anything that only reads
per-layer behaviour of the early layers - expert-selection profiling (TASKS #74),
placement plumbing, meta split gates - at a fraction of the size. The output head
is of course meaningless; do not read the generated text as quality.

Only tensor bytes are copied (no dequantisation), so any quant type works. The
source must be the shard that carries the wanted blocks AND the metadata (shard 1
of an unsloth split carries both; check with a shard map first).

Metadata is copied verbatim except: block_count is set to the kept count,
nextn_predict_layers is zeroed (the MTP block is the LAST block of the full model
and a prefix never includes it), and split.* keys are dropped so the result is a
standalone file.

  scripts/gguf-truncate.py src.gguf -n 6 -o out.gguf [--arch glm-dsa] [--dry-run]
"""
import argparse
import os
import re
import struct
import sys

SCALAR = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<?',
          10: '<Q', 11: '<q', 12: '<d'}
T_STRING, T_ARRAY, T_UINT32 = 8, 9, 4


class Reader:
    def __init__(self, path):
        self.f = open(path, 'rb')
        self.path = path

    def raw(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError(f"short read in {self.path}")
        return b

    def unpack(self, fmt):
        return struct.unpack(fmt, self.raw(struct.calcsize(fmt)))

    def string(self):
        (n,) = self.unpack('<Q')
        return self.raw(n).decode('utf-8', errors='replace')

    def value(self, t):
        if t == T_STRING:
            return self.string()
        if t == T_ARRAY:
            (et,), (n,) = self.unpack('<I'), self.unpack('<Q')
            return [self.value(et) for _ in range(n)]
        return self.unpack(SCALAR[t])[0]


def enc_string(s):
    b = s.encode('utf-8')
    return struct.pack('<Q', len(b)) + b


def read_gguf(path):
    r = Reader(path)
    magic, version, n_tensors, n_kv = r.unpack('<IIQQ')
    if magic != 0x46554747:
        sys.exit(f"{path}: not a GGUF file")
    kv = []  # (key, type, value, raw_span)
    for _ in range(n_kv):
        start = r.f.tell()
        key = r.string()
        (t,) = r.unpack('<I')
        val = r.value(t)
        end = r.f.tell()
        r.f.seek(start)
        span = r.raw(end - start)
        kv.append((key, t, val, span))
    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        (nd,) = r.unpack('<I')
        ne = [r.unpack('<Q')[0] for _ in range(nd)]
        (tt,) = r.unpack('<I')
        (off,) = r.unpack('<Q')
        tensors.append({'name': name, 'ne': ne, 'type': tt, 'off': off})
    align = next((v for k, _, v, _ in kv if k == 'general.alignment'), 32)
    pos = r.f.tell()
    data_start = (pos + align - 1) // align * align
    r.f.close()
    return version, kv, tensors, data_start, align


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src')
    ap.add_argument('-n', '--n-blocks', type=int, required=True,
                    help='number of leading blocks to keep (blk.0 .. blk.N-1)')
    ap.add_argument('-o', '--output', required=True)
    ap.add_argument('--arch', help='architecture prefix for the KV keys (default: read from the file)')
    ap.add_argument('--dry-run', action='store_true', help='report what would be written and exit')
    args = ap.parse_args()

    version, kv, tensors, data_start, align = read_gguf(args.src)
    arch = args.arch or next((v for k, _, v, _ in kv if k == 'general.architecture'), None)
    if arch is None:
        sys.exit("cannot determine architecture - pass --arch")

    # tensor byte lengths come from the on-disk layout, so no quant type table is
    # needed and any future type works unchanged
    src_size = os.path.getsize(args.src)
    by_off = sorted(tensors, key=lambda t: t['off'])
    for i, t in enumerate(by_off):
        end = by_off[i + 1]['off'] if i + 1 < len(by_off) else src_size - data_start
        t['nbytes'] = end - t['off']

    keep, dropped_blocks = [], set()
    for t in tensors:
        m = re.match(r'blk\.(\d+)\.', t['name'])
        if m is None:
            keep.append(t)
        elif int(m.group(1)) < args.n_blocks:
            keep.append(t)
        else:
            dropped_blocks.add(int(m.group(1)))
    kept_blocks = sorted({int(re.match(r'blk\.(\d+)\.', t['name']).group(1))
                          for t in keep if t['name'].startswith('blk.')})
    missing = [b for b in range(args.n_blocks) if b not in kept_blocks]
    if missing:
        sys.exit(f"source does not contain blocks {missing} - truncate from the shard that has them")

    # a block that straddles a shard boundary is present but PARTIAL, and the
    # resulting model dies at load with a missing-tensor error. Compare the last
    # kept block against its predecessor (same kind: the dense prefix is contiguous).
    if len(kept_blocks) >= 2:
        def suffixes(b):
            return {t['name'].split('.', 2)[2] for t in keep
                    if t['name'].startswith(f'blk.{b}.')}
        last, prev = kept_blocks[-1], kept_blocks[-2]
        lack = suffixes(prev) - suffixes(last)
        if lack:
            sys.exit(f"blk.{last} is INCOMPLETE in this source (missing {sorted(lack)[:3]}...) - "
                     f"it straddles a shard boundary. Use -n {last} to stop before it.")

    overrides = {f'{arch}.block_count': args.n_blocks}
    if any(k == f'{arch}.nextn_predict_layers' for k, _, _, _ in kv):
        overrides[f'{arch}.nextn_predict_layers'] = 0
    drop_keys = {'split.no', 'split.count', 'split.tensors.count'}

    total = sum(t['nbytes'] for t in keep)
    print(f"source     : {args.src}")
    all_blocks = set(kept_blocks) | dropped_blocks
    print(f"arch       : {arch}   blocks in file: {min(all_blocks)}..{max(all_blocks)}")
    print(f"keeping    : blk.0..blk.{args.n_blocks - 1} + {len([t for t in keep if not t['name'].startswith('blk.')])} non-block tensors")
    print(f"tensors    : {len(keep)} of {len(tensors)}")
    print(f"data bytes : {total / 1024**3:.2f} GiB")
    print(f"kv         : {len(kv) - len(drop_keys & {k for k, _, _, _ in kv})} of {len(kv)} "
          f"(overrides: {overrides})")
    if args.dry_run:
        return

    out_kv = [(k, t, v, span) for (k, t, v, span) in kv if k not in drop_keys]
    kv_blob = b''
    for key, t, val, span in out_kv:
        if key in overrides:
            if t != T_UINT32:
                sys.exit(f"{key} is type {t}, expected uint32 - refusing to guess an encoding")
            kv_blob += enc_string(key) + struct.pack('<I', t) + struct.pack('<I', overrides[key])
        else:
            kv_blob += span

    # tensor infos need fresh offsets; keep the source order for locality
    info_blob, run = b'', 0
    for t in keep:
        t['new_off'] = run
        info_blob += enc_string(t['name']) + struct.pack('<I', len(t['ne']))
        for d in t['ne']:
            info_blob += struct.pack('<Q', d)
        info_blob += struct.pack('<I', t['type']) + struct.pack('<Q', run)
        run += t['nbytes']

    header = struct.pack('<IIQQ', 0x46554747, version, len(keep), len(out_kv))
    pos = len(header) + len(kv_blob) + len(info_blob)
    pad = (align - pos % align) % align

    with open(args.output, 'wb') as out, open(args.src, 'rb') as src:
        out.write(header + kv_blob + info_blob + b'\x00' * pad)
        for i, t in enumerate(keep):
            src.seek(data_start + t['off'])
            left = t['nbytes']
            while left:
                chunk = src.read(min(left, 64 << 20))
                if not chunk:
                    sys.exit(f"unexpected EOF copying {t['name']}")
                out.write(chunk)
                left -= len(chunk)
            if (i + 1) % 50 == 0 or i + 1 == len(keep):
                print(f"  copied {i + 1}/{len(keep)} tensors", end='\r', flush=True)
    print()
    print(f"wrote {args.output} ({os.path.getsize(args.output) / 1024**3:.2f} GiB)")


if __name__ == '__main__':
    main()
