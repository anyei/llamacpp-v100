import struct, sys, collections

PATH = sys.argv[1]
f = open(PATH, "rb")

def rd(fmt):
    n = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read(n))

def rstr():
    (ln,) = rd("<Q")
    return f.read(ln).decode("utf-8", "replace")

GGUF_TYPES = {0:"<B",1:"<b",2:"<H",3:"<h",4:"<I",5:"<i",6:"<f",7:"<?",10:"<Q",11:"<q",12:"<d"}

def read_value(vt):
    if vt == 8:  # string
        return rstr()
    if vt == 9:  # array
        (et,) = rd("<I"); (cnt,) = rd("<Q")
        return [read_value(et) for _ in range(cnt)]
    fmt = GGUF_TYPES[vt]
    return rd(fmt)[0]

magic = f.read(4); (ver,) = rd("<I")
(n_tensors,) = rd("<Q"); (n_kv,) = rd("<Q")
print(f"magic={magic} ver={ver} n_tensors={n_tensors} n_kv={n_kv}")

kv = {}
for _ in range(n_kv):
    key = rstr(); (vt,) = rd("<I"); kv[key] = read_value(vt)

alignment = kv.get("general.alignment", 32)
# find arch-prefixed expert hparams
def find(suffix):
    for k,v in kv.items():
        if k.endswith(suffix):
            return k, v
    return None, None
print("arch:", kv.get("general.architecture"))
for s in ["block_count","expert_count","expert_used_count","expert_shared_count","embedding_length","expert_feed_forward_length"]:
    print(" ", *find("."+s))

# tensor infos
GGML_TYPE_SIZE = None  # we only need per-tensor byte size = nbytes; compute from dims+type via a small table
# block sizes/type sizes for the types we expect (name: (blck, type_size))
TS = {0:("F32",1,4),1:("F16",1,2),8:("Q8_0",32,34),
      # k-quants / iq — type_size per block of 256
      10:("Q2_K",256,84),11:("Q3_K",256,110),12:("Q4_K",256,144),13:("Q5_K",256,176),14:("Q6_K",256,210),
      16:("IQ2_XXS",256,66),17:("IQ2_XS",256,74),19:("IQ3_XXS",256,98),20:("IQ1_S",256,50),
      21:("IQ4_NL",32,18),23:("IQ3_S",256,110),24:("IQ2_S",256,82),25:("IQ4_XS",256,136)}

infos = []
for _ in range(n_tensors):
    name = rstr(); (nd,) = rd("<I")
    dims = [rd("<Q")[0] for _ in range(nd)]
    (typ,) = rd("<I"); (off,) = rd("<Q")
    ne = 1
    for d in dims: ne *= d
    tname, blck, tsz = TS.get(typ, (f"t{typ}", 256, 0))
    nbytes = (ne // blck) * tsz if blck and tsz else 0
    infos.append((name, dims, tname, off, nbytes))

# expert tensors
exps = [x for x in infos if "exps" in x[0]]
print(f"\nexpert tensors: {len(exps)}")
by_kind = collections.defaultdict(list)
for name, dims, tname, off, nbytes in exps:
    # blk.N.ffn_X_exps.weight -> kind = ffn_X_exps
    parts = name.split(".")
    kind = ".".join(p for p in parts if p not in ("weight",) and not p.isdigit())
    by_kind[kind].append((dims, tname, nbytes))
total_exp = 0
for kind, lst in by_kind.items():
    dims, tname, nbytes = lst[0]
    total_exp += sum(x[2] for x in lst)
    n_expert = dims[-1]
    slice_bytes = nbytes // n_expert
    print(f"  {kind}: {len(lst)} layers, dims={dims} type={tname} tensor={nbytes/1e6:.1f}MB "
          f"n_expert={n_expert} slice={slice_bytes/1e3:.1f}KB")
print(f"\ntotal expert bytes across model: {total_exp/1e9:.2f} GB")
nonexp = sum(x[4] for x in infos if "exps" not in x[0])
print(f"non-expert (resident) bytes: {nonexp/1e9:.2f} GB")
