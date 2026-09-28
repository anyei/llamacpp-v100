#include "mmsmt.cuh"
#include "dequantize.cuh"

#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <vector>

// Quantized weights x f32 activations for 2..32 tokens on Volta tensor cores.
//
// mma.sync.m8n8k4 is issued per quadpair (lanes {0-3,16-19}, {4-7,20-23}, ...): four independent
// 8x8x4 products per warp instruction. Quadpair q owns output rows q*8..q*8+7 of the CTA's 32 and
// every quadpair shares the same 8-token activation tile, so the warp tile is 8 tokens x 32 rows
// and one weight fragment feeds up to four token tiles (T <= 32). Each lane streams its own weight
// row global -> registers; the CTA's warps split K in 256-element super-blocks and reduce once
// through shared memory at the end. Weights enter the MMA as exact small integers in fp16 and the
// sub-block scale/min are applied to the fp32 partial sums, so the numerics are the MMVQ class
// with fp16 activations.

void ggml_cuda_mul_mat_cublas(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

#define MMSMT_COLS 32 // output rows per CTA (= mma N per warp; every warp decodes all 32 rows)
#define MMSMT_TILE  8 // tokens per activation tile (mma M)
#define MMSMT_NWARPS 4 // warps per CTA; they split the staged K range (docs/mmsmt-implementation.md 4.3)

// f32 [T][K] -> f16 [T][K]; rows >= nrows (tile padding) are written as zeros
static __global__ void mmsmt_prep_act(const float * __restrict__ x, const int64_t stride_x,
        half2 * __restrict__ x16, const int64_t ncols, const int64_t nrows) {
    const int64_t row = blockIdx.y;
    const int64_t k0  = ((int64_t) blockIdx.x*blockDim.x + threadIdx.x) * 16;
    if (k0 >= ncols) {
        return;
    }
    half2 * o = x16 + (row*ncols + k0)/2;
    if (row < nrows) {
        const float * xr = x + row*stride_x + k0;
#pragma unroll
        for (int j = 0; j < 16; j += 2) {
            o[j/2] = __floats2half2_rn(xr[j + 0], xr[j + 1]);
        }
    } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            o[j] = __floats2half2_rn(0.0f, 0.0f);
        }
    }
}

static __device__ __forceinline__ void mmsmt_mma(float (&d)[8], const uint32_t a0, const uint32_t a1, const uint32_t b0, const uint32_t b1) {
#if defined(VOLTA_MMA_AVAILABLE)
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, {%0, %1, %2, %3, %4, %5, %6, %7};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
#else
    GGML_UNUSED_VARS(d, a0, a1, b0, b1);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE)
}

// bytes (0,1) / (2,3) of w, each in [0,255], -> half2 {1024+b0, 1024+b1} raw bits
static __device__ __forceinline__ uint32_t mmsmt_bytes_lo(const uint32_t w) { return __byte_perm(w, 0x64006400u, 0x5150); }
static __device__ __forceinline__ uint32_t mmsmt_bytes_hi(const uint32_t w) { return __byte_perm(w, 0x64006400u, 0x5352); }

static __device__ __forceinline__ uint32_t mmsmt_h2sub(const uint32_t h, const uint32_t bias) {
    half2 a, b;
    *reinterpret_cast<uint32_t *>(&a) = h;
    *reinterpret_cast<uint32_t *>(&b) = bias;
    const half2 r = __hsub2(a, b);
    return *reinterpret_cast<const uint32_t *>(&r);
}

#define MMSMT_BIAS_1024 0x64006400u // half2 {1024, 1024}: unsigned nibbles/bytes ORed into the mantissa
#define MMSMT_BIAS_1152 0x64806480u // half2 {1152, 1152}: int8 after XOR 0x80

// NSL mma slices (4 k each) of one lane's weight row against the activation tiles at k0

// ---- MMA core -------------------------------------------------------------------------------------

// MMSMT_NACC independent accumulator chains per tile (slice s feeds chain s % MMSMT_NACC) so
// consecutive mma instructions do not serialize on one accumulator
#define MMSMT_NACC 2
template <int NTILES, int NSL>
static __device__ __forceinline__ void mmsmt_sub(float (&acc)[NTILES*MMSMT_NACC][8], const uint8_t * __restrict__ act,
        const int act_stride, const int r, const int64_t k0, const uint32_t (&bq)[NSL][2]) {
    static_assert(NSL % 2 == 0, "two slices per 16-byte activation load");
#pragma unroll
    for (int s = 0; s < NSL; s += 2) {
#pragma unroll
        for (int t = 0; t < NTILES; ++t) {
            const uint4 av = *reinterpret_cast<const uint4 *>(act + (t*MMSMT_TILE + r)*act_stride + (k0 + 4*s)*2);
            mmsmt_mma(acc[t*MMSMT_NACC + (s     % MMSMT_NACC)], av.x, av.y, bq[s + 0][0], bq[s + 0][1]);
            mmsmt_mma(acc[t*MMSMT_NACC + ((s + 1) % MMSMT_NACC)], av.z, av.w, bq[s + 1][0], bq[s + 1][1]);
        }
    }
}

// crun += d[col]*sum(chains) once per super-block. The m8n8k4 accumulator map gives this lane the
// four quadpair-local columns {c, c+1, c+4, c+5}, c = 2*((lane>>1)&1), while the lane streams column
// r only, so the super-block scale of each output column is gathered from its owner lane r == n
template <int NTILES>
static __device__ __forceinline__ void mmsmt_scale_acc(float (&crun)[NTILES][8], float (&cpart)[NTILES*MMSMT_NACC][8], const float d_own, const int lane) {
    const int c = 2*((lane >> 1) & 1);
    float d[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int n    = c + (j & 1) + 4*(j >> 1);
        const int srcl = (lane & 0xC) + (n & 3) + ((n & 4) << 2);
        d[j] = __shfl_sync(0xFFFFFFFF, d_own, srcl);
    }
#pragma unroll
    for (int t = 0; t < NTILES; ++t) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float v = 0.0f;
#pragma unroll
            for (int cc = 0; cc < MMSMT_NACC; ++cc) {
                v += cpart[t*MMSMT_NACC + cc][i];
                cpart[t*MMSMT_NACC + cc][i] = 0.0f;
            }
            crun[t][i] += d[(i & 1) | ((i >> 2) << 1)]*v;
        }
    }
}

// plain block types: sum the chains into crun (scales were folded into B)
template <int NTILES>
static __device__ __forceinline__ void mmsmt_fold(float (&crun)[NTILES][8], float (&cpart)[NTILES*MMSMT_NACC][8]) {
#pragma unroll
    for (int t = 0; t < NTILES; ++t) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
#pragma unroll
            for (int cc = 0; cc < MMSMT_NACC; ++cc) {
                crun[t][i] += cpart[t*MMSMT_NACC + cc][i];
                cpart[t*MMSMT_NACC + cc][i] = 0.0f;
            }
        }
    }
}

static __device__ __forceinline__ uint32_t mmsmt_h2(const float v) {
    const half2 h = __float2half2_rn(v);
    return *reinterpret_cast<const uint32_t *>(&h);
}

// (u - bias) * scale, all half2 bit patterns
static __device__ __forceinline__ uint32_t mmsmt_h2mul(const uint32_t u, const uint32_t bias, const uint32_t scale) {
    half2 a, b, c;
    *reinterpret_cast<uint32_t *>(&a) = u;
    *reinterpret_cast<uint32_t *>(&b) = bias;
    *reinterpret_cast<uint32_t *>(&c) = scale;
    const half2 r = __hmul2(__hsub2(a, b), c);
    return *reinterpret_cast<const uint32_t *>(&r);
}

// (u - bias) * scale - mr
static __device__ __forceinline__ uint32_t mmsmt_h2fma(const uint32_t u, const uint32_t bias, const uint32_t scale, const uint32_t mr) {
    half2 a, b, c, m;
    *reinterpret_cast<uint32_t *>(&a) = u;
    *reinterpret_cast<uint32_t *>(&b) = bias;
    *reinterpret_cast<uint32_t *>(&c) = scale;
    *reinterpret_cast<uint32_t *>(&m) = mr;
    const half2 r = __hfma2(__hsub2(a, b), c, __hneg2(m));
    return *reinterpret_cast<const uint32_t *>(&r);
}

// 4 quantized values in the byte lanes of u -> the two B registers of one mma slice
#define MMSMT_BQ_MUL(bq, s, u, bias, sc)      { (bq)[s][0] = mmsmt_h2mul(mmsmt_bytes_lo(u), bias, sc); (bq)[s][1] = mmsmt_h2mul(mmsmt_bytes_hi(u), bias, sc); }
#define MMSMT_BQ_FMA(bq, s, u, bias, sc, mr)  { (bq)[s][0] = mmsmt_h2fma(mmsmt_bytes_lo(u), bias, sc, mr); (bq)[s][1] = mmsmt_h2fma(mmsmt_bytes_hi(u), bias, sc, mr); }

#define MMSMT_BIAS_1025 0x64016401u // half2 {1025, 1025}: 2-bit code c -> c - 1
#define MMSMT_BIAS_1028 0x64046404u // half2 {1028, 1028}: 3-bit value - 4
#define MMSMT_BIAS_1032 0x64086408u // half2 {1032, 1032}: nibble - 8
#define MMSMT_BIAS_1056 0x64206420u // half2 {1056, 1056}: 6-bit value - 32

// ---- staged-row reads -----------------------------------------------------------------------------
//
// N consecutive 32-bit words of a staged row at byte pointer p. RW = 16/8/4 is the shared-memory load
// width; the row stride rule in mmsmt_layout keeps each width bank-conflict-free for the lanes that
// read together. PARITY formats (super-block size = 2 mod 4: Q6_K, Q3_K) keep the row's 2-byte parity:
// the 16-byte loads start at the 4-byte floor of p and the parity is funnel-shifted out (one extra
// word is read, inside the slot or the row's 16-byte pad).
template <int N, int RW, bool PARITY>
static __device__ __forceinline__ void mmsmt_load_words(const uint8_t * __restrict__ p, uint32_t (&w)[N]) {
    if (RW == 16) {
        constexpr int NV = (N + (PARITY ? 1 : 0) + 3)/4;
        uint32_t t[NV*4];
        const uint4 * q4 = reinterpret_cast<const uint4 *>(PARITY ? (p - (reinterpret_cast<uintptr_t>(p) & 2)) : p);
#pragma unroll
        for (int v = 0; v < NV; ++v) {
            const uint4 x = q4[v];
            t[4*v + 0] = x.x; t[4*v + 1] = x.y; t[4*v + 2] = x.z; t[4*v + 3] = x.w;
        }
        if (PARITY) {
            const uint32_t sh = (uint32_t) (reinterpret_cast<uintptr_t>(p) & 2)*8;
#pragma unroll
            for (int i = 0; i < N; ++i) {
                w[i] = __funnelshift_r(t[i], t[i + 1], sh);
            }
        } else {
#pragma unroll
            for (int i = 0; i < N; ++i) {
                w[i] = t[i];
            }
        }
    } else if (RW == 8) {
        constexpr int NV = (N + 1)/2;
        const uint2 * q2 = reinterpret_cast<const uint2 *>(p);
#pragma unroll
        for (int v = 0; v < NV; ++v) {
            const uint2 x = q2[v];
            w[2*v] = x.x;
            if (2*v + 1 < N) {
                w[2*v + 1] = x.y;
            }
        }
    } else {
        const uint32_t * q = reinterpret_cast<const uint32_t *>(p);
#pragma unroll
        for (int i = 0; i < N; ++i) {
            w[i] = q[i];
        }
    }
}

// byte i of a register word array (i constant after unrolling, so this folds to a shift/mask)
template <int N>
static __device__ __forceinline__ uint32_t mmsmt_byte(const uint32_t (&w)[N], const int i) {
    return (w[i >> 2] >> (8*(i & 3))) & 0xffu;
}
template <int N>
static __device__ __forceinline__ int mmsmt_sbyte(const uint32_t (&w)[N], const int i) {
    return ((int) (w[i >> 2] << (24 - 8*(i & 3)))) >> 24;
}

// Q4_K/Q5_K 6-bit scale + min j of the 12 packed scale bytes that start at byte 4 of the header words
template <int N>
static __device__ __forceinline__ void mmsmt_k4_scale_min(const uint32_t (&hw)[N], const int j, uint32_t & sc, uint32_t & m) {
    if (j < 4) {
        sc = mmsmt_byte(hw, 4 + j) & 63u;
        m  = mmsmt_byte(hw, 4 + j + 4) & 63u;
    } else {
        sc = (mmsmt_byte(hw, 4 + j + 4) & 0xFu) | ((mmsmt_byte(hw, 4 + j - 4) >> 6) << 4);
        m  = (mmsmt_byte(hw, 4 + j + 4) >> 4)   | ((mmsmt_byte(hw, 4 + j) >> 6) << 4);
    }
}

// Q3_K 6-bit scale `is` (0..15) of the 12 packed scale bytes in words sw[0..2]
template <int N>
static __device__ __forceinline__ int mmsmt_q3k_scale(const uint32_t (&sw)[N], const int is) {
    if (is < 4) {
        return (int) ((mmsmt_byte(sw, is) & 0xFu) | (((mmsmt_byte(sw, is + 8) >> 0) & 3u) << 4));
    }
    if (is < 8) {
        return (int) ((mmsmt_byte(sw, is) & 0xFu) | (((mmsmt_byte(sw, is + 4) >> 2) & 3u) << 4));
    }
    if (is < 12) {
        return (int) ((mmsmt_byte(sw, is - 8) >> 4) | (((mmsmt_byte(sw, is) >> 4) & 3u) << 4));
    }
    return (int) ((mmsmt_byte(sw, is - 8) >> 4) | (((mmsmt_byte(sw, is - 4) >> 6) & 3u) << 4));
}

// nibble word (4 values 0..15 in byte lanes) -> int8 table values of kvalues_iq4nl, in byte lanes
static __device__ __forceinline__ uint32_t mmsmt_iq4nl_lookup(const uint32_t u) {
    // kvalues_iq4nl = {-127,-104,-83,-65,-49,-35,-22,-10, 1,13,25,38,53,69,89,113} as little-endian bytes
    // __byte_perm takes one selector NIBBLE per output byte: gather the low 3 bits of each value into nibbles
    const uint32_t sel = (u & 0x7u) | ((u >> 4) & 0x70u) | ((u >> 8) & 0x700u) | ((u >> 12) & 0x7000u);
    const uint32_t lo  = __byte_perm(0xBFAD9881u, 0xF6EADDCFu, sel);
    const uint32_t hi  = __byte_perm(0x26190D01u, 0x71594535u, sel);
    const uint32_t m   = ((u >> 3) & 0x01010101u) * 0xffu;
    return (hi & m) | (lo & ~m);
}


// ---- per-format layouts ---------------------------------------------------------------------------
//
// A CTA step stages NSB consecutive super-blocks of its 32 rows: per row, one 16-byte-aligned SLOT per
// super-block holding the raw ggml block bytes (readers use the block's own offsets). GW = global load
// width (alignment of a super-block start in the weight buffer), RW = shared read/store width
// (alignment of every in-block offset a reader touches), PARITY = rows carry a 2-byte parity. The
// warps split the NSB*NCH chunk-units of a step evenly. Row stride (words): >= 16 B of pad past the
// last slot and, for the RW width, bank-conflict-free: RW 16 -> 4 mod 8, RW 8 -> 2 mod 4, RW 4 -> odd.

#define MMSMT_LPR 8                                    // lanes per row in the copy: 16-byte pieces, 128 B per row per instruction
#define MMSMT_RPI (WARP_SIZE/MMSMT_LPR)                // rows per warp instruction (4)
#define MMSMT_NRG (MMSMT_COLS/MMSMT_NWARPS/MMSMT_RPI)  // row groups a warp copies (2)

static constexpr __host__ __device__ int mmsmt_stride(const int w, const int rw) {
    return rw == 16 ? 8*((w - 4 + 7)/8) + 4 : rw == 8 ? 4*((w - 2 + 3)/4) + 2 : 2*(w/2) + 1;
}

template <ggml_type type> struct mmsmt_layout;

#define MMSMT_LAYOUT(T, SB_, SLOT_, NSB_, NCH_, GW_, RW_, PAR_)                                       \
template <> struct mmsmt_layout<T> {                                                                \
    static constexpr int  SB_BYTES = SB_, SLOT = SLOT_, NSB = NSB_, NCH = NCH_, GW = GW_, RW = RW_;  \
    static constexpr bool PARITY = PAR_;                                                            \
    static constexpr int  K_PER_CH = QK_K/NCH_;                                                     \
    static constexpr int  UNITS = NSB_*NCH_, U = UNITS/MMSMT_NWARPS;                                \
    static_assert(UNITS % MMSMT_NWARPS == 0, "chunk units must split evenly over the warps");       \
    static constexpr int  SEG_BYTES = SB_ + (PAR_ ? 2 : 0);   /* bytes copied per super-block row */ \
    static constexpr int  NPIECE = (SEG_BYTES + 15)/16;       /* 16-byte pieces per super-block row */ \
    static constexpr int  NJ = (NPIECE + MMSMT_LPR - 1)/MMSMT_LPR; /* copy instructions per row group */ \
    static constexpr int  NPF = NSB_*MMSMT_NRG*NJ*4;          /* prefetch words per lane */           \
    static constexpr int  SLOT_WORDS = SLOT_/4, ROW_WORDS = NSB_*SLOT_WORDS;                        \
    static constexpr int  STRIDE = mmsmt_stride(ROW_WORDS + 4, RW_);                                \
    static_assert(SLOT_ >= ((SEG_BYTES + RW_ - 1)/RW_)*RW_ && (RW_ < 16 || SLOT_ >= 16*NPIECE), "slot must hold the stored pieces"); \
    static_assert(SLOT_ % RW_ == 0 && (GW_ != 16 || SB_ % 16 == 0), "slot/segment alignment");      \
};
//            type            SB   SLOT NSB NCH GW  RW  parity
MMSMT_LAYOUT(GGML_TYPE_Q8_0,   272, 272, 1,  4,  16,  4, false)
MMSMT_LAYOUT(GGML_TYPE_Q4_0,   144, 144, 2,  2,  16,  8, false)
MMSMT_LAYOUT(GGML_TYPE_Q2_0,    72,  72, 4,  1,   8,  8, false)
MMSMT_LAYOUT(GGML_TYPE_Q4_K,   144, 144, 2,  2,  16, 16, false)
MMSMT_LAYOUT(GGML_TYPE_Q5_K,   176, 176, 2,  2,  16, 16, false)
MMSMT_LAYOUT(GGML_TYPE_Q6_K,   210, 224, 2,  2,   4, 16, true)
MMSMT_LAYOUT(GGML_TYPE_IQ4_XS, 136, 136, 2,  2,   8,  8, false)
MMSMT_LAYOUT(GGML_TYPE_Q3_K,   110, 112, 2,  2,   4, 16, true)
MMSMT_LAYOUT(GGML_TYPE_Q2_K,    84,  96, 2,  2,   4, 16, false)

// ---- readers: chunk<NTILES, C>(...) consumes chunk C of the super-block at hp, finish(...) closes it --
//
// hp = this lane's row slot (raw ggml block bytes, parity applied). Scales are folded into the fp16 B
// fragments; K-quants accumulate the super-block partial in cpart and finish() applies the per-column
// super-block scale. C is a template parameter so scale indices are constants (register decoding).

template <ggml_type type> struct mmsmt_reader;

#define MMSMT_READER_ARGS float (&crun)[NTILES][8], float (&cpart)[NTILES*MMSMT_NACC][8], const uint8_t * __restrict__ hp, \
        const uint8_t * __restrict__ act, const int act_stride, const int r, const int lane, const int64_t k0

template <> struct mmsmt_reader<GGML_TYPE_Q8_0> {
    using L = mmsmt_layout<GGML_TYPE_Q8_0>;
    // chunk C = two 34-byte blocks at 68C (17 words): block 0 has d in the low half of word 0 and its
    // quants 2 bytes in; block 1 has d in the high half of word 8 and its quants at words 9..16
    template <int NTILES, int C>
    static __device__ __forceinline__ void chunk(MMSMT_READER_ARGS) {
        GGML_UNUSED_VARS(crun, lane);
        uint32_t w[17];
        mmsmt_load_words<17, L::RW, L::PARITY>(hp + 68*C, w);
        uint32_t bq[8][2];
        const uint32_t d0 = mmsmt_h2(__half2float(__ushort_as_half((unsigned short) (w[0] & 0xffff))));
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            const uint32_t u = __funnelshift_r(w[s], w[s + 1], 16) ^ 0x80808080u;
            MMSMT_BQ_MUL(bq, s, u, MMSMT_BIAS_1152, d0);
        }
        mmsmt_sub<NTILES, 8>(cpart, act, act_stride, r, k0, bq);
        const uint32_t d1 = mmsmt_h2(__half2float(__ushort_as_half((unsigned short) (w[8] >> 16))));
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            const uint32_t u = w[9 + s] ^ 0x80808080u;
            MMSMT_BQ_MUL(bq, s, u, MMSMT_BIAS_1152, d1);
        }
        mmsmt_sub<NTILES, 8>(cpart, act, act_stride, r, k0 + 32, bq);
    }
    template <int NTILES>
    static __device__ __forceinline__ void finish(float (&crun)[NTILES][8], float (&cpart)[NTILES*MMSMT_NACC][8], const uint8_t * __restrict__ hp, const int lane) {
        GGML_UNUSED_VARS(hp, lane);
        mmsmt_fold<NTILES>(crun, cpart);
    }
};

// Q4_0 / Q2_0 share the 18-byte block shape: {half d; uint8 q[16]}; a chunk is four blocks (72 bytes)
template <int NTILES, bool TWOBIT>
static __device__ __forceinline__ void mmsmt_q40_q20_chunk(float (&cpart)[NTILES*MMSMT_NACC][8], const uint32_t (&w)[18],
        const uint8_t * __restrict__ act, const int act_stride, const int r, const int64_t k0) {
#pragma unroll
    for (int h = 0; h < 4; ++h) {
        uint32_t q[4];
        uint32_t d;
        const int b0 = 9*(h >> 1); // word of the block pair
        if ((h & 1) == 0) {
            d = mmsmt_h2(__half2float(__ushort_as_half((unsigned short) (w[b0] & 0xffff))));
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                q[i] = __funnelshift_r(w[b0 + i], w[b0 + i + 1], 16);
            }
        } else {
            d = mmsmt_h2(__half2float(__ushort_as_half((unsigned short) (w[b0 + 4] >> 16))));
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                q[i] = w[b0 + 5 + i];
            }
        }
        if (!TWOBIT) {
            // low nibbles k 0..15, high nibbles 16..31, value - 8
            uint32_t bq[8][2];
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                MMSMT_BQ_MUL(bq, s,     q[s] & 0x0f0f0f0fu,        MMSMT_BIAS_1032, d);
                MMSMT_BQ_MUL(bq, s + 4, (q[s] >> 4) & 0x0f0f0f0fu, MMSMT_BIAS_1032, d);
            }
            mmsmt_sub<NTILES, 8>(cpart, act, act_stride, r, k0 + 32*h, bq);
        } else {
            // 64 values per block: code c at byte i/4, bits 2*(i%4), value c - 1
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                uint32_t bq[8][2];
#pragma unroll
                for (int s = 0; s < 8; ++s) {
                    const uint32_t qb = (q[2*hh + s/4] >> (8*(s % 4))) & 0xffu;
                    const uint32_t u  = (qb & 0x3u) | ((qb & 0xCu) << 6) | ((qb & 0x30u) << 12) | ((qb & 0xC0u) << 18);
                    MMSMT_BQ_MUL(bq, s, u, MMSMT_BIAS_1025, d);
                }
                mmsmt_sub<NTILES, 8>(cpart, act, act_stride, r, k0 + 64*h + 32*hh, bq);
            }
        }
    }
}

template <> struct mmsmt_reader<GGML_TYPE_Q4_0> {
    using L = mmsmt_layout<GGML_TYPE_Q4_0>;
    template <int NTILES, int C>
    static __device__ __forceinline__ void chunk(MMSMT_READER_ARGS) {
        GGML_UNUSED_VARS(crun, lane);
        uint32_t w[18];
        mmsmt_load_words<18, L::RW, L::PARITY>(hp + 72*C, w);
        mmsmt_q40_q20_chunk<NTILES, false>(cpart, w, act, act_stride, r, k0);
    }
    template <int NTILES>
    static __device__ __forceinline__ void finish(float (&crun)[NTILES][8], float (&cpart)[NTILES*MMSMT_NACC][8], const uint8_t * __restrict__ hp, const int lane) {
        GGML_UNUSED_VARS(hp, lane);
        mmsmt_fold<NTILES>(crun, cpart);
    }
};

template <> struct mmsmt_reader<GGML_TYPE_Q2_0> {
    using L = mmsmt_layout<GGML_TYPE_Q2_0>;
    template <int NTILES, int C>
    static __device__ __forceinline__ void chunk(MMSMT_READER_ARGS) {
        GGML_UNUSED_VARS(crun, lane);
        uint32_t w[18];
        mmsmt_load_words<18, L::RW, L::PARITY>(hp + 72*C, w);
        mmsmt_q40_q20_chunk<NTILES, true>(cpart, w, act, act_stride, r, k0);
    }
    template <int NTILES>
    static __device__ __forceinline__ void finish(float (&crun)[NTILES][8], float (&cpart)[NTILES*MMSMT_NACC][8], const uint8_t * __restrict__ hp, const int lane) {
        GGML_UNUSED_VARS(hp, lane);
        mmsmt_fold<NTILES>(crun, cpart);
    }
};

template <> struct mmsmt_reader<GGML_TYPE_Q4_K> {
    using L = mmsmt_layout<GGML_TYPE_Q4_K>;
    // {half d, dmin; uint8 scales[12]; uint8 qs[128]}: chunk C = qs[64C..64C+63], byte j of chunk half il
    // holds k = 64*(2C+il) + j (low nibble) and k + 32 (high nibble)
    template <int NTILES, int C>
    static __device__ __forceinline__ void chunk(MMSMT_READER_ARGS) {
        GGML_UNUSED_VARS(crun, lane);
        uint32_t hw[4];
        mmsmt_load_words<4, L::RW, L::PARITY>(hp, hw);
        const float d    = __half2float(__ushort_as_half((unsigned short) (hw[0] & 0xffff)));
        const float dmin = __half2float(__ushort_as_half((unsigned short) (hw[0] >> 16)));
        const float mrat = d != 0.0f ? dmin/d : 0.0f;   // all-zero super-blocks have d == dmin == 0: no NaN
        uint32_t w[16];
        mmsmt_load_words<16, L::RW, L::PARITY>(hp + 16 + 64*C, w);
#pragma unroll
        for (int il = 0; il < 2; ++il) {
            const int isb = 2*(2*C + il);
            uint32_t sc, m;
            uint32_t bq[8][2];
            mmsmt_k4_scale_min(hw, isb + 0, sc, m);
            uint32_t sch = mmsmt_h2((float) sc);
            uint32_t mrh = mmsmt_h2(m*mrat);
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                MMSMT_BQ_FMA(bq, s, w[8*il + s] & 0x0f0f0f0fu, MMSMT_BIAS_1024, sch, mrh);
            }
            mmsmt_sub<NTILES, 8>(cpart, act, act_stride, r, k0 + 64*il, bq);
            mmsmt_k4_scale_min(hw, isb + 1, sc, m);
            sch = mmsmt_h2((float) sc);
            mrh = mmsmt_h2(m*mrat);
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                MMSMT_BQ_FMA(bq, s, (w[8*il + s] >> 4) & 0x0f0f0f0fu, MMSMT_BIAS_1024, sch, mrh);
            }
            mmsmt_sub<NTILES, 8>(cpart, act, act_stride, r, k0 + 64*il + 32, bq);
        }
    }
    template <int NTILES>
    static __device__ __forceinline__ void finish(float (&crun)[NTILES][8], float (&cpart)[NTILES*MMSMT_NACC][8], const uint8_t * __restrict__ hp, const int lane) {
        const float d = __half2float(*reinterpret_cast<const half *>(hp));
        mmsmt_scale_acc<NTILES>(crun, cpart, d, lane);
    }
};

template <> struct mmsmt_reader<GGML_TYPE_Q5_K> {
    using L = mmsmt_layout<GGML_TYPE_Q5_K>;
    // {half d, dmin; uint8 scales[12]; uint8 qh[32]; uint8 qs[128]}: chunk C = qs[64C..]; bit (2*il) / (2*il+1)
    // of qh[j] is the fifth bit of the low / high nibble of chunk il = 2C + il_local
    template <int NTILES, int C>
    static __device__ __forceinline__ void chunk(MMSMT_READER_ARGS) {
        GGML_UNUSED_VARS(crun, lane);
        uint32_t hw[4];
        mmsmt_load_words<4, L::RW, L::PARITY>(hp, hw);
        const float d    = __half2float(__ushort_as_half((unsigned short) (hw[0] & 0xffff)));
        const float dmin = __half2float(__ushort_as_half((unsigned short) (hw[0] >> 16)));
        const float mrat = d != 0.0f ? dmin/d : 0.0f;   // all-zero super-blocks have d == dmin == 0: no NaN
        uint32_t qh[8];
        mmsmt_load_words<8, L::RW, L::PARITY>(hp + 16, qh);
        uint32_t w[16];
        mmsmt_load_words<16, L::RW, L::PARITY>(hp + 48 + 64*C, w);
#pragma unroll
        for (int ill = 0; ill < 2; ++ill) {
            const int il = 2*C + ill;
            uint32_t sc, m;
            uint32_t bq[8][2];
            mmsmt_k4_scale_min(hw, 2*il + 0, sc, m);
            uint32_t sch = mmsmt_h2((float) sc);
            uint32_t mrh = mmsmt_h2(m*mrat);
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                const uint32_t u = (w[8*ill + s] & 0x0f0f0f0fu) | (((qh[s] >> (2*il)) & 0x01010101u) << 4);
                MMSMT_BQ_FMA(bq, s, u, MMSMT_BIAS_1024, sch, mrh);
            }
            mmsmt_sub<NTILES, 8>(cpart, act, act_stride, r, k0 + 64*ill, bq);
            mmsmt_k4_scale_min(hw, 2*il + 1, sc, m);
            sch = mmsmt_h2((float) sc);
            mrh = mmsmt_h2(m*mrat);
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                const uint32_t u = ((w[8*ill + s] >> 4) & 0x0f0f0f0fu) | (((qh[s] >> (2*il + 1)) & 0x01010101u) << 4);
                MMSMT_BQ_FMA(bq, s, u, MMSMT_BIAS_1024, sch, mrh);
            }
            mmsmt_sub<NTILES, 8>(cpart, act, act_stride, r, k0 + 64*ill + 32, bq);
        }
    }
    template <int NTILES>
    static __device__ __forceinline__ void finish(float (&crun)[NTILES][8], float (&cpart)[NTILES*MMSMT_NACC][8], const uint8_t * __restrict__ hp, const int lane) {
        const float d = __half2float(*reinterpret_cast<const half *>(hp));
        mmsmt_scale_acc<NTILES>(crun, cpart, d, lane);
    }
};

template <> struct mmsmt_reader<GGML_TYPE_Q6_K> {
    using L = mmsmt_layout<GGML_TYPE_Q6_K>;
    // {uint8 ql[128]; uint8 qh[64]; int8 scales[16]; half d}: chunk C = ip: ql[64C..] + qh[32C..];
    // k = 128*ip + 32*q + j takes nibble (q>>1) of ql[32*(q&1) + j] and bits 2q..2q+1 of qh[j]; value - 32
    template <int NTILES, int C>
    static __device__ __forceinline__ void chunk(MMSMT_READER_ARGS) {
        GGML_UNUSED_VARS(crun, lane);
        uint32_t scw[4];
        mmsmt_load_words<4, L::RW, L::PARITY>(hp + 192, scw);
        uint32_t ql[16];
        uint32_t qh[8];
        mmsmt_load_words<16, L::RW, L::PARITY>(hp + 64*C, ql);
        mmsmt_load_words<8, L::RW, L::PARITY>(hp + 128 + 32*C, qh);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            uint32_t bq[8][2];
            const uint32_t s0 = mmsmt_h2((float) mmsmt_sbyte(scw, 8*C + 2*q + 0));
            const uint32_t s1 = mmsmt_h2((float) mmsmt_sbyte(scw, 8*C + 2*q + 1));
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                const uint32_t u = ((ql[8*(q & 1) + s] >> (4*(q >> 1))) & 0x0f0f0f0fu) | (((qh[s] >> (2*q)) & 0x03030303u) << 4);
                MMSMT_BQ_MUL(bq, s, u, MMSMT_BIAS_1056, (s < 4) ? s0 : s1);
            }
            mmsmt_sub<NTILES, 8>(cpart, act, act_stride, r, k0 + 32*q, bq);
        }
    }
    template <int NTILES>
    static __device__ __forceinline__ void finish(float (&crun)[NTILES][8], float (&cpart)[NTILES*MMSMT_NACC][8], const uint8_t * __restrict__ hp, const int lane) {
        const float d = __half2float(*reinterpret_cast<const half *>(hp + 208));
        mmsmt_scale_acc<NTILES>(crun, cpart, d, lane);
    }
};

template <> struct mmsmt_reader<GGML_TYPE_IQ4_XS> {
    using L = mmsmt_layout<GGML_TYPE_IQ4_XS>;
    // {half d; uint16 scales_h; uint8 scales_l[4]; uint8 qs[128]}: chunk C = qs[64C..] = sub-blocks ib = 4C..4C+3:
    // bytes 16*ibl.. hold k = 32*ib + j (low nibble) and + 16 (high nibble), values through kvalues_iq4nl
    template <int NTILES, int C>
    static __device__ __forceinline__ void chunk(MMSMT_READER_ARGS) {
        GGML_UNUSED_VARS(crun, lane);
        uint32_t hw[2];
        mmsmt_load_words<2, L::RW, L::PARITY>(hp, hw);
        const uint32_t scales_h = hw[0] >> 16;
        const uint32_t scales_l = hw[1];
        uint32_t w[16];
        mmsmt_load_words<16, L::RW, L::PARITY>(hp + 8 + 64*C, w);
#pragma unroll
        for (int ibl = 0; ibl < 4; ++ibl) {
            const int ib = 4*C + ibl;
            const int ls = (int) (((scales_l >> (4*ib)) & 0xf) | (((scales_h >> (2*ib)) & 3) << 4)) - 32;
            const uint32_t lsh = mmsmt_h2((float) ls);
            uint32_t bq[8][2];
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                MMSMT_BQ_MUL(bq, s,     mmsmt_iq4nl_lookup(w[4*ibl + s] & 0x0f0f0f0fu) ^ 0x80808080u,        MMSMT_BIAS_1152, lsh);
                MMSMT_BQ_MUL(bq, s + 4, mmsmt_iq4nl_lookup((w[4*ibl + s] >> 4) & 0x0f0f0f0fu) ^ 0x80808080u, MMSMT_BIAS_1152, lsh);
            }
            mmsmt_sub<NTILES, 8>(cpart, act, act_stride, r, k0 + 32*ibl, bq);
        }
    }
    template <int NTILES>
    static __device__ __forceinline__ void finish(float (&crun)[NTILES][8], float (&cpart)[NTILES*MMSMT_NACC][8], const uint8_t * __restrict__ hp, const int lane) {
        const float d = __half2float(*reinterpret_cast<const half *>(hp));
        mmsmt_scale_acc<NTILES>(crun, cpart, d, lane);
    }
};

template <> struct mmsmt_reader<GGML_TYPE_Q3_K> {
    using L = mmsmt_layout<GGML_TYPE_Q3_K>;
    // {uint8 hmask[32]; uint8 qs[64]; uint8 scales[12]; half d}: chunk C = n: qs[32C..32C+31];
    // k = 128*n + 32*j + l takes bits 2j..2j+1 of qs[l] plus bit (4*n + j) of hmask[l] as bit 2, value - 4;
    // one 6-bit scale per 16 k
    template <int NTILES, int C>
    static __device__ __forceinline__ void chunk(MMSMT_READER_ARGS) {
        GGML_UNUSED_VARS(crun, lane);
        uint32_t hm[8];
        uint32_t scw[3];
        uint32_t qs[8];
        mmsmt_load_words<8, L::RW, L::PARITY>(hp, hm);
        mmsmt_load_words<3, L::RW, L::PARITY>(hp + 96, scw);
        mmsmt_load_words<8, L::RW, L::PARITY>(hp + 32 + 32*C, qs);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint32_t bq[8][2];
            const uint32_t s0 = mmsmt_h2((float) (mmsmt_q3k_scale(scw, 8*C + 2*j + 0) - 32));
            const uint32_t s1 = mmsmt_h2((float) (mmsmt_q3k_scale(scw, 8*C + 2*j + 1) - 32));
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                const uint32_t u = ((qs[s] >> (2*j)) & 0x03030303u) | (((hm[s] >> (4*C + j)) & 0x01010101u) << 2);
                MMSMT_BQ_MUL(bq, s, u, MMSMT_BIAS_1028, (s < 4) ? s0 : s1);
            }
            mmsmt_sub<NTILES, 8>(cpart, act, act_stride, r, k0 + 32*j, bq);
        }
    }
    template <int NTILES>
    static __device__ __forceinline__ void finish(float (&crun)[NTILES][8], float (&cpart)[NTILES*MMSMT_NACC][8], const uint8_t * __restrict__ hp, const int lane) {
        const float d = __half2float(*reinterpret_cast<const half *>(hp + 108));
        mmsmt_scale_acc<NTILES>(crun, cpart, d, lane);
    }
};

template <> struct mmsmt_reader<GGML_TYPE_Q2_K> {
    using L = mmsmt_layout<GGML_TYPE_Q2_K>;
    // {uint8 scales[16]; uint8 qs[64]; half d, dmin}: chunk C = n: qs[32C..]; k = 128*n + 32*jj + l
    // takes bits 2jj..2jj+1 of qs[l]; scales[is] = 4-bit scale | 4-bit min << 4 per 16 k, is = 8*n + 2*jj + l/16
    template <int NTILES, int C>
    static __device__ __forceinline__ void chunk(MMSMT_READER_ARGS) {
        GGML_UNUSED_VARS(crun, lane);
        uint32_t scw[4];
        mmsmt_load_words<4, L::RW, L::PARITY>(hp, scw);
        uint32_t dmw[1];
        mmsmt_load_words<1, L::RW, L::PARITY>(hp + 80, dmw);
        const float d    = __half2float(__ushort_as_half((unsigned short) (dmw[0] & 0xffff)));
        const float dmin = __half2float(__ushort_as_half((unsigned short) (dmw[0] >> 16)));
        const float mrat = d != 0.0f ? dmin/d : 0.0f;   // all-zero super-blocks have d == dmin == 0: no NaN
        uint32_t qs[8];
        mmsmt_load_words<8, L::RW, L::PARITY>(hp + 16 + 32*C, qs);
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            uint32_t bq[8][2];
            const int is0 = 8*C + 2*jj;
            const uint32_t b0  = mmsmt_byte(scw, is0);
            const uint32_t b1  = mmsmt_byte(scw, is0 + 1);
            const uint32_t sc0 = mmsmt_h2((float) (b0 & 0xFu));
            const uint32_t sc1 = mmsmt_h2((float) (b1 & 0xFu));
            const uint32_t mr0 = mmsmt_h2((b0 >> 4)*mrat);
            const uint32_t mr1 = mmsmt_h2((b1 >> 4)*mrat);
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                const uint32_t u = (qs[s] >> (2*jj)) & 0x03030303u;
                MMSMT_BQ_FMA(bq, s, u, MMSMT_BIAS_1024, (s < 4) ? sc0 : sc1, (s < 4) ? mr0 : mr1);
            }
            mmsmt_sub<NTILES, 8>(cpart, act, act_stride, r, k0 + 32*jj, bq);
        }
    }
    template <int NTILES>
    static __device__ __forceinline__ void finish(float (&crun)[NTILES][8], float (&cpart)[NTILES*MMSMT_NACC][8], const uint8_t * __restrict__ hp, const int lane) {
        const float d = __half2float(*reinterpret_cast<const half *>(hp + 80));
        mmsmt_scale_acc<NTILES>(crun, cpart, d, lane);
    }
};

// decode one chunk-unit (super-block at hp, chunk c) and close the super-block; c is warp-uniform
template <ggml_type type, int NTILES>
static __device__ __forceinline__ void mmsmt_decode_unit(float (&crun)[NTILES][8], float (&cpart)[NTILES*MMSMT_NACC][8],
        const uint8_t * __restrict__ hp, const int c, const uint8_t * __restrict__ act, const int act_stride,
        const int r, const int lane, const int64_t k0) {
    using L = mmsmt_layout<type>;
    using R = mmsmt_reader<type>;
    if (c == 0) {
        R::template chunk<NTILES, 0>(crun, cpart, hp, act, act_stride, r, lane, k0);
    } else if (L::NCH > 1 && c == 1) {
        R::template chunk<NTILES, 1>(crun, cpart, hp, act, act_stride, r, lane, k0);
    } else if (L::NCH > 2 && c == 2) {
        R::template chunk<NTILES, 2>(crun, cpart, hp, act, act_stride, r, lane, k0);
    } else if (L::NCH > 3) {
        R::template chunk<NTILES, 3>(crun, cpart, hp, act, act_stride, r, lane, k0);
    }
    R::template finish<NTILES>(crun, cpart, hp, lane);
}

// ---- kernel ---------------------------------------------------------------------------------------
//
// One CTA = 32 output rows, NWARPS warps. A step stages NSB consecutive super-blocks of all 32 rows
// (docs/mmsmt-implementation.md section 4): each warp copies 8 rows with 8 lanes per row and 16-byte
// pieces (128 contiguous bytes per row per instruction, the whole step is >= 272 contiguous bytes per
// row), keeps the NEXT step's pieces in registers while the current step computes, and decodes its
// share of the step's chunk-units (every lane decodes its own row from shared memory). The warps'
// partial sums over disjoint K are added through shared memory at the end; blockIdx.y splits K across
// CTAs into `part` (ksplit == 1 writes dst directly).

template <int GW>
static __device__ __forceinline__ void mmsmt_piece_ld(uint32_t * __restrict__ pf, const char * __restrict__ p, const int byte_off, const int seg_bytes) {
    // p = the piece's global address; only the GW-wide parts inside the segment are loaded
    if (GW == 16) {
        const uint4 v = __ldcs(reinterpret_cast<const uint4 *>(p));
        pf[0] = v.x; pf[1] = v.y; pf[2] = v.z; pf[3] = v.w;
    } else if (GW == 8) {
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            if (byte_off + 8*k < seg_bytes) {
                const uint2 v = __ldcs(reinterpret_cast<const uint2 *>(p) + k);
                pf[2*k] = v.x; pf[2*k + 1] = v.y;
            }
        }
    } else {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            if (byte_off + 4*k < seg_bytes) {
                pf[k] = __ldcs(reinterpret_cast<const unsigned int *>(p) + k);
            }
        }
    }
}
template <int RW>
static __device__ __forceinline__ void mmsmt_piece_st(uint32_t * __restrict__ s, const uint32_t * __restrict__ pf, const int byte_off, const int seg_bytes) {
    // RW-wide stores; a piece may end past the segment (inside the slot for RW 16, otherwise skipped)
    if (RW == 16) {
        *reinterpret_cast<uint4 *>(s) = make_uint4(pf[0], pf[1], pf[2], pf[3]);
    } else if (RW == 8) {
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            if (byte_off + 8*k < seg_bytes) {
                reinterpret_cast<uint2 *>(s)[k] = make_uint2(pf[2*k], pf[2*k + 1]);
            }
        }
    } else {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            if (byte_off + 4*k < seg_bytes) {
                s[k] = pf[k];
            }
        }
    }
}

template <ggml_type type, int NWARPS, int NTILES>
static __global__ void __launch_bounds__(NWARPS*WARP_SIZE, 4) mmsmt_kernel(
        const char * __restrict__ src0, const int64_t nb01, const half2 * __restrict__ x16,
        float * __restrict__ out, const int64_t ne00, const int64_t ne01, const int64_t ne11, const int sb_per_split, const int ablate) {
#if defined(VOLTA_MMA_AVAILABLE)
    using L = mmsmt_layout<type>;
    static_assert(NWARPS == MMSMT_NWARPS, "the layouts split chunk-units over MMSMT_NWARPS warps");
    static_assert((NWARPS - 1)*WARP_SIZE*NTILES*8 <= MMSMT_COLS*L::STRIDE, "stage buffer too small for the cross-warp reduce");
    __shared__ __align__(16) uint32_t stage[MMSMT_COLS*L::STRIDE];
    // per-warp activation slices: U units x NTILES*8 token rows x K_PER_CH fp16, row stride + 16 B
    // keeps the 8 rows of an mma slice-pair on distinct bank groups (docs/mmsmt-implementation.md 4.5)
    constexpr int ACT_ROW  = L::K_PER_CH*2 + 16;
    constexpr int ACT_UNIT = NTILES*MMSMT_TILE*ACT_ROW;
    constexpr int APL      = NTILES*L::K_PER_CH/32;   // 16-byte activation pieces per lane per unit
    static_assert(APL*32 == NTILES*MMSMT_TILE*(L::K_PER_CH/8), "activation pieces must split evenly over the lanes");
    __shared__ __align__(16) uint8_t act[NWARPS][L::U*ACT_UNIT];

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int qp   = (lane >> 2) & 3;                      // quadpair
    const int r    = (lane & 3) + ((lane & 16) ? 4 : 0);   // token row of the A fragment and row-in-quadpair of B
    const int lrow = qp*8 + r;                             // this lane's decode row within the CTA

    const int64_t colw = (int64_t) blockIdx.x*MMSMT_COLS;
    const int64_t col0 = colw + qp*8;
    const int64_t col  = col0 + r;
    const bool    good = col < ne01;

    // copy roles: 8 lanes per row, 4 rows per instruction; warp w copies rows 8w + 4g + lane/8
    const int lane8 = lane % MMSMT_LPR;
    const int rq    = lane / MMSMT_LPR;
    const char * rowbase[MMSMT_NRG];
    int          rpar[MMSMT_NRG];
#pragma unroll
    for (int g = 0; g < MMSMT_NRG; ++g) {
        int64_t row = colw + warp*(MMSMT_COLS/NWARPS) + g*MMSMT_RPI + rq;
        row = row < ne01 ? row : 0;
        rowbase[g] = src0 + row*nb01;
        rpar[g]    = (int) ((row*nb01) & 2);
    }
    uint32_t * sbase = stage + (warp*(MMSMT_COLS/NWARPS) + rq)*L::STRIDE + 4*lane8;

    const int nsb = (int) (ne00/QK_K);
    const int sb0 = blockIdx.y*sb_per_split;
    const int sb1 = min(nsb, sb0 + sb_per_split);

    float crun[NTILES][8];
    float cpart[NTILES*MMSMT_NACC][8];
#pragma unroll
    for (int t = 0; t < NTILES; ++t) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            crun[t][i] = 0.0f;
        }
    }
#pragma unroll
    for (int t = 0; t < NTILES*MMSMT_NACC; ++t) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            cpart[t][i] = 0.0f;
        }
    }

    uint32_t pf[L::NPF];
    uint4    apf[L::U*APL];
    uint8_t * act_w = act[warp];

    // this warp's activation slices for the units of the step at sb: token row / k-piece of piece p
    auto load_act = [&](const int sb) {
#pragma unroll
        for (int i = 0; i < L::U; ++i) {
            const int u   = warp*L::U + i;
            const int sbl = u / L::NCH;
            const int c   = u % L::NCH;
            if (sb + sbl < sb1) {
                const int64_t k0u = (int64_t) (sb + sbl)*QK_K + c*L::K_PER_CH;
#pragma unroll
                for (int j = 0; j < APL; ++j) {
                    const int p   = lane + WARP_SIZE*j;
                    const int row = p / (L::K_PER_CH/8);
                    const int kp  = p % (L::K_PER_CH/8);
                    apf[i*APL + j] = *reinterpret_cast<const uint4 *>(x16 + row*(ne00/2) + (k0u + 8*kp)/2);
                }
            }
        }
    };
    auto store_act = [&]() {
#pragma unroll
        for (int i = 0; i < L::U; ++i) {
#pragma unroll
            for (int j = 0; j < APL; ++j) {
                const int p   = lane + WARP_SIZE*j;
                const int row = p / (L::K_PER_CH/8);
                const int kp  = p % (L::K_PER_CH/8);
                *reinterpret_cast<uint4 *>(act_w + i*ACT_UNIT + row*ACT_ROW + 16*kp) = apf[i*APL + j];
            }
        }
    };

    auto load_tile = [&](const int sb) {
#pragma unroll
        for (int seg = 0; seg < L::NSB; ++seg) {
            if (sb + seg < sb1) {
                const int sbo = (sb + seg)*L::SB_BYTES;
#pragma unroll
                for (int g = 0; g < MMSMT_NRG; ++g) {
                    const int off = L::PARITY ? sbo - ((rpar[g] ^ sbo) & 2) : sbo;   // 4-byte floor for parity rows
                    const char * p = rowbase[g] + off + 16*lane8;
#pragma unroll
                    for (int j = 0; j < L::NJ; ++j) {
                        const int piece = lane8 + MMSMT_LPR*j;
                        if (piece < L::NPIECE) {
                            mmsmt_piece_ld<L::GW>(pf + ((seg*MMSMT_NRG + g)*L::NJ + j)*4, p + 128*j, 16*piece, L::SEG_BYTES);
                        }
                    }
                }
            }
        }
    };
    auto store_tile = [&]() {
#pragma unroll
        for (int seg = 0; seg < L::NSB; ++seg) {
#pragma unroll
            for (int g = 0; g < MMSMT_NRG; ++g) {
#pragma unroll
                for (int j = 0; j < L::NJ; ++j) {
                    const int piece = lane8 + MMSMT_LPR*j;
                    if (piece < L::NPIECE) {
                        mmsmt_piece_st<L::RW>(sbase + g*MMSMT_RPI*L::STRIDE + seg*L::SLOT_WORDS + 32*j, pf + ((seg*MMSMT_NRG + g)*L::NJ + j)*4, 16*piece, L::SEG_BYTES);
                    }
                }
            }
        }
    };

    if (sb0 < sb1) {
        load_tile(sb0);
        load_act(sb0);
    }
    for (int sb = sb0; sb < sb1; sb += L::NSB) {
        __syncthreads();                 // every warp is done reading the previous step
        // dev ablations (docs/mmsmt-implementation.md section 5): 1 = copy only, 2 = compute only
        if (ablate != 2 || sb == sb0) {
            store_tile();
        }
        store_act();
        __syncthreads();                 // the step is visible
        if (ablate != 2 && sb + L::NSB < sb1) {
            load_tile(sb + L::NSB);
        }
        if (ablate != 1) {
            const int nsb_step = min(L::NSB, sb1 - sb);
#pragma unroll
            for (int i = 0; i < L::U; ++i) {
                const int u   = warp*L::U + i;
                const int sbl = u / L::NCH;
                const int c   = u % L::NCH;
                if (sbl < nsb_step) {
                    const int par = L::PARITY ? (int) (((good ? col : 0)*nb01 + (int64_t) (sb + sbl)*L::SB_BYTES) & 2) : 0;
                    const uint8_t * hp = reinterpret_cast<const uint8_t *>(stage + lrow*L::STRIDE + sbl*L::SLOT_WORDS) + par;
                    mmsmt_decode_unit<type, NTILES>(crun, cpart, hp, c, act_w + i*ACT_UNIT, ACT_ROW, r, lane, 0);
                }
            }
        }
        __syncwarp();                    // this warp's activation reads are done: refill its private slice
        if (sb + L::NSB < sb1) {
            load_act(sb + L::NSB);
        }
    }

    // cross-warp reduce: warps 1.. park their partials in the (now free) stage buffer, warp 0 sums in order
    __syncthreads();
    float * red = reinterpret_cast<float *>(stage);
    if (warp > 0) {
#pragma unroll
        for (int t = 0; t < NTILES; ++t) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                red[((warp - 1)*WARP_SIZE + lane)*(NTILES*8) + t*8 + i] = crun[t][i];
            }
        }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int w = 1; w < NWARPS; ++w) {
#pragma unroll
            for (int t = 0; t < NTILES; ++t) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    crun[t][i] += red[((w - 1)*WARP_SIZE + lane)*(NTILES*8) + t*8 + i];
                }
            }
        }
        // C fragment map of m8n8k4: register i of this lane holds token row (i&2)|((lane&16)?4:0)|(lane&1)
        // and quadpair-local column (i&1)|(((lane>>1)&1)<<1)|((i>>2)<<2)
        const int64_t tpad = (int64_t) NTILES*MMSMT_TILE;
        float * p = out + blockIdx.y*tpad*ne01;
#pragma unroll
        for (int t = 0; t < NTILES; ++t) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int     row  = t*MMSMT_TILE + ((i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1));
                const int64_t ocol = col0 + ((i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
                if (row < ne11 && ocol < ne01) {
                    p[row*ne01 + ocol] = crun[t][i];
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(src0, nb01, x16, out, ne00, ne01, ne11, sb_per_split, ablate);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE)
}

static __global__ void mmsmt_reduce(const float * __restrict__ part, float * __restrict__ dst, const int64_t n, const int64_t stride_split, const int ksplit) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float acc = 0.0f;
    for (int k = 0; k < ksplit; ++k) {
        acc += part[k*stride_split + i];
    }
    dst[i] = acc;
}

template <ggml_type type, int NWARPS>
static void mmsmt_launch_tiles(const int ntiles, const dim3 grid, cudaStream_t stream,
        const char * src0, const int64_t nb01, const half2 * x16, float * out,
        const int64_t ne00, const int64_t ne01, const int64_t ne11, const int sb_per_split, const int ablate) {
    const dim3 block(NWARPS*WARP_SIZE);
    switch (ntiles) {
        case 1: mmsmt_kernel<type, NWARPS, 1><<<grid, block, 0, stream>>>(src0, nb01, x16, out, ne00, ne01, ne11, sb_per_split, ablate); break;
        case 2: mmsmt_kernel<type, NWARPS, 2><<<grid, block, 0, stream>>>(src0, nb01, x16, out, ne00, ne01, ne11, sb_per_split, ablate); break;
        default: GGML_ABORT("mmsmt: bad tile count");
    }
}

template <ggml_type type>
static void mmsmt_launch(const int ntiles, const dim3 grid, cudaStream_t stream,
        const char * src0, const int64_t nb01, const half2 * x16, float * out,
        const int64_t ne00, const int64_t ne01, const int64_t ne11, const int sb_per_split, const int ablate) {
    mmsmt_launch_tiles<type, MMSMT_NWARPS>(ntiles, grid, stream, src0, nb01, x16, out, ne00, ne01, ne11, sb_per_split, ablate);
}

static int mmsmt_nsb(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q8_0:   return mmsmt_layout<GGML_TYPE_Q8_0>::NSB;
        case GGML_TYPE_Q4_0:   return mmsmt_layout<GGML_TYPE_Q4_0>::NSB;
        case GGML_TYPE_Q2_0:   return mmsmt_layout<GGML_TYPE_Q2_0>::NSB;
        case GGML_TYPE_Q4_K:   return mmsmt_layout<GGML_TYPE_Q4_K>::NSB;
        case GGML_TYPE_Q5_K:   return mmsmt_layout<GGML_TYPE_Q5_K>::NSB;
        case GGML_TYPE_Q6_K:   return mmsmt_layout<GGML_TYPE_Q6_K>::NSB;
        case GGML_TYPE_IQ4_XS: return mmsmt_layout<GGML_TYPE_IQ4_XS>::NSB;
        case GGML_TYPE_Q3_K:   return mmsmt_layout<GGML_TYPE_Q3_K>::NSB;
        case GGML_TYPE_Q2_K:   return mmsmt_layout<GGML_TYPE_Q2_K>::NSB;
        default: return 1;
    }
}

static bool mmsmt_type_supported(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q2_0:
            return true;
        default:
            return false;
    }
}

bool ggml_cuda_should_use_mmsmt(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, const int cc) {
    static const bool enabled = [] {
        // opt-in while the kernel is at parity with MMVQ/MMQ (TASKS #153 T1, docs/volta-smallt-gemm-plan.md)
        const char * e = getenv("GGML_CUDA_SMT");
        return e != nullptr && atoi(e) != 0;
    }();
    if (!enabled || !volta_mma_available(cc) || !mmsmt_type_supported(src0->type)) {
        return false;
    }
    // per-type minimum width (docs/mmsmt-implementation.md 4.6): MMVQ stays ahead below it. Measured on
    // the X99 V100: Q4_K wins from width 3, Q8_0 from width 6 (MMVQ Q8_0 streams ~715 GB/s at width 2).
    // Other K-quants / IQ4_XS decode at Q4_K cost or more (same crossover assumed); Q4_0 / Q2_0 are
    // cheap decodes with a strong MMVQ like Q8_0. GGML_CUDA_SMT_MIN overrides for experiments.
    static const int min_env = getenv("GGML_CUDA_SMT_MIN") ? atoi(getenv("GGML_CUDA_SMT_MIN")) : 0;
    int min_width = 3;
    switch (src0->type) {
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q2_0:
            min_width = 6;
            break;
        default:
            break;
    }
    if (min_env > 0) {
        min_width = min_env;
    }
    const int64_t ne11 = src1->ne[1];
    if (ne11 < std::max(2, min_width) || ne11 > MMSMT_MAX_BATCH_SIZE) {
        return false;
    }
    if (src0->ne[0] % QK_K != 0 || src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src1->nb[0] != sizeof(float) || !ggml_is_contiguous(dst)) {
        return false;
    }
    return true;
}

void ggml_cuda_mul_mat_smt(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int64_t ne00 = src0->ne[0]; // K
    const int64_t ne01 = src0->ne[1]; // N
    const int64_t ne11 = src1->ne[1]; // T

    const int     ntiles = (int) ((ne11 + MMSMT_TILE - 1)/MMSMT_TILE);
    const int64_t tpad   = (int64_t) ntiles*MMSMT_TILE;
    cudaStream_t stream = ctx.stream();

    cudaStreamCaptureStatus capst = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capst));
    const bool capturing = capst != cudaStreamCaptureStatusNone;
    static const bool timing_env = getenv("GGML_CUDA_SMT_TIME") != nullptr && atoi(getenv("GGML_CUDA_SMT_TIME")) != 0;
    const bool timing = timing_env && !capturing;
    static cudaEvent_t ev[4];
    static bool ev_init = false;
    static double t_prep = 0.0, t_main = 0.0, t_red = 0.0, bytes = 0.0;
    static int64_t ncalls = 0;
    if (timing && !ev_init) {
        for (int i = 0; i < 4; ++i) {
            CUDA_CHECK(cudaEventCreate(&ev[i]));
        }
        ev_init = true;
    }
    if (timing) {
        CUDA_CHECK(cudaEventRecord(ev[0], stream));
    }

    ggml_cuda_pool_alloc<half2> x16(ctx.pool(), tpad*ne00/2);
    {
        const dim3 grid((unsigned) ((ne00/16 + 255)/256), (unsigned) tpad);
        mmsmt_prep_act<<<grid, 256, 0, stream>>>((const float *) src1->data, src1->nb[1]/sizeof(float), x16.get(), ne00, ne11);
    }
    if (timing) {
        CUDA_CHECK(cudaEventRecord(ev[1], stream));
    }

    // 32-row column groups; K split across CTAs (whole steps of NSB super-blocks) so the grid reaches
    // ~4 CTAs per SM; ksplit == 1 writes dst directly
    const int64_t ncolgrp = (ne01 + MMSMT_COLS - 1)/MMSMT_COLS;
    const int     nsb     = (int) (ne00/QK_K);
    const int     nsbstep = mmsmt_nsb(src0->type);
    const int     nsteps  = (nsb + nsbstep - 1)/nsbstep;
    int ksplit = (int) ((4*ggml_cuda_info().devices[ctx.device].nsm + ncolgrp - 1)/ncolgrp);
    ksplit = std::max(1, std::min(ksplit, nsteps));
    static const int ablate = getenv("GGML_CUDA_SMT_ABLATE") ? atoi(getenv("GGML_CUDA_SMT_ABLATE")) : 0; // dev only
    const int sb_per_split = ((nsteps + ksplit - 1)/ksplit)*nsbstep;
    ksplit = (nsb + sb_per_split - 1)/sb_per_split;

    ggml_cuda_pool_alloc<float> part(ctx.pool());
    if (ksplit > 1) {
        part.alloc((size_t) ksplit*tpad*ne01);
    }
    float * out = ksplit > 1 ? part.get() : (float *) dst->data;
    const dim3 grid((unsigned) ncolgrp, (unsigned) ksplit);

    static const bool check_env = getenv("GGML_CUDA_SMT_CHECK") != nullptr && atoi(getenv("GGML_CUDA_SMT_CHECK")) != 0;
    const bool check = check_env && !capturing;
    ggml_cuda_pool_alloc<float> ref(ctx.pool());
    if (check) {
        // reference through the dequant + cuBLAS route (fp16 activations too), then a structured diff
        ref.alloc(ne11*ne01);
        ggml_tensor dst_ref = *dst;
        dst_ref.data = ref.get();
        ggml_cuda_mul_mat_cublas(ctx, src0, src1, &dst_ref);
    }

    switch (src0->type) {
        case GGML_TYPE_Q8_0:
            mmsmt_launch<GGML_TYPE_Q8_0>(ntiles, grid, stream, (const char *) src0->data, src0->nb[1], x16.get(), out, ne00, ne01, ne11, sb_per_split, ablate);
            break;
        case GGML_TYPE_Q4_K:
            mmsmt_launch<GGML_TYPE_Q4_K>(ntiles, grid, stream, (const char *) src0->data, src0->nb[1], x16.get(), out, ne00, ne01, ne11, sb_per_split, ablate);
            break;
        case GGML_TYPE_Q4_0:
            mmsmt_launch<GGML_TYPE_Q4_0>(ntiles, grid, stream, (const char *) src0->data, src0->nb[1], x16.get(), out, ne00, ne01, ne11, sb_per_split, ablate);
            break;
        case GGML_TYPE_Q5_K:
            mmsmt_launch<GGML_TYPE_Q5_K>(ntiles, grid, stream, (const char *) src0->data, src0->nb[1], x16.get(), out, ne00, ne01, ne11, sb_per_split, ablate);
            break;
        case GGML_TYPE_Q6_K:
            mmsmt_launch<GGML_TYPE_Q6_K>(ntiles, grid, stream, (const char *) src0->data, src0->nb[1], x16.get(), out, ne00, ne01, ne11, sb_per_split, ablate);
            break;
        case GGML_TYPE_IQ4_XS:
            mmsmt_launch<GGML_TYPE_IQ4_XS>(ntiles, grid, stream, (const char *) src0->data, src0->nb[1], x16.get(), out, ne00, ne01, ne11, sb_per_split, ablate);
            break;
        case GGML_TYPE_Q3_K:
            mmsmt_launch<GGML_TYPE_Q3_K>(ntiles, grid, stream, (const char *) src0->data, src0->nb[1], x16.get(), out, ne00, ne01, ne11, sb_per_split, ablate);
            break;
        case GGML_TYPE_Q2_K:
            mmsmt_launch<GGML_TYPE_Q2_K>(ntiles, grid, stream, (const char *) src0->data, src0->nb[1], x16.get(), out, ne00, ne01, ne11, sb_per_split, ablate);
            break;
        case GGML_TYPE_Q2_0:
            mmsmt_launch<GGML_TYPE_Q2_0>(ntiles, grid, stream, (const char *) src0->data, src0->nb[1], x16.get(), out, ne00, ne01, ne11, sb_per_split, ablate);
            break;
        default:
            GGML_ABORT("mmsmt: unsupported type");
    }
    if (timing) {
        CUDA_CHECK(cudaEventRecord(ev[2], stream));
    }
    if (ksplit > 1) {
        const int64_t n = ne11*ne01;
        mmsmt_reduce<<<(unsigned) ((n + 255)/256), 256, 0, stream>>>(part.get(), (float *) dst->data, n, tpad*ne01, ksplit);
    }
    if (timing) {
        CUDA_CHECK(cudaEventRecord(ev[3], stream));
        CUDA_CHECK(cudaEventSynchronize(ev[3]));
        float ms01, ms12, ms23;
        CUDA_CHECK(cudaEventElapsedTime(&ms01, ev[0], ev[1]));
        CUDA_CHECK(cudaEventElapsedTime(&ms12, ev[1], ev[2]));
        CUDA_CHECK(cudaEventElapsedTime(&ms23, ev[2], ev[3]));
        t_prep += ms01; t_main += ms12; t_red += ms23;
        bytes += (double) ggml_nbytes(src0);
        if (++ncalls % 512 == 0) {
            fprintf(stderr, "mmsmt timing: %lld calls, prep %.1f ms, main %.1f ms (%.0f GB/s of weights), reduce %.1f ms; last shape K=%lld N=%lld T=%lld ksplit=%d\n",
                    (long long) ncalls, t_prep, t_main, bytes/(t_main*1e6), t_red, (long long) ne00, (long long) ne01, (long long) ne11, ksplit);
        }
    }

    if (check) {
        static int calls = 0;
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<float> h_out(ne11*ne01), h_ref(ne11*ne01);
        CUDA_CHECK(cudaMemcpy(h_out.data(), dst->data, h_out.size()*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_ref.data(), ref.get(),  h_ref.size()*sizeof(float), cudaMemcpyDeviceToHost));
        double num = 0.0, den = 0.0;
        double row_num[MMSMT_MAX_BATCH_SIZE] = {0}, row_den[MMSMT_MAX_BATCH_SIZE] = {0};
        double col_num[32] = {0}, col_den[32] = {0};
        for (int64_t t = 0; t < ne11; ++t) {
            for (int64_t n = 0; n < ne01; ++n) {
                const double o = h_out[t*ne01 + n], r = h_ref[t*ne01 + n], d = o - r;
                num += d*d; den += r*r;
                row_num[t] += d*d; row_den[t] += r*r;
                col_num[n % 32] += d*d; col_den[n % 32] += r*r;
            }
        }
        const bool bad = !(num/den < 1e-2) || std::isnan(num);
        static int nbad = 0;
        if (calls < 4 || (bad && nbad++ < 12)) {
            fprintf(stderr, "mmsmt check #%d: type %s K=%lld N=%lld T=%lld ksplit=%d tiles=%d nmse=%.3e\n", calls, ggml_type_name(src0->type),
                    (long long) ne00, (long long) ne01, (long long) ne11, ksplit, ntiles, num/den);
            fprintf(stderr, "  per row:");
            for (int64_t t = 0; t < ne11; ++t) fprintf(stderr, " %.1e", row_num[t]/row_den[t]);
            fprintf(stderr, "\n  per col%%32:");
            for (int n = 0; n < 32; ++n) fprintf(stderr, " %.1e", col_den[n] > 0 ? col_num[n]/col_den[n] : 0.0);
            fprintf(stderr, "\n  first row, first 8 cols out/ref:");
            for (int n = 0; n < 8 && n < ne01; ++n) fprintf(stderr, " %.4f/%.4f", h_out[n], h_ref[n]);
            fprintf(stderr, "\n");
        }
        calls++;
    }
}
