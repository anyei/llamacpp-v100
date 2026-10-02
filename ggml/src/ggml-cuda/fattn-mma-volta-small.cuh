#pragma once

#include "common.cuh"
#include "fattn-common.cuh"

// T3 (TASKS #153, docs/ninfer-t3-t4-plan.md section 3): Volta decode/verify FlashAttention on m8n8k4 tensor cores.
//
// The upstream Volta MMA kernel packs 32 query columns into the A tile and refuses ncols < 32; this kernel packs the
// query rows x the KV head's query heads (<= 32 rows, 8 per m8n8k4 tile) into the A tile instead, mirrored across
// the four quadpairs, so K/V are read once per KV head. Per CTA step of 128 KV positions:
//   phase 1: warp w streams K rows 32w..32w+31, one row per lane (B .col fragment = 4 dims of the lane's row),
//            S = Q K^T in fp32, softcap/mask/ALiBi, CTA-wide row max through shared memory,
//            P = exp(S - max) as fp16 A fragments in shared memory;
//   phase 2: warp w owns dims [w*D/4, (w+1)*D/4), the quadpairs split those dims (8-wide n-tiles; at D = 64 the
//            remaining factor splits positions) and loop over the M tiles, so each lane streams V[pos][its 4*NJ dims]
//            once per position as B .row fragments and O += P V in fp32 for every tile (increment 2, plan 7.2).
// q8_0 K/V are dequantised in registers (byte_perm into the fp16 mantissa, block scale folded in fp16); f16 K/V are
// raw 16-byte loads. Split-KV across CTAs and the final combine are launch_fattn's (parallel_blocks + dst_meta).
// Fragment layouts (probed on the V100, 153-p0/t3probe): lane t = (lane&3) + 4*(lane>>4) holds A row t;
// B .col: lane t holds B[k 0..3][n = t]; B .row: lane holds B[k = lane&3][n = 4*(lane>>4) + 0..3];
// D (f32): d[i] at row 4*(lane>>4) + (lane&1) + (i&2), column (lane&2) + (i&5).

#define FATTN_VS_NTHREADS 128
#define FATTN_VS_NWARPS   (FATTN_VS_NTHREADS/WARP_SIZE)
#define FATTN_VS_STEP     128   // KV positions per CTA step
#define FATTN_VS_PST      144   // P row stride in halves (128 positions + 16 pad: conflict-free 16-byte stores)
#define FATTN_VS_MAXCOLS  8     // Q rows per block (launch_fattn ncols1)
#define FATTN_VS_SMEM_HEAD(nrows) ((2*FATTN_VS_NWARPS + 1)*(nrows)*4)   // smax, ssum, sL bytes

template <int D, int NT>
static constexpr size_t fattn_vs_smem_bytes() {
    constexpr size_t nrows = 8*NT;
    constexpr size_t qp    = nrows*((D + 8) + FATTN_VS_PST)*2;   // Qs + Ps
    constexpr size_t os    = nrows*(D + 4)*4;                    // output staging (aliases Qs + Ps)
    return FATTN_VS_SMEM_HEAD(nrows) + (qp > os ? qp : os);
}

static __device__ __forceinline__ void fattn_vs_mma_rc(float (&d)[8], const uint32_t a0, const uint32_t a1, const uint32_t b0, const uint32_t b1) {
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

static __device__ __forceinline__ void fattn_vs_mma_rr(float (&d)[8], const uint32_t a0, const uint32_t a1, const uint32_t b0, const uint32_t b1) {
#if defined(VOLTA_MMA_AVAILABLE)
    asm volatile("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, {%0, %1, %2, %3, %4, %5, %6, %7};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
#else
    GGML_UNUSED_VARS(d, a0, a1, b0, b1);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE)
}

// bytes (0,1) / (2,3) of w -> half2 {1024 + b0, 1024 + b1} raw bits
static __device__ __forceinline__ uint32_t fattn_vs_bytes_lo(const uint32_t w) { return __byte_perm(w, 0x64006400u, 0x5150); }
static __device__ __forceinline__ uint32_t fattn_vs_bytes_hi(const uint32_t w) { return __byte_perm(w, 0x64006400u, 0x5352); }

// (u - 1152) * d as half2 bit patterns: u holds 1024 + (q ^ 0x80) per lane, so the result is the exact int8 q times d
static __device__ __forceinline__ uint32_t fattn_vs_dq8(const uint32_t u, const uint32_t d2) {
#ifdef FP16_AVAILABLE
    half2 a, b, c;
    *reinterpret_cast<uint32_t *>(&a) = u;
    *reinterpret_cast<uint32_t *>(&b) = 0x64806480u;
    *reinterpret_cast<uint32_t *>(&c) = d2;
    const half2 r = __hmul2(__hsub2(a, b), c);
    return *reinterpret_cast<const uint32_t *>(&r);
#else
    GGML_UNUSED_VARS(u, d2);
    NO_DEVICE_CODE;
    return 0;
#endif // FP16_AVAILABLE
}

static __device__ __forceinline__ uint32_t fattn_vs_pack(const float a, const float b) {
    const half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const uint32_t *>(&h);
}

// S[m] += Q[tile m] . K[lane's row] for the 8 dims at Qs4[.. + s8] and the 4 half2 in w
template <int NT, int QST4>
static __device__ __forceinline__ void fattn_vs_kq_slice(float (&S)[NT][8], const uint4 * __restrict__ Qs4, const int s8, const uint4 w) {
#pragma unroll
    for (int m = 0; m < NT; ++m) {
        const uint4 q = Qs4[m*8*QST4 + s8];
        fattn_vs_mma_rc(S[m], q.x, q.y, w.x, w.y);
        fattn_vs_mma_rc(S[m], q.z, q.w, w.z, w.w);
    }
}

// phase 1 K readers: the lane's K row (position) against all Q tiles. Qs4 points at row t of tile 0.
template <int D, int NT, int QST4, ggml_type type_K>
static __device__ __forceinline__ void fattn_vs_kq_row(float (&S)[NT][8], const uint4 * __restrict__ Qs4, const char * __restrict__ Krow, const bool active) {
    if constexpr (type_K == GGML_TYPE_F16) {
        const uint4 * K4 = (const uint4 *) Krow;
#pragma unroll
        for (int s8 = 0; s8 < D/8; ++s8) {
            const uint4 w = active ? K4[s8] : make_uint4(0, 0, 0, 0);
            fattn_vs_kq_slice<NT, QST4>(S, Qs4, s8, w);
        }
    } else {
        static_assert(type_K == GGML_TYPE_Q8_0, "unsupported K type");
        // a block pair (64 dims) is 17 words: block 0 = d in the low half of word 0 + quants 2 bytes in,
        // block 1 = d in the high half of word 8 + quants at words 9..16
        const uint32_t * Kw = (const uint32_t *) Krow;
#pragma unroll
        for (int bp = 0; bp < D/64; ++bp) {
            uint32_t w[17];
#pragma unroll
            for (int i = 0; i < 17; ++i) {
                w[i] = active ? Kw[17*bp + i] : 0u;
            }
            const uint32_t d0 = (w[0] & 0xffffu) | (w[0] << 16);
            const uint32_t d1 = (w[8] >> 16)     | (w[8] & 0xffff0000u);
#pragma unroll
            for (int s8 = 0; s8 < 4; ++s8) {
                const uint32_t u0 = __funnelshift_r(w[2*s8 + 0], w[2*s8 + 1], 16) ^ 0x80808080u;
                const uint32_t u1 = __funnelshift_r(w[2*s8 + 1], w[2*s8 + 2], 16) ^ 0x80808080u;
                uint4 v;
                v.x = fattn_vs_dq8(fattn_vs_bytes_lo(u0), d0);
                v.y = fattn_vs_dq8(fattn_vs_bytes_hi(u0), d0);
                v.z = fattn_vs_dq8(fattn_vs_bytes_lo(u1), d0);
                v.w = fattn_vs_dq8(fattn_vs_bytes_hi(u1), d0);
                fattn_vs_kq_slice<NT, QST4>(S, Qs4, 8*bp + s8, v);
            }
#pragma unroll
            for (int s8 = 0; s8 < 4; ++s8) {
                const uint32_t u0 = w[9 + 2*s8 + 0] ^ 0x80808080u;
                const uint32_t u1 = w[9 + 2*s8 + 1] ^ 0x80808080u;
                uint4 v;
                v.x = fattn_vs_dq8(fattn_vs_bytes_lo(u0), d1);
                v.y = fattn_vs_dq8(fattn_vs_bytes_hi(u0), d1);
                v.z = fattn_vs_dq8(fattn_vs_bytes_lo(u1), d1);
                v.w = fattn_vs_dq8(fattn_vs_bytes_hi(u1), d1);
                fattn_vs_kq_slice<NT, QST4>(S, Qs4, 8*bp + 4 + s8, v);
            }
        }
    }
}

// phase 2 V readers: the lane's DL contiguous dims of one V row as DL/4 B .row fragment pairs (n-tile j = dims 4j..4j+3)
template <int DL, ggml_type type_V>
static __device__ __forceinline__ void fattn_vs_v_row(uint32_t (&b)[DL/4][2], const char * __restrict__ Vrow, const int dim0, const bool active) {
    static_assert(DL == 4 || DL == 8, "lane V run must be 8 or 16 bytes of f16");
    if (!active) {
#pragma unroll
        for (int j = 0; j < DL/4; ++j) {
            b[j][0] = b[j][1] = 0u;
        }
        return;
    }
    if constexpr (type_V == GGML_TYPE_F16) {
        if constexpr (DL == 8) {
            const uint4 w = *(const uint4 *) (Vrow + 2*dim0);
            b[0][0] = w.x; b[0][1] = w.y;
            b[1][0] = w.z; b[1][1] = w.w;
        } else {
            const uint2 w = *(const uint2 *) (Vrow + 2*dim0);
            b[0][0] = w.x; b[0][1] = w.y;
        }
    } else {
        static_assert(type_V == GGML_TYPE_Q8_0, "unsupported V type");
        constexpr int NWQ = (DL + 6)/4; // quant words incl. the misalignment of up to 2 bytes
        const int blk  = dim0 / 32;
        const int dpos = 34*blk;                    // byte offset of the block scale
        const int qpos = dpos + 2 + (dim0 % 32);    // byte offset of the lane's first quant
        const uint32_t * Vd = (const uint32_t *) (Vrow + (dpos & ~3));
        const uint32_t * Vq = (const uint32_t *) (Vrow + (qpos & ~3));
        const uint32_t dw = Vd[0];
        const uint32_t d16 = (dpos & 2) ? (dw >> 16) : (dw & 0xffffu);
        const uint32_t d2  = d16 | (d16 << 16);
        const uint32_t sh = 8*(qpos & 3);
        uint32_t w[NWQ + 1];
#pragma unroll
        for (int i = 0; i < NWQ - 1; ++i) {
            w[i] = Vq[i];
        }
        w[NWQ - 1] = sh != 0 ? Vq[NWQ - 1] : 0;   // word-aligned quants need one word less: never read past the row
        w[NWQ] = 0;
#pragma unroll
        for (int j = 0; j < DL/4; ++j) {
            const uint32_t u = __funnelshift_r(w[j], w[j + 1], sh) ^ 0x80808080u;
            b[j][0] = fattn_vs_dq8(fattn_vs_bytes_lo(u), d2);
            b[j][1] = fattn_vs_dq8(fattn_vs_bytes_hi(u), d2);
        }
    }
}

template <int D, int NT, ggml_type type_K, ggml_type type_V, bool use_logit_softcap, bool skip_masked>
__launch_bounds__(FATTN_VS_NTHREADS, 2)
static __global__ void flash_attn_ext_mma_volta_small(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
    ggml_cuda_pdl_lc();
#if defined(FLASH_ATTN_AVAILABLE) && defined(VOLTA_MMA_AVAILABLE)
    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        NO_DEVICE_CODE;
        return;
    }
    const char * GGML_CUDA_RESTRICT Q        = Q_ptr;
    const char * GGML_CUDA_RESTRICT K        = K_ptr;
    const char * GGML_CUDA_RESTRICT V        = V_ptr;
    const char * GGML_CUDA_RESTRICT mask     = mask_ptr;
    const char * GGML_CUDA_RESTRICT sinks    = sinks_ptr;
    const int  * GGML_CUDA_RESTRICT KV_max   = KV_max_ptr;
    float      * GGML_CUDA_RESTRICT dst      = dst_ptr;
    float2     * GGML_CUDA_RESTRICT dst_meta = dst_meta_ptr;

    constexpr int NROWS = 8*NT;          // M rows (Q rows x packed heads, padded)
    static_assert(NROWS >= 2*FATTN_VS_NWARPS, "the masked-position ballots borrow sL: 2 words per warp");
    constexpr int QST   = D + 8;         // Q row stride in halves (16-byte pad: conflict-free fragment loads)
    constexpr int QST4  = QST/8;         // ... in uint4
    constexpr int NJW   = D/32;          // 8-wide n-tiles per warp (the warp owns D/4 dims)
    constexpr int NQ_D  = NJW < 4 ? NJW : 4; // quadpairs splitting the warp's n-tiles
    constexpr int NJ    = NJW/NQ_D;      // n-tiles per quadpair (its lanes hold 4*NJ contiguous dims each)
    constexpr int DL    = 4*NJ;          // V dims per lane in phase 2
    constexpr int NQ_P  = 4/NQ_D;        // position slices per warp (only D = 64 splits positions)
    constexpr int NCH   = 16/NQ_P;       // 8-position chunks per slice
    constexpr int PST4  = FATTN_VS_PST/8;    // P row stride in uint4
    constexpr int OST   = D + 4;         // output staging row stride in floats

    extern __shared__ char fattn_vs_smem[];
    float * smax = (float *) fattn_vs_smem;                       // [NWARPS][NROWS] per-step row max
    float * ssum = smax + FATTN_VS_NWARPS*NROWS;                  // [NWARPS][NROWS] row sums
    float * sL   = ssum + FATTN_VS_NWARPS*NROWS;                  // [NROWS] final row sums
    half  * Qs   = (half  *) (fattn_vs_smem + FATTN_VS_SMEM_HEAD(NROWS));       // [NROWS][QST]
    half  * Ps   = Qs + NROWS*QST;                                // [NROWS][FATTN_VS_PST]
    float * Os   = (float *) Qs;                                  // [NROWS][OST], after the KV loop

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid  = warp*WARP_SIZE + lane;
    const int t    = (lane & 3) + 4*(lane >> 4);   // A/B .col fragment row of this lane
    const int qp   = (lane >> 2) & 3;              // quadpair
    const int r0   = 4*(lane >> 4) + (lane & 1);   // D fragment rows r0, r0 + 2
    const int c0   = lane & 2;                     // D fragment columns c0, c0 + 1, c0 + 4, c0 + 5

    // block -> (sequence, KV head, head group); the group packs `pack` query heads of the KV head as rows
    const int gqa_ratio    = ne02 / ne12;
    const int ntiles_z_gqa = gridDim.z / (ne12*ne03);
    const int zb           = blockIdx.z;
    const int sequence     = zb / (ntiles_z_gqa*ne12);
    const int zr           = zb - sequence*ntiles_z_gqa*ne12;
    const int kv_head      = zr / ntiles_z_gqa;
    const int gtile        = zr - kv_head*ntiles_z_gqa;
    const int pack         = (gqa_ratio + ntiles_z_gqa - 1) / ntiles_z_gqa;
    const int h0           = kv_head*gqa_ratio + gtile*pack;
    const int nheads       = min(pack, gqa_ratio - gtile*pack);
    const int ic0          = blockIdx.x*FATTN_VS_MAXCOLS;
    const int ncols        = min(FATTN_VS_MAXCOLS, int(ne01.z) - ic0);
    const int nrows        = ncols*nheads;                // rows in use (row r = j*pack + hq)

    K += nb13*sequence + nb12*kv_head;
    V += nb23*sequence + nb22*kv_head;

    // stage Q (f32, scaled) as f16 rows; padding rows are zero
    ggml_cuda_pdl_sync();
    for (int idx = tid; idx < NROWS*(D/8); idx += FATTN_VS_NTHREADS) {
        const int r  = idx / (D/8);
        const int c8 = idx - r*(D/8);
        uint4 v = make_uint4(0, 0, 0, 0);
        if (r < nrows) {
            const int j  = r / pack;
            const int hq = r - j*pack;
            const float * Qr = (const float *) (Q + nb03*sequence + nb02*(h0 + hq) + nb01*(ic0 + j)) + 8*c8;
            v.x = fattn_vs_pack(scale*Qr[0], scale*Qr[1]);
            v.y = fattn_vs_pack(scale*Qr[2], scale*Qr[3]);
            v.z = fattn_vs_pack(scale*Qr[4], scale*Qr[5]);
            v.w = fattn_vs_pack(scale*Qr[6], scale*Qr[7]);
        }
        *(uint4 *) (Qs + r*QST + 8*c8) = v;
    }
    __syncthreads();

    // per-row constants for this lane's D rows (row r = 8m + r0 + 2h)
    float slope[NT][2];
    const half * maskrow[NT][2];
#pragma unroll
    for (int m = 0; m < NT; ++m) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r  = 8*m + r0 + 2*h;
            const int rr = r < nrows ? r : 0;
            const int j  = rr / pack;
            const int hq = rr - j*pack;
            slope[m][h]   = get_alibi_slope(max_bias, h0 + hq, n_head_log2, m0, m1);
            maskrow[m][h] = mask ? (const half *) (mask + nb33*(sequence % ne33) + nb31*(ic0 + j)) : nullptr;
        }
    }

    float O[NT][NJ][8];
#pragma unroll
    for (int m = 0; m < NT; ++m) {
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                O[m][j][i] = 0.0f;
            }
        }
    }
    float M[NT][2], L[NT][2];
#pragma unroll
    for (int m = 0; m < NT; ++m) {
        M[m][0] = M[m][1] = -FLT_MAX/2.0f;
        L[m][0] = L[m][1] = 0.0f;
    }

    // phase 2 geometry: quadpair qp -> its n-tiles (qd) and position slice; lane -> 4*NJ contiguous dims
    // (n-tile j, column n <-> dim Wd + 8*NJ*qd + 4*NJ*(n/4) + 4*j + n%4); every quadpair covers every M tile
    const int qd     = qp % NQ_D;
    const int oslice = qp / NQ_D;
    const int dim0   = warp*(D/4) + 8*NJ*qd + DL*(lane >> 4);
    const uint4 * Qs4 = (const uint4 *) (Qs + t*QST);
    const uint4 * Ps4 = (const uint4 *) (Ps + t*FATTN_VS_PST);   // tile m: + m*8*PST4

    const int k_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;

    // TASKS #156: a position that every query row of the block masks with -inf contributes exact zeros, so its K and V
    // rows are not loaded (zeros enter the mma instead) and a step without any visible position is skipped. Sparse
    // masks (qwen4exp QSA: ~2k of n_kv cells per row) then read only the selected rows. sact holds one ballot word per
    // warp, double-buffered by step parity; sL is free until the epilogue.
    uint32_t * sact = (uint32_t *) sL;
    const char * mask_seq = skip_masked && mask ? mask + nb33*(sequence % ne33) + nb31*ic0 : nullptr;
    int it = 0;

    for (int k0 = blockIdx.y*FATTN_VS_STEP; k0 < k_max; k0 += gridDim.y*FATTN_VS_STEP, ++it) {
        // ---- phase 1: S = Q K^T for the warp's 32 positions, one K row per lane ----
        const int pos = k0 + 32*warp + lane;
        bool active = true;
        if (mask_seq) {
            active = false;
            for (int j = 0; j < ncols; ++j) {
                active |= __half2float(((const half *) (mask_seq + nb31*j))[pos]) != -INFINITY;
            }
        }
        if constexpr (skip_masked) {
            const uint32_t ballot = __ballot_sync(0xFFFFFFFF, active);
            if (lane == 0) {
                sact[(it & 1)*FATTN_VS_NWARPS + warp] = ballot;
            }
        }
        float S[NT][8];
#pragma unroll
        for (int m = 0; m < NT; ++m) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                S[m][i] = 0.0f;
            }
        }
        fattn_vs_kq_row<D, NT, QST4, type_K>(S, Qs4, K + (int64_t) pos*nb11, active);

        // softcap, mask, ALiBi; the lane's columns are positions 32*warp + 4*qp + c0 + {0,1} and + 16 + {0,1}
        const int pc0 = k0 + 32*warp + 4*qp + c0;
#pragma unroll
        for (int m = 0; m < NT; ++m) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                float2 mk0 = make_float2(0.0f, 0.0f);
                float2 mk1 = make_float2(0.0f, 0.0f);
                if (mask) {
                    mk0 = __half22float2(*(const half2 *) (maskrow[m][h] + pc0));
                    mk1 = __half22float2(*(const half2 *) (maskrow[m][h] + pc0 + 16));
                }
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    if ((i & 2) != 2*h) {
                        continue;
                    }
                    float s = S[m][i];
                    if (use_logit_softcap) {
                        s = logit_softcap*tanhf(s);
                    }
                    if (mask) {
                        const float mv = (i & 4) ? ((i & 1) ? mk1.y : mk1.x) : ((i & 1) ? mk0.y : mk0.x);
                        s += slope[m][h]*mv;
                    }
                    S[m][i] = s;
                }
            }
        }

        // row max over the warp's 32 positions -> shared
#pragma unroll
        for (int m = 0; m < NT; ++m) {
            float mx0 = fmaxf(fmaxf(S[m][0], S[m][1]), fmaxf(S[m][4], S[m][5]));
            float mx1 = fmaxf(fmaxf(S[m][2], S[m][3]), fmaxf(S[m][6], S[m][7]));
#pragma unroll
            for (int off = 2; off <= 8; off <<= 1) {
                mx0 = fmaxf(mx0, __shfl_xor_sync(0xFFFFFFFF, mx0, off, WARP_SIZE));
                mx1 = fmaxf(mx1, __shfl_xor_sync(0xFFFFFFFF, mx1, off, WARP_SIZE));
            }
            if ((lane & 0xE) == 0) {
                smax[warp*NROWS + 8*m + r0    ] = mx0;
                smax[warp*NROWS + 8*m + r0 + 2] = mx1;
            }
        }
        __syncthreads();

        const uint32_t * sa = sact + (it & 1)*FATTN_VS_NWARPS;
        if constexpr (skip_masked) {
            uint32_t any_active = 0;
#pragma unroll
            for (int w = 0; w < FATTN_VS_NWARPS; ++w) {
                any_active |= sa[w];
            }
            if (any_active == 0) {
                continue;
            }
        }

        // CTA-wide new max per row, rescale, P = exp(S - max) as fp16 A fragments -> Ps
#pragma unroll
        for (int m = 0; m < NT; ++m) {
            float Mn[2], sc[2];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                float mx = smax[8*m + r0 + 2*h];
#pragma unroll
                for (int w = 1; w < FATTN_VS_NWARPS; ++w) {
                    mx = fmaxf(mx, smax[w*NROWS + 8*m + r0 + 2*h]);
                }
                Mn[h] = fmaxf(M[m][h], mx + FATTN_KQ_MAX_OFFSET);
                sc[h] = expf(M[m][h] - Mn[h]);
                M[m][h] = Mn[h];
                L[m][h] *= sc[h];
            }
            float P[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                P[i] = expf(S[m][i] - Mn[(i >> 1) & 1]);
            }
            L[m][0] += (P[0] + P[1]) + (P[4] + P[5]);
            L[m][1] += (P[2] + P[3]) + (P[6] + P[7]);
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    O[m][j][i] *= sc[(i >> 1) & 1];
                }
            }
            // D layout -> A layout: lanes 2 apart exchange the half of their rows they do not keep
            const uint32_t h01 = fattn_vs_pack(P[0], P[1]);
            const uint32_t h23 = fattn_vs_pack(P[2], P[3]);
            const uint32_t h45 = fattn_vs_pack(P[4], P[5]);
            const uint32_t h67 = fattn_vs_pack(P[6], P[7]);
            const bool     up  = (lane & 2) != 0;
            const uint32_t rc0 = __shfl_xor_sync(0xFFFFFFFF, up ? h01 : h23, 2, WARP_SIZE);
            const uint32_t rc1 = __shfl_xor_sync(0xFFFFFFFF, up ? h45 : h67, 2, WARP_SIZE);
            uint4 a;
            a.x = up ? rc0 : h01;
            a.y = up ? h23 : rc0;
            a.z = up ? rc1 : h45;
            a.w = up ? h67 : rc1;
            *(uint4 *) (Ps + (8*m + t)*FATTN_VS_PST + 32*warp + 8*qp) = a;
        }
        __syncthreads();

        // ---- phase 2: O += P V over this quadpair's chunks for every tile; chunk kc = warp kc/4, quadpair kc%4 of phase 1 ----
#pragma unroll
        for (int ch = 0; ch < NCH; ++ch) {
            const int kc = oslice*NCH + ch;
            const int pv = k0 + 32*(kc >> 2) + 4*(kc & 3) + (lane & 3);  // mma 0 position; mma 1 = pv + 16
            uint32_t b0[NJ][2], b1[NJ][2];
            const int o0 = pv - k0;
            const int o1 = o0 + 16;
            fattn_vs_v_row<DL, type_V>(b0, V + (int64_t) pv*nb21,        dim0, !skip_masked || ((sa[o0 >> 5] >> (o0 & 31)) & 1u));
            fattn_vs_v_row<DL, type_V>(b1, V + (int64_t) (pv + 16)*nb21, dim0, !skip_masked || ((sa[o1 >> 5] >> (o1 & 31)) & 1u));
#pragma unroll
            for (int m = 0; m < NT; ++m) {
                const uint4 a = Ps4[m*8*PST4 + kc];
#pragma unroll
                for (int j = 0; j < NJ; ++j) {
                    fattn_vs_mma_rr(O[m][j], a.x, a.y, b0[j][0], b0[j][1]);
                    fattn_vs_mma_rr(O[m][j], a.z, a.w, b1[j][0], b1[j][1]);
                }
            }
        }
    }

    // ---- epilogue ----
    // position slices (D = 64 only): sum the partial O across the two slices (lane bit 3)
    if (NQ_P == 2) {
#pragma unroll
        for (int m = 0; m < NT; ++m) {
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    O[m][j][i] += __shfl_xor_sync(0xFFFFFFFF, O[m][j][i], 8, WARP_SIZE);
                }
            }
        }
    }
    // row sums: reduce over the lanes sharing a row, then over warps through shared memory
#pragma unroll
    for (int m = 0; m < NT; ++m) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int off = 2; off <= 8; off <<= 1) {
                L[m][h] += __shfl_xor_sync(0xFFFFFFFF, L[m][h], off, WARP_SIZE);
            }
        }
        if ((lane & 0xE) == 0) {
            ssum[warp*NROWS + 8*m + r0    ] = L[m][0];
            ssum[warp*NROWS + 8*m + r0 + 2] = L[m][1];
        }
    }
    __syncthreads();  // ssum visible; Qs/Ps reads are done (Os aliases them)

    // sinks: one extra logit per row (all lanes of the row apply the same rescale)
    float Ltot[NT][2];
#pragma unroll
    for (int m = 0; m < NT; ++m) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float l = 0.0f;
#pragma unroll
            for (int w = 0; w < FATTN_VS_NWARPS; ++w) {
                l += ssum[w*NROWS + 8*m + r0 + 2*h];
            }
            if (sinks && blockIdx.y == 0) {
                const int r  = 8*m + r0 + 2*h;
                const int rr = r < nrows ? r : 0;
                const int hq = rr - (rr / pack)*pack;
                const float sink = ((const float *) sinks)[h0 + hq];
                const float Mn = fmaxf(M[m][h], sink);
                const float sc = expf(M[m][h] - Mn);
                l = l*sc + expf(sink - Mn);
                M[m][h] = Mn;
#pragma unroll
                for (int j = 0; j < NJ; ++j) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        if (((i >> 1) & 1) == h) {
                            O[m][j][i] *= sc;
                        }
                    }
                }
            }
            Ltot[m][h] = l;
            if (warp == 0 && (lane & 0xE) == 0) {
                sL[8*m + r0 + 2*h] = l;
            }
        }
    }

    // stage O rows: lane holds rows r0/r0+2 of every tile; column n of n-tile j -> dim Wd + 8*NJ*qd + 4*NJ*(n/4) + 4*j + n%4
    if (oslice == 0) {
#pragma unroll
        for (int m = 0; m < NT; ++m) {
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int r = 8*m + r0 + 2*h;
                    float * Or = Os + r*OST + warp*(D/4) + 8*NJ*qd + 4*j + c0;
                    *(float2 *) (Or     ) = make_float2(O[m][j][2*h + 0], O[m][j][2*h + 1]);
                    *(float2 *) (Or + DL) = make_float2(O[m][j][2*h + 4], O[m][j][2*h + 5]);
                }
            }
        }
    }
    __syncthreads();

    // write out: row r -> (query ic0 + j, head h0 + hq); normalise unless split-KV combine follows
    for (int idx = tid; idx < nrows*(D/4); idx += FATTN_VS_NTHREADS) {
        const int r  = idx / (D/4);
        const int c4 = idx - r*(D/4);
        const int j  = r / pack;
        const int hq = r - j*pack;
        const float4 o = *(const float4 *) (Os + r*OST + 4*c4);
        const int j_dst = ((sequence*int(ne01.z) + ic0 + j)*ne02 + h0 + hq)*gridDim.y + blockIdx.y;
        float4 res = o;
        if (gridDim.y == 1) {
            const float l = sL[r];
            res.x = o.x/l; res.y = o.y/l; res.z = o.z/l; res.w = o.w/l;
        }
        *(float4 *) (dst + (int64_t) j_dst*D + 4*c4) = res;
    }
    if (gridDim.y != 1 && warp == 0) {
        // dst_meta per row: (max, sum); lanes 0,1,16,17 hold distinct rows
        if ((lane & 0xE) == 0) {
#pragma unroll
            for (int m = 0; m < NT; ++m) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int r = 8*m + r0 + 2*h;
                    if (r < nrows) {
                        const int j  = r / pack;
                        const int hq = r - j*pack;
                        dst_meta[((sequence*int(ne01.z) + ic0 + j)*ne02 + h0 + hq)*gridDim.y + blockIdx.y] = make_float2(M[m][h], Ltot[m][h]);
                    }
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, dst_ptr, dst_meta_ptr, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // defined(FLASH_ATTN_AVAILABLE) && defined(VOLTA_MMA_AVAILABLE)
}

template <int D, int NT, ggml_type type_K, ggml_type type_V, bool use_logit_softcap, bool skip_masked>
static void ggml_cuda_flash_attn_ext_mma_volta_small_case_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const int ncols2) {
    fattn_kernel_t fattn_kernel = flash_attn_ext_mma_volta_small<D, NT, type_K, type_V, use_logit_softcap, skip_masked>;
    constexpr size_t nbytes_shared = fattn_vs_smem_bytes<D, NT>();
    const bool need_f16_K = type_K == GGML_TYPE_F16;
    const bool need_f16_V = type_V == GGML_TYPE_F16;
    switch (ncols2) {
        case  1: launch_fattn<D, FATTN_VS_MAXCOLS,  1>(ctx, dst, fattn_kernel, FATTN_VS_NWARPS, nbytes_shared, FATTN_VS_STEP, need_f16_K, need_f16_V, false); break;
        case  2: launch_fattn<D, FATTN_VS_MAXCOLS,  2>(ctx, dst, fattn_kernel, FATTN_VS_NWARPS, nbytes_shared, FATTN_VS_STEP, need_f16_K, need_f16_V, false); break;
        case  4: launch_fattn<D, FATTN_VS_MAXCOLS,  4>(ctx, dst, fattn_kernel, FATTN_VS_NWARPS, nbytes_shared, FATTN_VS_STEP, need_f16_K, need_f16_V, false); break;
        case  8: launch_fattn<D, FATTN_VS_MAXCOLS,  8>(ctx, dst, fattn_kernel, FATTN_VS_NWARPS, nbytes_shared, FATTN_VS_STEP, need_f16_K, need_f16_V, false); break;
        case 16: launch_fattn<D, FATTN_VS_MAXCOLS, 16>(ctx, dst, fattn_kernel, FATTN_VS_NWARPS, nbytes_shared, FATTN_VS_STEP, need_f16_K, need_f16_V, false); break;
        case 32: launch_fattn<D, FATTN_VS_MAXCOLS, 32>(ctx, dst, fattn_kernel, FATTN_VS_NWARPS, nbytes_shared, FATTN_VS_STEP, need_f16_K, need_f16_V, false); break;
        default: GGML_ABORT("fatal error");
    }
}

template <int D, int NT, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_mma_volta_small_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const int ncols2) {
    float logit_softcap;
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    // TASKS #156: positions every query row masks are not loaded (exact); GGML_CUDA_FA_SKIP_MASKED=0 restores the dense read
    static const bool skip = [] {
        const char * e = getenv("GGML_CUDA_FA_SKIP_MASKED");
        return e == nullptr || atoi(e) != 0;
    }();
    if (logit_softcap == 0.0f) {
        if (skip) {
            ggml_cuda_flash_attn_ext_mma_volta_small_case_impl<D, NT, type_K, type_V, false, true>(ctx, dst, ncols2);
        } else {
            ggml_cuda_flash_attn_ext_mma_volta_small_case_impl<D, NT, type_K, type_V, false, false>(ctx, dst, ncols2);
        }
    } else {
        if (skip) {
            ggml_cuda_flash_attn_ext_mma_volta_small_case_impl<D, NT, type_K, type_V, true, true>(ctx, dst, ncols2);
        } else {
            ggml_cuda_flash_attn_ext_mma_volta_small_case_impl<D, NT, type_K, type_V, true, false>(ctx, dst, ncols2);
        }
    }
}

#define DECL_FATTN_MMA_VOLTA_SMALL_CASE(D, NT, type_K, type_V)                                  \
    template void ggml_cuda_flash_attn_ext_mma_volta_small_case                                 \
    <D, NT, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const int ncols2) \

#define EXTERN_DECL_FATTN_MMA_VOLTA_SMALL_CASES(D, type_K, type_V)     \
    extern DECL_FATTN_MMA_VOLTA_SMALL_CASE(D, 1, type_K, type_V);      \
    extern DECL_FATTN_MMA_VOLTA_SMALL_CASE(D, 2, type_K, type_V);      \
    extern DECL_FATTN_MMA_VOLTA_SMALL_CASE(D, 3, type_K, type_V);      \
    extern DECL_FATTN_MMA_VOLTA_SMALL_CASE(D, 4, type_K, type_V);      \

EXTERN_DECL_FATTN_MMA_VOLTA_SMALL_CASES( 64, GGML_TYPE_F16,  GGML_TYPE_F16)
EXTERN_DECL_FATTN_MMA_VOLTA_SMALL_CASES(128, GGML_TYPE_F16,  GGML_TYPE_F16)
EXTERN_DECL_FATTN_MMA_VOLTA_SMALL_CASES(256, GGML_TYPE_F16,  GGML_TYPE_F16)
EXTERN_DECL_FATTN_MMA_VOLTA_SMALL_CASES( 64, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_MMA_VOLTA_SMALL_CASES(128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_MMA_VOLTA_SMALL_CASES(256, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
