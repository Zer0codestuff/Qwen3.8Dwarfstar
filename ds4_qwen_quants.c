#include "ds4_qwen_quants.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#include <Accelerate/Accelerate.h>
#endif

#define QK_K 256
#define K_SCALE_SIZE 12
#define QK8_0 32

typedef struct {
    uint8_t hmask[QK_K / 8];
    uint8_t qs[QK_K / 4];
    uint8_t scales[12];
    uint16_t d;
} block_q3_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K / 2];
} block_q4_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K / 8];
    uint8_t qs[QK_K / 2];
} block_q5_K;

typedef struct {
    uint8_t ql[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t scales[QK_K / 16];
    uint16_t d;
} block_q6_K;

typedef struct {
    uint16_t d;
    int8_t qs[QK8_0];
} block_q8_0;

_Static_assert(sizeof(block_q3_K) == 110, "q3_K block");
_Static_assert(sizeof(block_q4_K) == 144, "q4_K block");
_Static_assert(sizeof(block_q5_K) == 176, "q5_K block");
_Static_assert(sizeof(block_q6_K) == 210, "q6_K block");
_Static_assert(sizeof(block_q8_0) == 34, "q8_0 block");

static inline float fp16_to_f32(uint16_t h)
{
#if defined(__APPLE__)
    __fp16 x;
    memcpy(&x, &h, sizeof(x));
    return (float)x;
#else
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        bits = sign;
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
#endif
}

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m)
{
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

static void dequant_q3_k(const block_q3_K *x, float *y, int64_t k)
{
    assert(k % QK_K == 0);
    const int nb = (int)(k / QK_K);
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    const int8_t *scales = (const int8_t *)aux;

    for (int i = 0; i < nb; i++) {
        const float d_all = fp16_to_f32(x[i].d);
        const uint8_t *q = x[i].qs;
        const uint8_t *hm = x[i].hmask;
        uint8_t m = 1;
        memcpy(aux, x[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        int is = 0;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4));
                }
                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                }
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

static void dequant_q4_k(const block_q4_K *x, float *y, int64_t k)
{
    assert(k % QK_K == 0);
    const int nb = (int)(k / QK_K);
    for (int i = 0; i < nb; i++) {
        const uint8_t *q = x[i].qs;
        const float d = fp16_to_f32(x[i].d);
        const float min = fp16_to_f32(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
            q += 32;
            is += 2;
        }
    }
}

static void dequant_q5_k(const block_q5_K *x, float *y, int64_t k)
{
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    for (int i = 0; i < nb; i++) {
        const uint8_t *ql = x[i].qs;
        const uint8_t *qh = x[i].qh;
        const float d = fp16_to_f32(x[i].d);
        const float min = fp16_to_f32(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

static void dequant_q6_k(const block_q6_K *x, float *y, int64_t k)
{
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_f32(x[i].d);
        const uint8_t *ql = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t *sc = x[i].scales;
        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

static void dequant_q8_0(const block_q8_0 *x, float *y, int64_t k)
{
    assert(k % QK8_0 == 0);
    const int nb = (int)(k / QK8_0);
    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_f32(x[i].d);
        for (int j = 0; j < QK8_0; ++j) {
            y[i * QK8_0 + j] = x[i].qs[j] * d;
        }
    }
}

size_t ds4_qwen_row_bytes(uint32_t type, uint64_t n_elem)
{
    switch (type) {
    case DS4_QWEN_F32:  return (size_t)n_elem * 4u;
    case DS4_QWEN_Q8_0: return (size_t)((n_elem / QK8_0) * sizeof(block_q8_0));
    case DS4_QWEN_Q3_K: return (size_t)((n_elem / QK_K) * sizeof(block_q3_K));
    case DS4_QWEN_Q4_K: return (size_t)((n_elem / QK_K) * sizeof(block_q4_K));
    case DS4_QWEN_Q5_K: return (size_t)((n_elem / QK_K) * sizeof(block_q5_K));
    case DS4_QWEN_Q6_K: return (size_t)((n_elem / QK_K) * sizeof(block_q6_K));
    default: return 0;
    }
}

int ds4_qwen_dequant_row(uint32_t type, const void *row, float *out, uint64_t n_elem)
{
    if (!row || !out || n_elem == 0) return 1;
    switch (type) {
    case DS4_QWEN_F32:
        memcpy(out, row, (size_t)n_elem * sizeof(float));
        return 0;
    case DS4_QWEN_Q8_0:
        dequant_q8_0((const block_q8_0 *)row, out, (int64_t)n_elem);
        return 0;
    case DS4_QWEN_Q3_K:
        dequant_q3_k((const block_q3_K *)row, out, (int64_t)n_elem);
        return 0;
    case DS4_QWEN_Q4_K:
        dequant_q4_k((const block_q4_K *)row, out, (int64_t)n_elem);
        return 0;
    case DS4_QWEN_Q5_K:
        dequant_q5_k((const block_q5_K *)row, out, (int64_t)n_elem);
        return 0;
    case DS4_QWEN_Q6_K:
        dequant_q6_k((const block_q6_K *)row, out, (int64_t)n_elem);
        return 0;
    default:
        return 1;
    }
}

typedef struct {
    float d;
    int8_t qs[QK_K];
    int16_t bsums[QK_K / 16];
} block_q8_K;

_Static_assert(sizeof(block_q8_K) == 292, "q8_K block");

static inline int nearest_int(float f)
{
    return (int)roundf(f);
}

static void quantize_row_q8_K(const float *x, block_q8_K *y, int64_t k)
{
    const int nb = (int)(k / QK_K);
    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;
        float max = 0.0f;
        for (int j = 0; j < QK_K; j++) {
            const float ax = fabsf(x[j]);
            if (ax > amax) {
                amax = ax;
                max = x[j];
            }
        }
        if (amax == 0.0f) {
            y[i].d = 0.0f;
            memset(y[i].qs, 0, QK_K);
            memset(y[i].bsums, 0, sizeof(y[i].bsums));
            x += QK_K;
            continue;
        }
        const float iscale = -127.0f / max;
        for (int j = 0; j < QK_K; j++) {
            int v = nearest_int(iscale * x[j]);
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            y[i].qs[j] = (int8_t)v;
        }
        for (int j = 0; j < QK_K / 16; j++) {
            int sum = 0;
            for (int ii = 0; ii < 16; ii++) sum += y[i].qs[j * 16 + ii];
            y[i].bsums[j] = (int16_t)sum;
        }
        y[i].d = 1.0f / iscale;
        x += QK_K;
    }
}

static float dot_q3_K_q8_K(const block_q3_K *x, const block_q8_K *y, int n)
{
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    const int nb = n / QK_K;
    int8_t aux8[QK_K];
    int16_t aux16[8];
    float sums[8];
    int32_t aux32[8];
    memset(sums, 0, sizeof(sums));
    uint32_t auxs[4];
    const int8_t *scales = (const int8_t *)auxs;
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const uint8_t *q3 = x[i].qs;
        const uint8_t *hm = x[i].hmask;
        const int8_t *q8 = y[i].qs;
        memset(aux32, 0, sizeof(aux32));
        int8_t *a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) a[l] = q3[l] & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 2) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 4) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 6) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            q3 += 32;
        }
        a = aux8;
        memcpy(auxs, x[i].scales, 12);
        uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        for (int j = 0; j < QK_K / 16; ++j) {
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = fp16_to_f32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * (float)aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    return sumf;
}

static float dot_q4_K_q8_K(const block_q4_K *x, const block_q8_K *y, int n)
{
    const int nb = n / QK_K;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];
    const uint8_t *scales = (const uint8_t *)&utmp[0];
    const uint8_t *mins = (const uint8_t *)&utmp[2];
    int8_t aux8[QK_K];
    int16_t aux16[8];
    float sums[8];
    int32_t aux32[8];
    memset(sums, 0, sizeof(sums));
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const uint8_t *q4 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        memset(aux32, 0, sizeof(aux32));
        int8_t *a = aux8;
        for (int j = 0; j < QK_K / 64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] >> 4);
            a += 32; q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        int sumi = 0;
        for (int j = 0; j < QK_K / 16; ++j) sumi += y[i].bsums[j] * mins[j / 2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K / 32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = fp16_to_f32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * (float)aux32[l];
        const float dmin = fp16_to_f32(x[i].dmin) * y[i].d;
        sumf -= dmin * (float)sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    return sumf;
}

static float dot_q5_K_q8_K(const block_q5_K *x, const block_q8_K *y, int n)
{
    const int nb = n / QK_K;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];
    const uint8_t *scales = (const uint8_t *)&utmp[0];
    const uint8_t *mins = (const uint8_t *)&utmp[2];
    int8_t aux8[QK_K];
    int16_t aux16[8];
    float sums[8];
    int32_t aux32[8];
    memset(sums, 0, sizeof(sums));
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const uint8_t *q4 = x[i].qs;
        const uint8_t *hm = x[i].qh;
        const int8_t *q8 = y[i].qs;
        memset(aux32, 0, sizeof(aux32));
        int8_t *a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K / 64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] >> 4);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        int sumi = 0;
        for (int j = 0; j < QK_K / 16; ++j) sumi += y[i].bsums[j] * mins[j / 2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K / 32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = fp16_to_f32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * (float)aux32[l];
        const float dmin = fp16_to_f32(x[i].dmin) * y[i].d;
        sumf -= dmin * (float)sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    return sumf;
}

static float dot_q6_K_q8_K(const block_q6_K *x, const block_q8_K *y, int n)
{
    const int nb = n / QK_K;
    int8_t aux8[QK_K];
    int16_t aux16[8];
    float sums[8];
    int32_t aux32[8];
    memset(sums, 0, sizeof(sums));
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const uint8_t *q4 = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t *q8 = y[i].qs;
        memset(aux32, 0, sizeof(aux32));
        int8_t *a = aux8;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                a[l + 0] = (int8_t)((q4[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                a[l + 32] = (int8_t)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                a[l + 64] = (int8_t)((q4[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                a[l + 96] = (int8_t)((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
            a += 128;
            q4 += 64;
            qh += 32;
        }
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K / 16; ++j) {
            int scale = x[i].scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = fp16_to_f32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * (float)aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    return sumf;
}

typedef struct {
    uint32_t type;
    const uint8_t *weight;
    const float *x;
    const block_q8_K *xq;
    float *out;
    uint64_t n_in;
    size_t row_bytes;
} qwen_mv_job;

static void qwen_mv_row(size_t row, const qwen_mv_job *job)
{
    const uint64_t n_in = job->n_in;
    const void *src = job->weight + row * job->row_bytes;
    if (job->xq) {
        float acc = 0.0f;
        switch (job->type) {
        case DS4_QWEN_Q3_K:
            acc = dot_q3_K_q8_K((const block_q3_K *)src, job->xq, (int)n_in);
            break;
        case DS4_QWEN_Q4_K:
            acc = dot_q4_K_q8_K((const block_q4_K *)src, job->xq, (int)n_in);
            break;
        case DS4_QWEN_Q5_K:
            acc = dot_q5_K_q8_K((const block_q5_K *)src, job->xq, (int)n_in);
            break;
        case DS4_QWEN_Q6_K:
            acc = dot_q6_K_q8_K((const block_q6_K *)src, job->xq, (int)n_in);
            break;
        default:
            break;
        }
        job->out[row] = acc;
        return;
    }
    float tmp[17408];
    float *rowbuf = tmp;
    float *heap = NULL;
    if (n_in > 17408) {
        heap = (float *)malloc((size_t)n_in * sizeof(float));
        rowbuf = heap;
        if (!rowbuf) return;
    }
    if (ds4_qwen_dequant_row(job->type, src, rowbuf, n_in) != 0) {
        free(heap);
        return;
    }
#if defined(__APPLE__)
    job->out[row] = cblas_sdot((int)n_in, rowbuf, 1, job->x, 1);
#else
    float acc = 0.0f;
    for (uint64_t i = 0; i < n_in; i++) acc += rowbuf[i] * job->x[i];
    job->out[row] = acc;
#endif
    free(heap);
}

void ds4_qwen_matvec(uint32_t type, const void *weight, const float *x,
                     float *out, uint64_t n_in, uint64_t n_out)
{
    if (type == DS4_QWEN_F32) {
        const float *w = (const float *)weight;
#if defined(__APPLE__)
        cblas_sgemv(CblasColMajor, CblasTrans,
                    (int)n_in, (int)n_out, 1.0f,
                    w, (int)n_in, x, 1, 0.0f, out, 1);
#else
        for (uint64_t r = 0; r < n_out; r++) {
            float acc = 0.0f;
            const float *row = w + r * n_in;
            for (uint64_t i = 0; i < n_in; i++) acc += row[i] * x[i];
            out[r] = acc;
        }
#endif
        return;
    }

    qwen_mv_job job;
    job.type = type;
    job.weight = (const uint8_t *)weight;
    job.x = x;
    job.xq = NULL;
    job.out = out;
    job.n_in = n_in;
    job.row_bytes = ds4_qwen_row_bytes(type, n_in);
    if (job.row_bytes == 0) return;

    block_q8_K *xq = NULL;
    const int use_q8 = (n_in % QK_K == 0) &&
        (type == DS4_QWEN_Q3_K || type == DS4_QWEN_Q4_K ||
         type == DS4_QWEN_Q5_K || type == DS4_QWEN_Q6_K);
    if (use_q8) {
        xq = (block_q8_K *)malloc((size_t)(n_in / QK_K) * sizeof(block_q8_K));
        if (xq) {
            quantize_row_q8_K(x, xq, (int64_t)n_in);
            job.xq = xq;
        }
    }

#if defined(__APPLE__)
    dispatch_apply((size_t)n_out, DISPATCH_APPLY_AUTO, ^(size_t row) {
        qwen_mv_row(row, &job);
    });
#else
    for (uint64_t r = 0; r < n_out; r++) qwen_mv_row((size_t)r, &job);
#endif
    free(xq);
}

int ds4_qwen_matvec_selftest(uint32_t type, const void *weight,
                             uint64_t n_in, uint64_t n_out)
{
    if (!weight || n_in == 0 || n_out == 0) return 1;
    if (type != DS4_QWEN_Q3_K && type != DS4_QWEN_Q4_K &&
        type != DS4_QWEN_Q5_K && type != DS4_QWEN_Q6_K) {
        return 0;
    }
    const uint64_t rows = n_out < 8 ? n_out : 8;
    const size_t rb = ds4_qwen_row_bytes(type, n_in);
    if (rb == 0) return 1;
    float *x = (float *)malloc((size_t)n_in * sizeof(float));
    float *fast = (float *)malloc((size_t)rows * sizeof(float));
    float *rowbuf = (float *)malloc((size_t)n_in * sizeof(float));
    if (!x || !fast || !rowbuf) {
        free(x);
        free(fast);
        free(rowbuf);
        return 1;
    }
    for (uint64_t i = 0; i < n_in; i++) {
        x[i] = sinf((float)i * 0.017f) * 0.35f;
    }
    ds4_qwen_matvec(type, weight, x, fast, n_in, rows);
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    for (uint64_t r = 0; r < rows; r++) {
        const uint8_t *src = (const uint8_t *)weight + r * rb;
        if (ds4_qwen_dequant_row(type, src, rowbuf, n_in) != 0) {
            free(x);
            free(fast);
            free(rowbuf);
            return 1;
        }
        float ref = 0.0f;
        for (uint64_t i = 0; i < n_in; i++) ref += rowbuf[i] * x[i];
        const float err = fabsf(fast[r] - ref);
        if (err > max_abs) max_abs = err;
        const float rel = err / (fabsf(ref) + 1e-3f);
        if (rel > max_rel) max_rel = rel;
    }
    free(x);
    free(fast);
    free(rowbuf);
    fprintf(stderr, "ds4: qwen matvec selftest type=%u rows=%llu max_abs=%.5g max_rel=%.5g\n",
            type, (unsigned long long)rows, max_abs, max_rel);
    return max_rel > 0.05f ? 1 : 0;
}
