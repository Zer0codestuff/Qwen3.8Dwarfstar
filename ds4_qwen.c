#include "ds4_qwen.h"
#include "ds4_qwen_quants.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define QWEN_MAX_BLOCKS 80
#define QWEN_NEG_INF (-1.0e30f)

typedef enum {
    QWEN_LAYER_LINEAR = 0,
    QWEN_LAYER_FULL = 1,
} qwen_layer_kind;

typedef struct {
    ds4_qwen_wt attn_norm;
    ds4_qwen_wt post_norm;
    ds4_qwen_wt ffn_gate;
    ds4_qwen_wt ffn_up;
    ds4_qwen_wt ffn_down;
    ds4_qwen_wt attn_qkv;
    ds4_qwen_wt attn_gate;
    ds4_qwen_wt ssm_conv1d;
    ds4_qwen_wt ssm_dt;
    ds4_qwen_wt ssm_a;
    ds4_qwen_wt ssm_alpha;
    ds4_qwen_wt ssm_beta;
    ds4_qwen_wt ssm_norm;
    ds4_qwen_wt ssm_out;
} qwen_linear_w;

typedef struct {
    ds4_qwen_wt attn_norm;
    ds4_qwen_wt post_norm;
    ds4_qwen_wt ffn_gate;
    ds4_qwen_wt ffn_up;
    ds4_qwen_wt ffn_down;
    ds4_qwen_wt attn_q;
    ds4_qwen_wt attn_k;
    ds4_qwen_wt attn_v;
    ds4_qwen_wt attn_output;
    ds4_qwen_wt attn_q_norm;
    ds4_qwen_wt attn_k_norm;
} qwen_full_w;

typedef struct {
    qwen_layer_kind kind;
    union {
        qwen_linear_w linear;
        qwen_full_w full;
    } u;
} qwen_layer;

struct ds4_qwen_runtime {
    uint32_t n_vocab;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_head_dim;
    uint32_t n_rot;
    uint32_t interval;
    uint32_t d_conv;
    uint32_t d_state;
    uint32_t n_k_head;
    uint32_t n_v_head;
    uint32_t conv_dim;
    uint32_t ctx;
    uint32_t pos;
    int32_t rope_sections[4];
    float rms_eps;
    float rope_base;
    ds4_qwen_wt token_embd;
    ds4_qwen_wt output;
    ds4_qwen_wt output_norm;
    qwen_layer layers[QWEN_MAX_BLOCKS];
    float *x;
    float *xn;
    float *resid;
    float *qkv;
    float *conv_y;
    float *z;
    float *gate_up;
    float *ffn_mid;
    float *q;
    float *k;
    float *v;
    float *attn_out;
    float *scores;
    float *conv_state;
    float *recurrent;
    float *kv_k;
    float *kv_v;
};

static int req_wt(ds4_engine *e, const char *name, ds4_qwen_wt *out)
{
    if (ds4_qwen_weight_lookup(e, name, out) != 0) {
        fprintf(stderr, "ds4: qwen35 missing tensor %s\n", name);
        return 1;
    }
    return 0;
}

static int req_wt_fmt(ds4_engine *e, ds4_qwen_wt *out, const char *fmt, uint32_t il)
{
    char name[96];
    snprintf(name, sizeof(name), fmt, il);
    return req_wt(e, name, out);
}

static uint32_t meta_u32_or(ds4_engine *e, const char *key, uint32_t fallback)
{
    uint32_t v = 0;
    if (ds4_qwen_meta_u32(e, key, &v) != 0) return fallback;
    return v;
}

static float meta_f32_or(ds4_engine *e, const char *key, float fallback)
{
    float v = 0.0f;
    if (ds4_qwen_meta_f32(e, key, &v) != 0) return fallback;
    return v;
}

static inline float qwen_silu(float x)
{
    return x / (1.0f + expf(-x));
}

static inline float qwen_sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static inline float qwen_softplus(float x)
{
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return logf(1.0f + expf(x));
}

static void rmsnorm(float *dst, const float *src, const float *w, uint32_t n, float eps)
{
    float ss = 0.0f;
    for (uint32_t i = 0; i < n; i++) ss += src[i] * src[i];
    const float inv = 1.0f / sqrtf(ss / (float)n + eps);
    if (w) {
        for (uint32_t i = 0; i < n; i++) dst[i] = src[i] * inv * w[i];
    } else {
        for (uint32_t i = 0; i < n; i++) dst[i] = src[i] * inv;
    }
}

static void l2norm_inplace(float *x, uint32_t n, float eps)
{
    float ss = 0.0f;
    for (uint32_t i = 0; i < n; i++) ss += x[i] * x[i];
    const float inv = 1.0f / fmaxf(sqrtf(ss), eps);
    for (uint32_t i = 0; i < n; i++) x[i] *= inv;
}

static void matvec_wt(const ds4_qwen_wt *w, const float *x, float *out)
{
    const uint64_t n_in = w->dim[0];
    const uint64_t n_out = w->ndim >= 2 ? w->dim[1] : 1;
    ds4_qwen_matvec(w->type, w->data, x, out, n_in, n_out);
}

static int embed_row(const ds4_qwen_wt *embd, int token, float *out)
{
    if (token < 0 || (uint64_t)token >= embd->dim[1]) return 1;
    const uint64_t n = embd->dim[0];
    const size_t rb = ds4_qwen_row_bytes(embd->type, n);
    const uint8_t *row = (const uint8_t *)embd->data + (size_t)token * rb;
    return ds4_qwen_dequant_row(embd->type, row, out, n);
}

static void rope_mrope(float *q, uint32_t n_heads, uint32_t head_dim, uint32_t n_rot,
                       uint32_t pos, const int32_t sections[4], float base)
{
    const uint32_t n_pairs = n_rot / 2u;
    uint32_t pos_ids[4];
    pos_ids[0] = pos;
    pos_ids[1] = pos;
    pos_ids[2] = pos;
    pos_ids[3] = pos;
    for (uint32_t h = 0; h < n_heads; h++) {
        float *v = q + (size_t)h * head_dim;
        uint32_t pair = 0;
        for (int sec = 0; sec < 4; sec++) {
            const uint32_t nsec = sections[sec] > 0 ? (uint32_t)sections[sec] : 0u;
            for (uint32_t p = 0; p < nsec && pair < n_pairs; p++, pair++) {
                const float freq = powf(base, -2.0f * (float)pair / (float)n_rot);
                const float ang = (float)pos_ids[sec] * freq;
                const float c = cosf(ang);
                const float s = sinf(ang);
                const float x0 = v[pair];
                const float x1 = v[pair + n_pairs];
                v[pair] = x0 * c - x1 * s;
                v[pair + n_pairs] = x0 * s + x1 * c;
            }
        }
    }
}

static void softmax_inplace(float *x, uint32_t n)
{
    float m = QWEN_NEG_INF;
    for (uint32_t i = 0; i < n; i++) if (x[i] > m) m = x[i];
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        x[i] = expf(x[i] - m);
        sum += x[i];
    }
    const float inv = 1.0f / sum;
    for (uint32_t i = 0; i < n; i++) x[i] *= inv;
}

static void ffn_swiglu(ds4_qwen_runtime *rt, const ds4_qwen_wt *gate,
                       const ds4_qwen_wt *up, const ds4_qwen_wt *down,
                       const float *x, float *out)
{
    matvec_wt(gate, x, rt->gate_up);
    matvec_wt(up, x, rt->ffn_mid);
    for (uint32_t i = 0; i < rt->n_ff; i++) {
        rt->ffn_mid[i] *= qwen_silu(rt->gate_up[i]);
    }
    matvec_wt(down, rt->ffn_mid, out);
}

static void gdn_step(ds4_qwen_runtime *rt, uint32_t il, const float *qkv_silu,
                     const float *beta, const float *g, float *out)
{
    const uint32_t sk = rt->d_state;
    const uint32_t nk = rt->n_k_head;
    const uint32_t nv = rt->n_v_head;
    const float scale = 1.0f / sqrtf((float)sk);
    const float *q_in = qkv_silu;
    const float *k_in = qkv_silu + nk * sk;
    const float *v_in = qkv_silu + 2u * nk * sk;
    float *Sbase = rt->recurrent + (size_t)il * nv * sk * sk;

    for (uint32_t vh = 0; vh < nv; vh++) {
        const uint32_t kh = vh % nk;
        float qh[128];
        float khv[128];
        memcpy(qh, q_in + kh * sk, sk * sizeof(float));
        memcpy(khv, k_in + kh * sk, sk * sizeof(float));
        l2norm_inplace(qh, sk, rt->rms_eps);
        l2norm_inplace(khv, sk, rt->rms_eps);
        const float *vhv = v_in + vh * sk;
        float *s_out = Sbase + (size_t)vh * sk * sk;
        const float gv = expf(g[vh]);
        for (uint32_t i = 0; i < sk * sk; i++) s_out[i] *= gv;
        float delta[128];
        for (uint32_t j = 0; j < sk; j++) {
            float sum = 0.0f;
            const float *row = s_out + (size_t)j * sk;
            for (uint32_t i = 0; i < sk; i++) sum += row[i] * khv[i];
            delta[j] = (vhv[j] - sum) * beta[vh];
        }
        for (uint32_t j = 0; j < sk; j++) {
            float *row = s_out + (size_t)j * sk;
            const float dj = delta[j];
            for (uint32_t i = 0; i < sk; i++) row[i] += dj * khv[i];
        }
        float *oh = out + vh * sk;
        for (uint32_t j = 0; j < sk; j++) {
            float sum = 0.0f;
            const float *row = s_out + (size_t)j * sk;
            for (uint32_t i = 0; i < sk; i++) sum += row[i] * qh[i];
            oh[j] = sum * scale;
        }
    }
}

static void conv1d_step(ds4_qwen_runtime *rt, uint32_t il, const float *x, float *y)
{
    const uint32_t k = rt->d_conv;
    const uint32_t c = rt->conv_dim;
    float *state = rt->conv_state + (size_t)il * (k - 1u) * c;
    const float *kernel = (const float *)rt->layers[il].u.linear.ssm_conv1d.data;
    for (uint32_t ch = 0; ch < c; ch++) {
        float acc = 0.0f;
        for (uint32_t t = 0; t + 1 < k; t++) {
            acc += state[(size_t)t * c + ch] * kernel[t + (size_t)ch * k];
        }
        acc += x[ch] * kernel[(k - 1u) + (size_t)ch * k];
        y[ch] = qwen_silu(acc);
    }
    if (k > 2) {
        memmove(state, state + c, (size_t)(k - 2u) * c * sizeof(float));
    }
    memcpy(state + (size_t)(k - 2u) * c, x, (size_t)c * sizeof(float));
}

static int eval_linear(ds4_qwen_runtime *rt, uint32_t il)
{
    const qwen_linear_w *w = &rt->layers[il].u.linear;
    const float *attn_norm = (const float *)w->attn_norm.data;
    const float *post_norm = (const float *)w->post_norm.data;
    rmsnorm(rt->xn, rt->x, attn_norm, rt->n_embd, rt->rms_eps);
    matvec_wt(&w->attn_qkv, rt->xn, rt->qkv);
    matvec_wt(&w->attn_gate, rt->xn, rt->z);
    float beta[64];
    float alpha[64];
    float g[64];
    matvec_wt(&w->ssm_beta, rt->xn, beta);
    matvec_wt(&w->ssm_alpha, rt->xn, alpha);
    const float *dt = (const float *)w->ssm_dt.data;
    const float *a = (const float *)w->ssm_a.data;
    for (uint32_t i = 0; i < rt->n_v_head; i++) {
        beta[i] = qwen_sigmoid(beta[i]);
        g[i] = a[i] * qwen_softplus(alpha[i] + dt[i]);
    }
    conv1d_step(rt, il, rt->qkv, rt->conv_y);
    gdn_step(rt, il, rt->conv_y, beta, g, rt->attn_out);
    const float *ssm_norm = (const float *)w->ssm_norm.data;
    for (uint32_t h = 0; h < rt->n_v_head; h++) {
        float *row = rt->attn_out + h * rt->d_state;
        const float *zrow = rt->z + h * rt->d_state;
        rmsnorm(row, row, ssm_norm, rt->d_state, rt->rms_eps);
        for (uint32_t i = 0; i < rt->d_state; i++) {
            row[i] *= qwen_silu(zrow[i]);
        }
    }
    matvec_wt(&w->ssm_out, rt->attn_out, rt->xn);
    for (uint32_t i = 0; i < rt->n_embd; i++) rt->x[i] += rt->xn[i];
    memcpy(rt->resid, rt->x, (size_t)rt->n_embd * sizeof(float));
    rmsnorm(rt->xn, rt->x, post_norm, rt->n_embd, rt->rms_eps);
    ffn_swiglu(rt, &w->ffn_gate, &w->ffn_up, &w->ffn_down, rt->xn, rt->attn_out);
    for (uint32_t i = 0; i < rt->n_embd; i++) rt->x[i] = rt->resid[i] + rt->attn_out[i];
    return 0;
}

static int eval_full(ds4_qwen_runtime *rt, uint32_t il)
{
    const qwen_full_w *w = &rt->layers[il].u.full;
    const float *attn_norm = (const float *)w->attn_norm.data;
    const float *post_norm = (const float *)w->post_norm.data;
    const float *q_norm = (const float *)w->attn_q_norm.data;
    const float *k_norm = (const float *)w->attn_k_norm.data;
    const uint32_t hd = rt->n_head_dim;
    const uint32_t nq = rt->n_head;
    const uint32_t nkv = rt->n_head_kv;
    const uint32_t pos = rt->pos;
    const uint32_t kv_slot = pos;
    if (kv_slot >= rt->ctx) return 1;

    rmsnorm(rt->xn, rt->x, attn_norm, rt->n_embd, rt->rms_eps);
    matvec_wt(&w->attn_q, rt->xn, rt->q);
    matvec_wt(&w->attn_k, rt->xn, rt->k);
    matvec_wt(&w->attn_v, rt->xn, rt->v);

    const uint32_t q_stride = hd * 2u;
    for (uint32_t h = 0; h < nq; h++) {
        float *qh = rt->q + (size_t)h * q_stride;
        rmsnorm(qh, qh, q_norm, hd, rt->rms_eps);
    }
    for (uint32_t h = 0; h < nkv; h++) {
        float *kh = rt->k + (size_t)h * hd;
        rmsnorm(kh, kh, k_norm, hd, rt->rms_eps);
    }

    float q_rot[24 * 256];
    float gate[24 * 256];
    for (uint32_t h = 0; h < nq; h++) {
        memcpy(q_rot + h * hd, rt->q + (size_t)h * q_stride, hd * sizeof(float));
        memcpy(gate + h * hd, rt->q + (size_t)h * q_stride + hd, hd * sizeof(float));
    }
    rope_mrope(q_rot, nq, hd, rt->n_rot, pos, rt->rope_sections, rt->rope_base);
    rope_mrope(rt->k, nkv, hd, rt->n_rot, pos, rt->rope_sections, rt->rope_base);

    float *k_cache = rt->kv_k + ((size_t)il * rt->ctx * nkv + (size_t)kv_slot * nkv) * hd;
    float *v_cache = rt->kv_v + ((size_t)il * rt->ctx * nkv + (size_t)kv_slot * nkv) * hd;
    memcpy(k_cache, rt->k, (size_t)nkv * hd * sizeof(float));
    memcpy(v_cache, rt->v, (size_t)nkv * hd * sizeof(float));

    const uint32_t tlen = pos + 1u;
    const float scale = 1.0f / sqrtf((float)hd);
    const uint32_t gqa = nq / nkv;
    memset(rt->attn_out, 0, (size_t)nq * hd * sizeof(float));
    for (uint32_t h = 0; h < nq; h++) {
        const uint32_t kvh = h / gqa;
        const float *qh = q_rot + h * hd;
        for (uint32_t t = 0; t < tlen; t++) {
            const float *kh = rt->kv_k + ((size_t)il * rt->ctx * nkv + (size_t)t * nkv + kvh) * hd;
            float dot = 0.0f;
            for (uint32_t i = 0; i < hd; i++) dot += qh[i] * kh[i];
            rt->scores[t] = dot * scale;
        }
        softmax_inplace(rt->scores, tlen);
        float *oh = rt->attn_out + h * hd;
        for (uint32_t t = 0; t < tlen; t++) {
            const float *vh = rt->kv_v + ((size_t)il * rt->ctx * nkv + (size_t)t * nkv + kvh) * hd;
            const float a = rt->scores[t];
            for (uint32_t i = 0; i < hd; i++) oh[i] += a * vh[i];
        }
        for (uint32_t i = 0; i < hd; i++) {
            oh[i] *= qwen_sigmoid(gate[h * hd + i]);
        }
    }

    matvec_wt(&w->attn_output, rt->attn_out, rt->xn);
    for (uint32_t i = 0; i < rt->n_embd; i++) rt->x[i] += rt->xn[i];
    memcpy(rt->resid, rt->x, (size_t)rt->n_embd * sizeof(float));
    rmsnorm(rt->xn, rt->x, post_norm, rt->n_embd, rt->rms_eps);
    ffn_swiglu(rt, &w->ffn_gate, &w->ffn_up, &w->ffn_down, rt->xn, rt->attn_out);
    for (uint32_t i = 0; i < rt->n_embd; i++) rt->x[i] = rt->resid[i] + rt->attn_out[i];
    return 0;
}

static int eval_layers(ds4_qwen_runtime *rt)
{
    for (uint32_t il = 0; il < rt->n_layer; il++) {
        if (rt->layers[il].kind == QWEN_LAYER_LINEAR) {
            if (eval_linear(rt, il) != 0) return 1;
        } else {
            if (eval_full(rt, il) != 0) return 1;
        }
    }
    return 0;
}

bool ds4_gguf_is_qwen35(const char *path)
{
    if (!path) return false;
    int fd = open(path, O_RDONLY);
    if (fd < 0) return false;
    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return false;
    }
    const size_t cap = st.st_size < (off_t)(64 * 1024 * 1024) ? (size_t)st.st_size : (size_t)(64 * 1024 * 1024);
    void *map = mmap(NULL, cap, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) return false;
    const uint8_t *p = (const uint8_t *)map;
    size_t off = 0;
    bool ok = false;
    if (cap < 24) goto done;
    uint32_t magic = 0, version = 0;
    memcpy(&magic, p + off, 4); off += 4;
    memcpy(&version, p + off, 4); off += 4;
    uint64_t n_tensors = 0, n_kv = 0;
    memcpy(&n_tensors, p + off, 8); off += 8;
    memcpy(&n_kv, p + off, 8); off += 8;
    if (magic != 0x46554747u || version != 3) goto done;
    for (uint64_t i = 0; i < n_kv && off + 12 < cap; i++) {
        uint64_t klen = 0;
        memcpy(&klen, p + off, 8); off += 8;
        if (off + klen + 4 > cap) break;
        const char *key = (const char *)(p + off);
        off += (size_t)klen;
        uint32_t typ = 0;
        memcpy(&typ, p + off, 4); off += 4;
        if (klen == 21 && memcmp(key, "general.architecture", 21) == 0 && typ == 8) {
            uint64_t vlen = 0;
            if (off + 8 > cap) break;
            memcpy(&vlen, p + off, 8); off += 8;
            if (off + vlen > cap) break;
            ok = (vlen == 6 && memcmp(p + off, "qwen35", 6) == 0);
            break;
        }
        switch (typ) {
        case 0: case 1: case 7: off += 1; break;
        case 2: case 3: off += 2; break;
        case 4: case 5: case 6: off += 4; break;
        case 10: case 11: case 12: off += 8; break;
        case 8: {
            uint64_t n = 0;
            if (off + 8 > cap) goto done;
            memcpy(&n, p + off, 8); off += 8 + (size_t)n;
            break;
        }
        case 9: {
            uint32_t at = 0;
            uint64_t alen = 0;
            if (off + 12 > cap) goto done;
            memcpy(&at, p + off, 4); off += 4;
            memcpy(&alen, p + off, 8); off += 8;
            uint64_t esz = 0;
            switch (at) {
            case 0: case 1: case 7: esz = 1; break;
            case 2: case 3: esz = 2; break;
            case 4: case 5: case 6: esz = 4; break;
            case 10: case 11: case 12: esz = 8; break;
            case 8:
                for (uint64_t a = 0; a < alen && off + 8 < cap; a++) {
                    uint64_t n = 0;
                    memcpy(&n, p + off, 8);
                    off += 8 + (size_t)n;
                }
                esz = 0;
                break;
            default:
                goto done;
            }
            if (esz) off += (size_t)(alen * esz);
            break;
        }
        default:
            goto done;
        }
    }
done:
    munmap(map, cap);
    return ok;
}

int ds4_qwen_runtime_open(ds4_qwen_runtime **out, ds4_engine *e, uint32_t ctx)
{
    if (!out || !e) return 1;
    ds4_qwen_runtime *rt = calloc(1, sizeof(*rt));
    if (!rt) return 1;
    rt->n_embd = meta_u32_or(e, "qwen35.embedding_length", 5120);
    rt->n_ff = meta_u32_or(e, "qwen35.feed_forward_length", 17408);
    uint32_t n_block = meta_u32_or(e, "qwen35.block_count", 65);
    uint32_t n_nextn = meta_u32_or(e, "qwen35.nextn_predict_layers", 1);
    rt->n_layer = n_block > n_nextn ? n_block - n_nextn : n_block;
    rt->n_head = meta_u32_or(e, "qwen35.attention.head_count", 24);
    rt->n_head_kv = meta_u32_or(e, "qwen35.attention.head_count_kv", 4);
    rt->n_head_dim = meta_u32_or(e, "qwen35.attention.key_length", 256);
    rt->n_rot = meta_u32_or(e, "qwen35.rope.dimension_count", 64);
    rt->interval = meta_u32_or(e, "qwen35.full_attention_interval", 4);
    rt->d_conv = meta_u32_or(e, "qwen35.ssm.conv_kernel", 4);
    rt->d_state = meta_u32_or(e, "qwen35.ssm.state_size", 128);
    rt->n_k_head = meta_u32_or(e, "qwen35.ssm.group_count", 16);
    rt->n_v_head = meta_u32_or(e, "qwen35.ssm.time_step_rank", 48);
    rt->conv_dim = rt->n_k_head * rt->d_state * 2u + rt->n_v_head * rt->d_state;
    rt->rms_eps = meta_f32_or(e, "qwen35.attention.layer_norm_rms_epsilon", 1.0e-6f);
    rt->rope_base = meta_f32_or(e, "qwen35.rope.freq_base", 10000000.0f);
    rt->rope_sections[0] = 11;
    rt->rope_sections[1] = 11;
    rt->rope_sections[2] = 10;
    rt->rope_sections[3] = 0;
    if (ctx == 0 || ctx > 4096) ctx = 512;
    rt->ctx = ctx;
    if (rt->n_layer > QWEN_MAX_BLOCKS || rt->d_state > 128 || rt->n_v_head > 64) {
        free(rt);
        return 1;
    }
    if (req_wt(e, "token_embd.weight", &rt->token_embd) != 0 ||
        req_wt(e, "output.weight", &rt->output) != 0 ||
        req_wt(e, "output_norm.weight", &rt->output_norm) != 0) {
        free(rt);
        return 1;
    }
    rt->n_vocab = (uint32_t)rt->token_embd.dim[1];

    uint32_t n_linear = 0, n_full = 0;
    for (uint32_t il = 0; il < rt->n_layer; il++) {
        qwen_layer *L = &rt->layers[il];
        const int is_full = ((il + 1u) % rt->interval) == 0;
        L->kind = is_full ? QWEN_LAYER_FULL : QWEN_LAYER_LINEAR;
        if (is_full) {
            n_full++;
            if (req_wt_fmt(e, &L->u.full.attn_norm, "blk.%u.attn_norm.weight", il) ||
                req_wt_fmt(e, &L->u.full.post_norm, "blk.%u.post_attention_norm.weight", il) ||
                req_wt_fmt(e, &L->u.full.ffn_gate, "blk.%u.ffn_gate.weight", il) ||
                req_wt_fmt(e, &L->u.full.ffn_up, "blk.%u.ffn_up.weight", il) ||
                req_wt_fmt(e, &L->u.full.ffn_down, "blk.%u.ffn_down.weight", il) ||
                req_wt_fmt(e, &L->u.full.attn_q, "blk.%u.attn_q.weight", il) ||
                req_wt_fmt(e, &L->u.full.attn_k, "blk.%u.attn_k.weight", il) ||
                req_wt_fmt(e, &L->u.full.attn_v, "blk.%u.attn_v.weight", il) ||
                req_wt_fmt(e, &L->u.full.attn_output, "blk.%u.attn_output.weight", il) ||
                req_wt_fmt(e, &L->u.full.attn_q_norm, "blk.%u.attn_q_norm.weight", il) ||
                req_wt_fmt(e, &L->u.full.attn_k_norm, "blk.%u.attn_k_norm.weight", il)) {
                free(rt);
                return 1;
            }
        } else {
            n_linear++;
            if (req_wt_fmt(e, &L->u.linear.attn_norm, "blk.%u.attn_norm.weight", il) ||
                req_wt_fmt(e, &L->u.linear.post_norm, "blk.%u.post_attention_norm.weight", il) ||
                req_wt_fmt(e, &L->u.linear.ffn_gate, "blk.%u.ffn_gate.weight", il) ||
                req_wt_fmt(e, &L->u.linear.ffn_up, "blk.%u.ffn_up.weight", il) ||
                req_wt_fmt(e, &L->u.linear.ffn_down, "blk.%u.ffn_down.weight", il) ||
                req_wt_fmt(e, &L->u.linear.attn_qkv, "blk.%u.attn_qkv.weight", il) ||
                req_wt_fmt(e, &L->u.linear.attn_gate, "blk.%u.attn_gate.weight", il) ||
                req_wt_fmt(e, &L->u.linear.ssm_conv1d, "blk.%u.ssm_conv1d.weight", il) ||
                req_wt_fmt(e, &L->u.linear.ssm_dt, "blk.%u.ssm_dt.bias", il) ||
                req_wt_fmt(e, &L->u.linear.ssm_a, "blk.%u.ssm_a", il) ||
                req_wt_fmt(e, &L->u.linear.ssm_alpha, "blk.%u.ssm_alpha.weight", il) ||
                req_wt_fmt(e, &L->u.linear.ssm_beta, "blk.%u.ssm_beta.weight", il) ||
                req_wt_fmt(e, &L->u.linear.ssm_norm, "blk.%u.ssm_norm.weight", il) ||
                req_wt_fmt(e, &L->u.linear.ssm_out, "blk.%u.ssm_out.weight", il)) {
                free(rt);
                return 1;
            }
        }
    }

    const size_t emb = rt->n_embd;
    const size_t ff = rt->n_ff;
    const size_t qdim = (size_t)rt->n_head * rt->n_head_dim * 2u;
    const size_t kvdim = (size_t)rt->n_head_kv * rt->n_head_dim;
    const size_t inner = (size_t)rt->n_v_head * rt->d_state;
    rt->x = calloc(emb, sizeof(float));
    rt->xn = calloc(emb, sizeof(float));
    rt->resid = calloc(emb, sizeof(float));
    rt->qkv = calloc(rt->conv_dim, sizeof(float));
    rt->conv_y = calloc(rt->conv_dim, sizeof(float));
    rt->z = calloc(inner, sizeof(float));
    rt->gate_up = calloc(ff, sizeof(float));
    rt->ffn_mid = calloc(ff, sizeof(float));
    rt->q = calloc(qdim > inner ? qdim : inner, sizeof(float));
    rt->k = calloc(kvdim, sizeof(float));
    rt->v = calloc(kvdim, sizeof(float));
    rt->attn_out = calloc(qdim > inner ? qdim : (inner > emb ? inner : emb), sizeof(float));
    rt->scores = calloc(rt->ctx, sizeof(float));
    rt->conv_state = calloc((size_t)rt->n_layer * (rt->d_conv - 1u) * rt->conv_dim, sizeof(float));
    rt->recurrent = calloc((size_t)rt->n_layer * rt->n_v_head * rt->d_state * rt->d_state, sizeof(float));
    rt->kv_k = calloc((size_t)rt->n_layer * rt->ctx * kvdim, sizeof(float));
    rt->kv_v = calloc((size_t)rt->n_layer * rt->ctx * kvdim, sizeof(float));
    if (!rt->x || !rt->xn || !rt->resid || !rt->qkv || !rt->conv_y || !rt->z ||
        !rt->gate_up || !rt->ffn_mid || !rt->q || !rt->k || !rt->v || !rt->attn_out ||
        !rt->scores || !rt->conv_state || !rt->recurrent || !rt->kv_k || !rt->kv_v) {
        ds4_qwen_runtime_free(rt);
        return 1;
    }

    fprintf(stderr,
            "ds4: Qwen 3.8 27B CPU stream  layers=%u linear=%u full=%u ctx=%u  "
            "working set is mmap plus two-layer activations\n",
            rt->n_layer, n_linear, n_full, rt->ctx);
    {
        const ds4_qwen_wt *probe = NULL;
        for (uint32_t il = 0; il < rt->n_layer && !probe; il++) {
            if (rt->layers[il].kind == QWEN_LAYER_LINEAR) {
                probe = &rt->layers[il].u.linear.ffn_down;
            }
        }
        if (probe && ds4_qwen_matvec_selftest(probe->type, probe->data,
                                              probe->dim[0], probe->dim[1]) != 0) {
            fprintf(stderr, "ds4: qwen35 matvec selftest failed\n");
            ds4_qwen_runtime_free(rt);
            return 1;
        }
    }
    *out = rt;
    return 0;
}

void ds4_qwen_runtime_free(ds4_qwen_runtime *rt)
{
    if (!rt) return;
    free(rt->x);
    free(rt->xn);
    free(rt->resid);
    free(rt->qkv);
    free(rt->conv_y);
    free(rt->z);
    free(rt->gate_up);
    free(rt->ffn_mid);
    free(rt->q);
    free(rt->k);
    free(rt->v);
    free(rt->attn_out);
    free(rt->scores);
    free(rt->conv_state);
    free(rt->recurrent);
    free(rt->kv_k);
    free(rt->kv_v);
    free(rt);
}

void ds4_qwen_runtime_reset(ds4_qwen_runtime *rt)
{
    if (!rt) return;
    rt->pos = 0;
    const size_t kvdim = (size_t)rt->n_head_kv * rt->n_head_dim;
    memset(rt->conv_state, 0, (size_t)rt->n_layer * (rt->d_conv - 1u) * rt->conv_dim * sizeof(float));
    memset(rt->recurrent, 0, (size_t)rt->n_layer * rt->n_v_head * rt->d_state * rt->d_state * sizeof(float));
    memset(rt->kv_k, 0, (size_t)rt->n_layer * rt->ctx * kvdim * sizeof(float));
    memset(rt->kv_v, 0, (size_t)rt->n_layer * rt->ctx * kvdim * sizeof(float));
}

int ds4_qwen_eval_token(ds4_qwen_runtime *rt, int token, float *logits)
{
    if (!rt) return 1;
    if (rt->pos >= rt->ctx) return 1;
    if (embed_row(&rt->token_embd, token, rt->x) != 0) return 1;
    fprintf(stderr, "ds4: qwen stream pos=%u/%u\n", rt->pos, rt->ctx);
    if (eval_layers(rt) != 0) return 1;
    if (logits) {
        const float *on = (const float *)rt->output_norm.data;
        rmsnorm(rt->xn, rt->x, on, rt->n_embd, rt->rms_eps);
        matvec_wt(&rt->output, rt->xn, logits);
    }
    rt->pos++;
    return 0;
}
