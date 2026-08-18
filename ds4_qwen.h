#ifndef DS4_QWEN_H
#define DS4_QWEN_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#include "ds4.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DS4_QWEN_MAX_DIMS 8

typedef struct {
    const void *data;
    uint32_t type;
    uint32_t ndim;
    uint64_t dim[DS4_QWEN_MAX_DIMS];
    uint64_t bytes;
} ds4_qwen_wt;

typedef struct ds4_qwen_runtime ds4_qwen_runtime;

bool ds4_gguf_is_qwen35(const char *path);

int ds4_qwen_weight_lookup(const ds4_engine *e, const char *name, ds4_qwen_wt *out);
int ds4_qwen_meta_u32(const ds4_engine *e, const char *key, uint32_t *out);
int ds4_qwen_meta_f32(const ds4_engine *e, const char *key, float *out);

int ds4_qwen_runtime_open(ds4_qwen_runtime **out, ds4_engine *e, uint32_t ctx);
void ds4_qwen_runtime_free(ds4_qwen_runtime *rt);
void ds4_qwen_runtime_reset(ds4_qwen_runtime *rt);

int ds4_qwen_eval_token(ds4_qwen_runtime *rt, int token, float *logits);

#ifdef __cplusplus
}
#endif

#endif
