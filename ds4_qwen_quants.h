#ifndef DS4_QWEN_QUANTS_H
#define DS4_QWEN_QUANTS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    DS4_QWEN_F32  = 0,
    DS4_QWEN_Q8_0 = 8,
    DS4_QWEN_Q3_K = 11,
    DS4_QWEN_Q4_K = 12,
    DS4_QWEN_Q5_K = 13,
    DS4_QWEN_Q6_K = 14,
};

size_t ds4_qwen_row_bytes(uint32_t type, uint64_t n_elem);
int ds4_qwen_dequant_row(uint32_t type, const void *row, float *out, uint64_t n_elem);
void ds4_qwen_matvec(uint32_t type, const void *weight, const float *x,
                     float *out, uint64_t n_in, uint64_t n_out);
int ds4_qwen_matvec_selftest(uint32_t type, const void *weight,
                             uint64_t n_in, uint64_t n_out);

#ifdef __cplusplus
}
#endif

#endif
