#ifndef __HLS_ATTENTION_H__
#define __HLS_ATTENTION_H__

typedef float data_t;
const int SEQ = 20;
const int DIM = 512;
const int HEAD_SIZE = 8;
void project_top(data_t input[SEQ][DIM], data_t output[SEQ][DIM]);

#endif
