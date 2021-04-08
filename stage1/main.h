#ifndef __HLS_MAIN_H__
#define __HLS_MAIN_H__
#include "ap_fixed.h"
//typedef ap_fixed<16, 8, AP_RND, AP_SAT> data_t;
//typedef float data_t;
typedef ap_fixed<8, 3, AP_RND, AP_SAT> data_t;

const int SEQ = 20;
const int DIM = 32;
const int HEAD_SIZE = 8;
void project_top(data_t input[SEQ][DIM], data_t output[SEQ][DIM]);

#endif
