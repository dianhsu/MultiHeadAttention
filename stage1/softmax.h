#ifndef __HLS_SOFTMAX_H__
#define __HLS_SOFTMAX_H__

template<typename T, int DIM, int SEQ>
void softmaxForward(T (&input)[SEQ][DIM], T (&output)[SEQ][DIM]) {
    T input_pl[SEQ][DIM];
#pragma HLS ARRAY_PARTITION variable=input_pl dim=1 complete
    T output_pl[SEQ][DIM];
#pragma HLS ARRAY_PARTITION variable=output_pl dim=1 complete
    for (int i = 0; i < SEQ; ++i) {
        for (int j = 0; j < DIM; ++j) {
            input_pl[i][j] = input[i][j];
        }
    }
    T tmp[DIM];
    for (int j = 0; j < DIM; ++j) {
#pragma HLS UNROLL
        tmp[j] = 0;
    }
    for (int i = 0; i < SEQ; ++i) {
#pragma HLS PIPELINE off        
        for (int j = 0; j < DIM; ++j) {
#pragma HLS UNROLL
            tmp[j] += input_pl[i][j];
        }
    }
    for(int j = 0; j < DIM; ++j) {
#pragma HLS UNROLL
        tmp[j] = 1/tmp[j];
    }
    for (int i = 0; i < SEQ; ++i) {
#pragma HLS PIPELINE off        
        for (int j = 0; j < DIM; ++j) {
#pragma HLS UNROLL
            output_pl[i][j] = input_pl[i][j] * tmp[j];
        }
    }
    for (int i = 0; i < SEQ; ++i) {        
        for (int j = 0; j < DIM; ++j) {
            output[i][j] = output_pl[i][j];
        }
    }
}

#endif
