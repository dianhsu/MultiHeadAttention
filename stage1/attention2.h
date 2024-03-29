#ifndef __HLS_ATTENTION2_H__
#define __HLS_ATTENTION2_H__

#include "linear.h"
#include "softmax.h"

template<typename T, int DIM, int SEQ>
void scaledDotProductAttention(T (&Q)[SEQ][DIM], T (&K)[SEQ][DIM], T (&V)[SEQ][DIM], T (&output)[SEQ][DIM]) {
    //T scale;
    T Q_pl[SEQ][DIM];
#pragma HLS ARRAY_PARTITION variable=Q_pl dim=1 complete
    T K_pl[SEQ][DIM];
#pragma HLS ARRAY_PARTITION variable=K_pl dim=1 complete
    T V_pl[SEQ][DIM];
#pragma HLS ARRAY_PARTITION variable=V_pl dim=1 complete
    T output_pl[SEQ][DIM];
#pragma HLS ARRAY_PARTITION variable=output_pl dim=1 complete

    for(int i = 0; i < SEQ; ++i) {
        for(int j = 0; j < DIM; ++j) {
            Q_pl[i][j] = Q[i][j];
            K_pl[i][j] = K[i][j];
            V_pl[i][j] = V[i][j];
        }
    }

    T q_tmp[SEQ][DIM];
#pragma HLS ARRAY_PARTITION variable=q_tmp dim=1 complete
    T k_tmp[SEQ][DIM];
#pragma HLS ARRAY_PARTITION variable=k_tmp dim=1 complete
    T v_tmp[SEQ][DIM];
#pragma HLS ARRAY_PARTITION variable=v_tmp dim=1 complete
    T q_tmp1[SEQ][DIM];
#pragma HLS ARRAY_PARTITION variable=q_tmp1 dim=1 complete
    T nex_tmp[SEQ][SEQ];
#pragma HLS ARRAY_PARTITION variable=nex_tmp dim=0 complete
    T nex_tmp2[SEQ][SEQ];
#pragma HLS ARRAY_PARTITION variable=nex_tmp2 dim=1 complete
    linearForward<T, DIM, DIM, SEQ>(Q_pl, q_tmp);
    linearForward<T, DIM, DIM, SEQ>(K_pl, k_tmp);
    linearForward<T, DIM, DIM, SEQ>(V_pl, v_tmp);
    // for(int i = 0; i < SEQ; ++i){
    //     for(int j = 0; j < DIM; ++j){
    //         q_tmp1[i][j] = q_tmp[i][j] * scale;
    //     }
    // }
    for(int i = 0; i < SEQ; ++i) {
#pragma HLS UNROLL
        for(int j = 0; j < SEQ; ++j) {
            nex_tmp[i][j] = 0;
        }
    }

    for(int i = 0; i < SEQ; ++i) {
#pragma HLS UNROLL
        for(int j = 0; j < SEQ; ++j) {
#pragma HLS UNROLL
            for(int k = 0; k < DIM; ++k) {
#pragma HLS PIPELINE off
                nex_tmp[i][j] += q_tmp1[i][k] * k_tmp[j][k];
            }
        }
    }
    softmaxForward<T, SEQ, SEQ>(nex_tmp, nex_tmp2);
    for(int i = 0; i < SEQ; ++i) {
#pragma HLS UNROLL
        for(int j = 0; j < DIM; ++j) {
#pragma HLS UNROLL
            output_pl[i][j] = 0;
        }
    }
    for(int i = 0; i < SEQ; ++i) {
#pragma HLS UNROLL
        for(int j = 0; j < DIM; ++j) {
#pragma HLS UNROLL
            for(int k = 0; k < SEQ; ++k) {
#pragma HLS PIPELINE off
                output_pl[i][j] += nex_tmp2[i][k] * v_tmp[k][j];
            }
        }
    }
    for(int i = 0; i < SEQ; ++i) {
        for(int j = 0; j < DIM; ++j) {
            output[i][j] = output_pl[i][j];
        }
    }
}


#endif