#ifndef __HLS_ATTENTION_H__
#define __HLS_ATTENTION_H__

#include <cmath>
#include "linear.h"
#include "dropout.h"
#include "softmax.h"

template<typename T, int DIM, int HEAD_SIZE>
struct MultiHeadAttentionParameter {
	T dr = 0.1;
};
template<typename T, int DIM, int SEQ>
void scaleDotSelfAttentionForward(T (&Q)[SEQ][DIM], T (&K)[SEQ][DIM],
		T (&V)[SEQ][DIM], T (&output)[SEQ][DIM], T scale, T dr) {
#pragma HLS DATAFLOW
	T Q_pl[SEQ][DIM];
	T K_pl[SEQ][DIM];
	T V_pl[SEQ][DIM];
	T output_pl[SEQ][DIM];
	for (int i = 0; i < SEQ; ++i) {
		for (int j = 0; j < DIM; ++j) {
			Q_pl[i][j] = Q[i][j];
			K_pl[i][j] = K[i][j];
			V_pl[i][j] = V[i][j];
		}
	}
	T q_tmp[SEQ][DIM];
	T k_tmp[SEQ][DIM];
	T v_tmp[SEQ][DIM];
#pragma HLS ARRAY_PARTITION variable=v_tmp dim=1 complete
	T q_tmp_1[SEQ][DIM];

	linearForward<T, DIM, DIM, SEQ>(Q_pl, q_tmp);
	linearForward<T, DIM, DIM, SEQ>(K_pl, k_tmp);
	linearForward<T, DIM, DIM, SEQ>(V_pl, v_tmp);
	SDSAF_LOOP0: for (int i = 0; i < SEQ; ++i) {
		dropoutForward<T, DIM>(q_tmp[i], q_tmp_1[i], dr);
		SDSAF_LOOP1: for (int j = 0; j < DIM; ++j) {
			q_tmp_1[i][j] *= scale;
		}
	}

	T nex_tmp[SEQ][SEQ];
#pragma HLS ARRAY_PARTITION variable=nex_tmp dim=1 complete
	SDSAF_LOOP2: for (int i = 0; i < SEQ; ++i) {
#pragma HLS PIPELINE
		SDSAF_LOOP3: for (int j = 0; j < SEQ; ++j) {
			nex_tmp[i][j] = 0;
			SDSAF_LOOP4: for (int k = 0; k < DIM; ++k) {
				nex_tmp[i][j] += q_tmp_1[i][k] * k_tmp[j][k];
			}
		}
	}
	T nex_tmp_2[SEQ][SEQ];
#pragma HLS ARRAY_PARTITION variable=nex_tmp_2 dim=2 complete
	softmaxForward<T, SEQ, SEQ>(nex_tmp, nex_tmp_2);
	SDSAF_LOOP5: for (int i = 0; i < SEQ; ++i) {
		SDSAF_LOOP6: for (int j = 0; j < DIM; ++j) {
			output_pl[i][j] = 0;
			SDSAF_LOOP7: for (int k = 0; k < SEQ; ++k) {
#pragma HLS UNROLL
				output_pl[i][j] += nex_tmp_2[i][k] * v_tmp[k][j];
			}
		}
	}
	for (int i = 0; i < SEQ; ++i) {
		for (int j = 0; j < DIM; ++j) {
			output[i][j] = output_pl[i][j];
		}
	}
}
template<typename T, int DIM, int SEQ, int HEAD_SIZE>
void multiHeadAttentionForward(T (&Q)[SEQ][DIM], T (&K)[SEQ][DIM],
		T (&V)[SEQ][DIM], T (&output)[SEQ][DIM], T dr) {
	T scale = 1.0 / sqrt((double) DIM * 1.0 / HEAD_SIZE);
	T tmp[HEAD_SIZE][SEQ][DIM];
#pragma HLS ARRAY_PARTITION variable=tmp dim=1 complete
#pragma HLS ARRAY_PARTITION variable=tmp dim=2 complete
	T tmp2[HEAD_SIZE][SEQ][DIM];
#pragma HLS ARRAY_PARTITION variable=tmp2 dim=2 complete
#pragma HLS ARRAY_PARTITION variable=tmp2 dim=1 complete
	T result[SEQ][DIM];
#pragma HLS ARRAY_PARTITION variable=result dim=1 complete
	for (int i = 0; i < SEQ; ++i) {
		MHAF_LOOP1: for (int j = 0; j < DIM; ++j) {
#pragma HLS UNROLL
			result[i][j] = 0;
		}
	}
	MHAF_LOOP0: for (int h = 0; h < HEAD_SIZE; ++h) {
		scaleDotSelfAttentionForward<T, DIM, SEQ>(Q, K, V, tmp[h], scale, dr);
		linearForward<T, DIM, DIM, SEQ>(tmp[h], tmp2[h]);
	}
	//scaleDotSelfAttentionForward<T, DIM, SEQ, HEAD_SIZE>(Q, K, V, tmp, scale, dr);
	for (int i = 0; i < SEQ; ++i) {
		for (int j = 0; j < DIM; ++j) {
			for (int h = 0; h < HEAD_SIZE; ++h) {
				result[i][j] += tmp2[h][i][j];
			}
		}
	}
	for (int i = 0; i < SEQ; ++i) {
		for (int j = 0; j < DIM; ++j) {
			output[i][j] = result[i][j];
		}
	}
}
#endif
