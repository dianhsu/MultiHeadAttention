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
	T q_tmp[SEQ][DIM];
	T k_tmp[SEQ][DIM];
	T v_tmp[SEQ][DIM];
	T q_tmp_1[SEQ][DIM];
	SDSAF_BLOCK0: {
		linearForward<T, DIM, DIM, SEQ>(Q, q_tmp);
		linearForward<T, DIM, DIM, SEQ>(K, k_tmp);
		linearForward<T, DIM, DIM, SEQ>(V, v_tmp);
		SDSAF_LOOP0: for (int i = 0; i < SEQ; ++i) {
#pragma HLS PIPELINE off
			dropoutForward<T, DIM>(q_tmp[i], q_tmp_1[i], dr);
			SDSAF_LOOP1: for (int j = 0; j < DIM; ++j) {
				q_tmp_1[i][j] *= scale;
			}
		}
	}

	T nex_tmp[SEQ][SEQ];
	SDSAF_LOOP2: for (int i = 0; i < SEQ; ++i) {
#pragma HLS PIPELINE off
		SDSAF_LOOP3: for (int j = 0; j < SEQ; ++j) {
#pragma HLS PIPELINE off
			nex_tmp[i][j] = 0;
			SDSAF_LOOP4: for (int k = 0; k < DIM; ++k) {
#pragma HLS PIPELINE off
				nex_tmp[i][j] += q_tmp_1[i][k] * k_tmp[j][k];
			}
		}
	}
	T nex_tmp_2[SEQ][SEQ];
	softmaxForward<T, SEQ, SEQ>(nex_tmp, nex_tmp_2);
	SDSAF_LOOP5: for (int i = 0; i < SEQ; ++i) {
#pragma HLS PIPELINE off
		SDSAF_LOOP6: for (int j = 0; j < DIM; ++j) {
#pragma HLS PIPELINE off
			output[i][j] = 0;
			SDSAF_LOOP7: for (int k = 0; k < SEQ; ++k) {
#pragma HLS PIPELINE off
				output[i][j] += nex_tmp_2[i][k] * v_tmp[k][j];
			}
		}
	}
}
template<typename T, int DIM, int SEQ, int HEAD_SIZE>
void multiHeadAttentionForward(T (&Q)[SEQ][DIM], T (&K)[SEQ][DIM],
		T (&V)[SEQ][DIM], T (&output)[SEQ][DIM],
		T dr) {
	T scale = 1.0 / sqrt((double) DIM * 1.0 / HEAD_SIZE);
	T tmp[HEAD_SIZE][SEQ][DIM];

	MHAF_LOOP0: for (int h = 0; h < HEAD_SIZE; ++h) {
#pragma HLS PIPELINE off
		scaleDotSelfAttentionForward<T, DIM, SEQ>(Q, K, V, tmp[h], scale, dr);
	}
	T fc_tmp[SEQ][DIM * HEAD_SIZE];
	MHAF_LOOP1: for (int h = 0; h < HEAD_SIZE; ++h) {
#pragma HLS PIPELINE off
		MHAF_LOOP2: for (int i = 0; i < SEQ; ++i) {
#pragma HLS PIPELINE off
			MHAF_LOOP3: for (int j = 0; j < DIM; ++j) {
#pragma HLS PIPELINE off
				fc_tmp[i][h * HEAD_SIZE + j] = tmp[h][i][j];
			}
		}
	}
	linearForward<T, DIM * HEAD_SIZE, DIM, SEQ>(fc_tmp, output);
}
#endif
