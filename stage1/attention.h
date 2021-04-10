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
template<typename T, int DIM, int SEQ, int HEAD_SIZE>
void scaleDotSelfAttentionForward(T (&Q)[SEQ][DIM], T (&K)[SEQ][DIM],
		T (&V)[SEQ][DIM], T (&output)[HEAD_SIZE][SEQ][DIM], T scale, T dr) {
	T Q_pl[SEQ][DIM];
#pragma HLS ARRAY_PARTITION variable=Q_pl dim=1 complete
	T K_pl[SEQ][DIM];
#pragma HLS ARRAY_PARTITION variable=K_pl dim=1 complete
	T V_pl[SEQ][DIM];
#pragma HLS ARRAY_PARTITION variable=V_pl dim=1 complete
	T output_pl[HEAD_SIZE][SEQ][DIM];
#pragma HLS ARRAY_PARTITION variable=output_pl dim=1 complete        
#pragma HLS ARRAY_PARTITION variable=output_pl dim=2 complete
	for (int i = 0; i < SEQ; ++i) {
		for (int j = 0; j < DIM; ++j) {
			Q_pl[i][j] = Q[i][j];
			K_pl[i][j] = K[i][j];
			V_pl[i][j] = V[i][j];
		}
	}
	T q_tmp[HEAD_SIZE][SEQ][DIM];
#pragma HLS ARRAY_PARTITION variable=q_tmp dim=1 complete
#pragma HLS ARRAY_PARTITION variable=q_tmp dim=2 complete
	T k_tmp[HEAD_SIZE][SEQ][DIM];
#pragma HLS ARRAY_PARTITION variable=k_tmp dim=1 complete
#pragma HLS ARRAY_PARTITION variable=k_tmp dim=2 complete
	T v_tmp[HEAD_SIZE][SEQ][DIM];
#pragma HLS ARRAY_PARTITION variable=v_tmp dim=1 complete
#pragma HLS ARRAY_PARTITION variable=v_tmp dim=2 complete
	T q_tmp_1[HEAD_SIZE][SEQ][DIM];
#pragma HLS ARRAY_PARTITION variable=q_tmp_1 dim=1 complete
#pragma HLS ARRAY_PARTITION variable=q_tmp_1 dim=2 complete
	T nex_tmp[HEAD_SIZE][SEQ][SEQ];
#pragma HLS ARRAY_PARTITION variable=nex_tmp dim=0 complete
T nex_tmp_2[HEAD_SIZE][SEQ][SEQ];
#pragma HLS ARRAY_PARTITION variable=nex_tmp_2 dim=2 complete
#pragma HLS ARRAY_PARTITION variable=nex_tmp_2 dim=1 complete
	for (int h = 0; h < HEAD_SIZE; ++h) {
		linearForward<T, DIM, DIM, SEQ>(Q_pl, q_tmp[h]);
		linearForward<T, DIM, DIM, SEQ>(K_pl, k_tmp[h]);
		linearForward<T, DIM, DIM, SEQ>(V_pl, v_tmp[h]);
		for (int i = 0; i < SEQ; ++i) {
			for (int j = 0; j < DIM; ++j) {
				q_tmp_1[h][i][j] = q_tmp[h][i][j] * scale;
			}
		}
		for (int i = 0; i < SEQ; ++i) {
			for (int j = 0; j < SEQ; ++j) {
				nex_tmp[h][i][j] = 0;
			}
		}
		for (int i = 0; i < SEQ; ++i) {
			for (int j = 0; j < SEQ; ++j) {
				for (int k = 0; k < DIM; ++k) {
					nex_tmp[h][i][j] += q_tmp_1[h][i][k] * k_tmp[h][j][k];
				}
			}
		}
		softmaxForward<T, SEQ, SEQ>(nex_tmp[h], nex_tmp_2[h]);
		for (int i = 0; i < SEQ; ++i) {
			for (int j = 0; j < DIM; ++j) {
				output_pl[h][i][j] = 0;
				for (int k = 0; k < SEQ; ++k) {
#pragma HLS UNROLL
					output_pl[h][i][j] += nex_tmp_2[h][i][k] * v_tmp[h][k][j];
				}
			}
		}
	}
	for(int h = 0; h < HEAD_SIZE; ++h)
	for (int i = 0; i < SEQ; ++i) {
		for (int j = 0; j < DIM; ++j) {
			output[h][i][j] = output_pl[h][i][j];
		}
	}
}
template<typename T, int DIM, int SEQ, int HEAD_SIZE>
void multiHeadAttentionForward(T (&Q)[SEQ][DIM], T (&K)[SEQ][DIM],
		T (&V)[SEQ][DIM], T (&output)[SEQ][DIM], T dr) {
	T scale = 1.0 / sqrt((double) DIM * 1.0 / HEAD_SIZE);
	T tmp[HEAD_SIZE][SEQ][DIM];
#pragma HLS ARRAY_PARTITION variable=tmp dim=1 complete
	T output_pl[SEQ][DIM];
#pragma HLS ARRAY_PARTITION variable=output_pl dim=1 complete

	scaleDotSelfAttentionForward<T, DIM, SEQ, HEAD_SIZE>(Q, K, V, tmp, scale,
			dr);

	T fc_tmp[SEQ][DIM * HEAD_SIZE];
#pragma HLS ARRAY_PARTITION variable=fc_tmp dim=1 complete
	for (int h = 0; h < HEAD_SIZE; ++h) {
		for (int i = 0; i < SEQ; ++i) {
			for (int j = 0; j < DIM; ++j) {
				fc_tmp[i][h * HEAD_SIZE + j] = tmp[h][i][j];
			}
		}
	}
	linearForward<T, DIM * HEAD_SIZE, DIM, SEQ>(fc_tmp, output);
}
#endif
