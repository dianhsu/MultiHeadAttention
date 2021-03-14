#ifndef __HLS_ATTENTION_H__
#define __HLS_ATTENTION_H__

#include <cmath>
#include "linear.h"
#include "dropout.h"
#include "softmax.h"

template<typename T, int DIM, int HEAD_SIZE>
class MultiHeadAttentionParameter {
public:
	LinearParameter<T, DIM, DIM> lp_Q[HEAD_SIZE], lp_K[HEAD_SIZE],
			lp_V[HEAD_SIZE];
	LinearParameter<T, DIM * HEAD_SIZE, DIM> lp;
	T dr = 0.1;
};

template<typename T, int DIM, int SEQ, int HEAD_SIZE>
class MultiHeadAttention {
public:
	static void forward(T Q[SEQ][DIM], T K[SEQ][DIM], T V[SEQ][DIM],
			T output[SEQ][DIM],
			MultiHeadAttentionParameter<T, DIM, HEAD_SIZE> *param) {
		T scale = 1.0 / sqrt((T) DIM * 1.0 / HEAD_SIZE);
		T q_tmp[2][HEAD_SIZE][SEQ][DIM];
		T k_tmp[HEAD_SIZE][SEQ][DIM];
		T v_tmp[HEAD_SIZE][SEQ][DIM];
		for (int i = 0; i < HEAD_SIZE; ++i) {
			Linear<T, DIM, DIM, SEQ>::forward(Q, q_tmp[0][i], param->lp_Q[i]);
			Linear<T, DIM, DIM, SEQ>::forward(K, k_tmp[i], param->lp_K[i]);
			Linear<T, DIM, DIM, SEQ>::forward(V, v_tmp[i], param->lp_V[i]);
			for (int j = 0; j < SEQ; ++j) {
				Dropout<T, DIM>::forward(q_tmp[0][i][j], q_tmp[1][i][j],
						param->dr);
				for (int k = 0; k < DIM; ++k) {
					q_tmp[1][i][j][k] *= scale;
				}
			}
		}
		T nex_tmp[2][HEAD_SIZE][SEQ][SEQ];
		for (int h = 0; h < HEAD_SIZE; ++h) {
			for (int i = 0; i < SEQ; ++i) {
				for (int j = 0; j < SEQ; ++j) {
					nex_tmp[0][h][i][j] = 0;
					for(int k = 0; k < DIM; ++k){
						nex_tmp[0][h][i][j] += q_tmp[1][h][i][k] * k_tmp[h][j][k];
					}
				}
			}
			Softmax<T, SEQ, SEQ>::forward(nex_tmp[0][h], nex_tmp[1][h]);
		}
		T fc_tmp[SEQ][DIM*HEAD_SIZE];
		for(int h = 0; h < HEAD_SIZE; ++h){
			for(int i = 0; i < SEQ; ++i){
				for(int j = 0; j < DIM; ++j){
					fc_tmp[i][h*HEAD_SIZE +j] = 0;
					for(int k = 0; k < SEQ; ++k){
						fc_tmp[i][h*HEAD_SIZE+j] += nex_tmp[1][h][i][k] * v_tmp[h][k][j];
					}
				}
			}
		}
		Linear<T, DIM*HEAD_SIZE, DIM, SEQ>::forward(fc_tmp, output, p->lp);
	}
};
#endif
