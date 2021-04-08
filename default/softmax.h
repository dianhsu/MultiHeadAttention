#ifndef __HLS_SOFTMAX_H__
#define __HLS_SOFTMAX_H__

template<typename T, int DIM, int SEQ>
void softmaxForward(T (&input)[SEQ][DIM], T (&output)[SEQ][DIM]) {
	T tmp[DIM];
	SF_LOOP0: for (int j = 0; j < DIM; ++j) {
		tmp[j] = 0;
	}
	SF_LOOP1: for (int i = 0; i < SEQ; ++i) {
#pragma HLS PIPELINE off
		SF_LOOP2: for (int j = 0; j < DIM; ++j) {
#pragma HLS PIPELINE off
			tmp[j] += input[i][j];
		}
	}
	SF_LOOP3:for(int j = 0; j < DIM; ++j){
#pragma HLS PIPELINE off
		tmp[j] = 1/tmp[j];
	}
	SF_LOOP4: for (int i = 0; i < SEQ; ++i) {
#pragma HLS PIPELINE off
		SF_LOOP5: for (int j = 0; j < DIM; ++j) {
#pragma HLS PIPELINE off
			output[i][j] = input[i][j] * tmp[j];
		}
	}
}

#endif
