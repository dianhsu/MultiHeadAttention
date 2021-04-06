#ifndef __HLS_SOFTMAX_H__
#define __HLS_SOFTMAX_H__

template<typename T, int DIM, int SEQ>
void softmaxForward(T (&input)[SEQ][DIM], T (&output)[SEQ][DIM]) {
	T input_pl[SEQ][DIM];
	T output_pl[SEQ][DIM];
	SF_LOOP0: for (int i = 0; i < SEQ; ++i) {
		SF_LOOP1: for (int j = 0; j < DIM; ++j) {
			input_pl[i][j] = input[i][j];
		}
	}
	T tmp[DIM];
	SF_LOOP2: for (int j = 0; j < DIM; ++j) {
		tmp[j] = 0;
	}
	SF_LOOP3: for (int i = 0; i < SEQ; ++i) {
		SF_LOOP4: for (int j = 0; j < DIM; ++j) {
			tmp[j] += input_pl[i][j];
		}
	}
	SF_LOOP9:for(int j = 0; j < DIM; ++j){
		tmp[j] = 1/tmp[j];
	}
	SF_LOOP5: for (int i = 0; i < SEQ; ++i) {
		SF_LOOP6: for (int j = 0; j < DIM; ++j) {
			output_pl[i][j] = input_pl[i][j] * tmp[j];
		}
	}
	SF_LOOP7: for (int i = 0; i < SEQ; ++i) {
		SF_LOOP8: for (int j = 0; j < DIM; ++j) {
			output[i][j] = output_pl[i][j];
		}
	}
}

#endif
