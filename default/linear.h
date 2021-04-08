#ifndef __HLS_LINEAR_H__
#define __HLS_LINEAR_H__
template<typename T, int DIM_IN, int DIM_OUT>
struct LinearParameter {
	T weights[DIM_IN][DIM_OUT];
	T bias[DIM_OUT];
};

template<typename T, int DIM_IN, int DIM_OUT>
void singleLinearForward(T (&input)[DIM_IN], T (&output)[DIM_OUT]) {
	LinearParameter<T, DIM_IN, DIM_OUT> param;
	T input_pl[DIM_IN];
	T output_pl[DIM_OUT];
	SLF_LOOP0: for (int i = 0; i < DIM_IN; ++i) {
		input_pl[i] = input[i];
	}
	SLF_LOOP1: for (int j = 0; j < DIM_OUT; ++j) {
		output_pl[j] = param.bias[j];
	}
	SLF_LOOP2: for (int i = 0; i < DIM_IN; ++i) {
#pragma HLS PIPELINE off
		SLF_LOOP3: for (int j = 0; j < DIM_OUT; ++j) {
#pragma HLS PIPELINE off
			output_pl[j] += input_pl[i] * param.weights[i][j];
		}
	}
	SLF_LOOP4: for (int j = 0; j < DIM_OUT; ++j) {
		output[j] = output_pl[j];
	}
}

template<typename T, int DIM_IN, int DIM_OUT, int SEQ>
void linearForward(T (&input)[SEQ][DIM_IN], T (&output)[SEQ][DIM_OUT]) {
	T input_pl[SEQ][DIM_IN];
	T output_pl[SEQ][DIM_OUT];
	LF_LOOP0:for (int i = 0; i < SEQ * DIM_IN; ++i) {
		input_pl[i / DIM_IN][i % DIM_IN] = input[i / DIM_IN][i % DIM_IN];
	}
	LF_LOOP1:for (int q = 0; q < SEQ; ++q) {
		singleLinearForward(input_pl[q], output_pl[q]);
	}
	LF_LOOP2:for (int i = 0; i < SEQ * DIM_OUT; ++i) {
		output[i / DIM_OUT][i % DIM_OUT] = output_pl[i / DIM_OUT][i % DIM_OUT];
	}
}

#endif
