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
#pragma HLS ARRAY_PARTITION variable=param.bias dim=1 complete
#pragma HLS ARRAY_PARTITION variable=param.weights dim=1 complete
	T input_pl[DIM_IN];
#pragma HLS ARRAY_PARTITION variable=input_pl dim=1 complete
	T output_pl[DIM_OUT];
#pragma HLS ARRAY_PARTITION variable=output_pl dim=1 complete
	SLF_LOOP0: for (int i = 0; i < DIM_IN; ++i) {
		input_pl[i] = input[i];
	}
	SLF_LOOP1: for (int j = 0; j < DIM_OUT; ++j) {
#pragma HLS UNROLL
		output_pl[j] = param.bias[j];
	}
	SLF_LOOP2: for (int i = 0; i < DIM_IN; ++i) {
		SLF_LOOP3: for (int j = 0; j < DIM_OUT; ++j) {
#pragma HLS UNROLL
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
#pragma HLS ARRAY_PARTITION variable=input_pl dim=1 complete
	T output_pl[SEQ][DIM_OUT];
#pragma HLS ARRAY_PARTITION variable=output_pl dim=1 complete
	LF_LOOP0:for (int i = 0; i < SEQ * DIM_IN; ++i) {
		input_pl[i / DIM_IN][i % DIM_IN] = input[i / DIM_IN][i % DIM_IN];
	}
	LF_LOOP1:for (int q = 0; q < SEQ; ++q) {
#pragma HLS UNROLL
		singleLinearForward(input_pl[q], output_pl[q]);
	}
	LF_LOOP2:for (int i = 0; i < SEQ * DIM_OUT; ++i) {
#pragma HLS UNROLL
		output[i / DIM_OUT][i % DIM_OUT] = output_pl[i / DIM_OUT][i % DIM_OUT];
	}
}

#endif
