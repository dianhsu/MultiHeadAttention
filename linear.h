#ifndef __HLS_LINEAR_H__
#define __HLS_LINEAR_H__
template<typename T, int DIM_IN, int DIM_OUT>
class LinearParameter {
public:
	T weights[DIM_IN][DIM_OUT];
	T bias[DIM_OUT];
};

template<typename T, int DIM_IN, int DIM_OUT, int SEQ>
class Linear {
public:
	static void forward(T input[SEQ][DIM_IN], T output[SEQ][DIM_OUT],
			LinearParameter<T, DIM_IN, DIM_OUT> *param) {
		for (int i = 0; i < SEQ; ++i) {
			forward_label0: for (int j = 0; j < DIM_OUT; ++j) {
				output[i][j] = param->bias[j];
			}
		}
		for (int i = 0; i < SEQ; ++i) {
			for (int j = 0; j < DIM_IN; ++j) {
				forward_label1: for (int k = 0; k < DIM_OUT; ++k) {
					output[i][k] += input[i][j] * param->weights[j][k];
				}
			}
		}

	}
};

#endif
