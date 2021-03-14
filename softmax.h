#ifndef __HLS_SOFTMAX_H__
#define __HLS_SOFTMAX_H__

template<typename T, int DIM, int SEQ>
class Softmax{
public:
	static void forward(T input[SEQ][DIM], T output[SEQ][DIM]){
		T tmp[DIM];
		for(int j = 0; j < DIM; ++j){
			tmp[j] = 0;
			for(int i = 0; i < SEQ; ++i){
				tmp[j] += input[i][j];
			}
		}
		for(int j = 0; j < DIM; ++j){
			for(int i = 0; i < SEQ; ++i){
				output[i][j] = input[i][j] / tmp[j];
			}
		}
	}
};

#endif
