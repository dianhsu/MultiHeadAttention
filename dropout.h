#ifndef __HLS_DROPOUT_H__
#define __HLS_DROPOUT_H__

template<typename T, int DIM>
class Dropout{
public:
	static void forward(T input[DIM], T output[DIM], T dropout_rate){
		for(int i = 0; i < DIM; ++i){
			if(input[i] < dropout_rate){
				output[i] = 0;
			}else{
				output[i] = input[i];
			}
		}
	}
};

#endif
