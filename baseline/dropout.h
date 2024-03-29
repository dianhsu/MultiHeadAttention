#ifndef __HLS_DROPOUT_H__
#define __HLS_DROPOUT_H__

template<typename T, int DIM>
void dropoutForward(T (&input)[DIM], T (&output)[DIM], T dropout_rate) {
	DF_LOOP0:for (int i = 0; i < DIM; ++i) {
#pragma HLS PIPELINE off
		if (input[i] < dropout_rate) {
			output[i] = 0;
		} else {
			output[i] = input[i];
		}
	}
}
#endif
