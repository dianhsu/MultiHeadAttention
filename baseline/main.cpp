#include <cstring>
#include "main.h"
#include "attention.h"
#include "linear.h"
void project_top(data_t (&input)[SEQ][DIM], data_t (&output)[SEQ][DIM]) {
#pragma HLS TOP name=project_top
	//MultiHeadAttentionParameter<data_t, DIM, HEAD_SIZE> param;
	//multiHeadAttentionForward<data_t, DIM, SEQ, HEAD_SIZE>(input_pl, input_pl, input_pl, output_pl, param);
	//LinearParameter<data_t, DIM, DIM> param;
	//linearForward<data_t, DIM, DIM, SEQ>(input_pl, output_pl);
	data_t scale = 0.1;
	data_t dr = 0.1;
	multiHeadAttentionForward<data_t, DIM, SEQ, HEAD_SIZE>(input, input, input, output, dr);
}
