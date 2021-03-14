#include <cstring>
#include "main.h"
#include "attention.h"
#include "linear.h"
void project_top(data_t input[SEQ][DIM], data_t output[SEQ][DIM]) {
	//MultiHeadAttentionParameter<data_t, DIM, HEAD_SIZE> param;
	//MultiHeadAttention<data_t, DIM, SEQ, HEAD_SIZE>::forward(input, output, &param);
	data_t input_pl[SEQ][DIM];
	data_t output_pl[SEQ][DIM];
	memcpy(input_pl, input, sizeof(data_t)*SEQ*DIM);
	LinearParameter<data_t, DIM, DIM> param;
	Linear<data_t, DIM, DIM, SEQ>::forward(input_pl, output_pl, &param);
	memcpy(output, output_pl, sizeof(data_t)*SEQ*DIM);
}
