#include <cstring>
#include "main.h"
//#include "attention.h"
#include "attention2.h"
#include "linear.h"
void project_top(data_t input[SEQ][DIM], data_t output[SEQ][DIM]) {
#pragma HLS TOP name=project_top
    data_t input_pl[SEQ][DIM];
#pragma HLS ARRAY_PARTITION variable=input_pl dim=1 complete
    data_t output_pl[SEQ][DIM];
#pragma HLS ARRAY_PARTITION variable=output_pl dim=1 complete
    memcpy(input_pl, input, sizeof(data_t) * SEQ * DIM);
    data_t scale = 0.1;
    data_t dr = 0.1;
    scaledDotProductAttention<data_t, DIM, SEQ>(input_pl, input_pl, input_pl, output_pl);
    //linearForward<data_t, DIM, DIM, SEQ>(input_pl, output_pl);
    //multiHeadAttentionForward<data_t, DIM, SEQ, HEAD_SIZE>(input_pl, input_pl, input_pl, output_pl, dr);
    memcpy(output, output_pl, sizeof(data_t) * SEQ * DIM);
}
