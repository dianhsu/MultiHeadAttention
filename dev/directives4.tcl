############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
############################################################
set_directive_top -name project_top "project_top"
set_directive_array_partition -type complete -dim 1 "project_top" input_pl
set_directive_array_partition -type complete -dim 1 "project_top" output_pl
set_directive_pipeline "softmaxForward"
set_directive_unroll "dropoutForward/DF_LOOP0"
set_directive_inline "dropoutForward"
set_directive_array_partition -type complete -dim 1 "multiHeadAttentionForward" fc_tmp
set_directive_array_partition -type complete -dim 1 "multiHeadAttentionForward" tmp
set_directive_array_partition -type complete -dim 1 "scaleDotSelfAttentionForward" nex_tmp
set_directive_unroll "linearForward/LF_LOOP2"
set_directive_array_partition -type complete -dim 1 "singleLinearForward" param.weights
set_directive_array_partition -type complete -dim 1 "singleLinearForward" param.bias
set_directive_array_partition -type complete -dim 1 "singleLinearForward" input_pl
set_directive_array_partition -type complete -dim 1 "singleLinearForward" output_pl
set_directive_unroll "singleLinearForward/SLF_LOOP1"
set_directive_unroll "singleLinearForward/SLF_LOOP3"
set_directive_dataflow "project_top"
set_directive_dataflow "scaleDotSelfAttentionForward/SDSAF_BLOCK0"
set_directive_dataflow "scaleDotSelfAttentionForward"
set_directive_array_partition -type complete -dim 1 "softmaxForward" output_pl
set_directive_array_partition -type complete -dim 1 "softmaxForward" input_pl
set_directive_unroll "softmaxForward/SF_LOOP2"
set_directive_unroll "softmaxForward/SF_LOOP1"
set_directive_unroll "softmaxForward/SF_LOOP4"
set_directive_unroll "softmaxForward/SF_LOOP6"
set_directive_array_partition -type complete -dim 1 "scaleDotSelfAttentionForward" v_tmp
set_directive_array_partition -type complete -dim 2 "scaleDotSelfAttentionForward" nex_tmp_2
set_directive_unroll "scaleDotSelfAttentionForward/SDSAF_LOOP7"
set_directive_dataflow "multiHeadAttentionForward"
set_directive_array_partition -type complete -dim 1 "linearForward" input_pl
set_directive_array_partition -type complete -dim 1 "linearForward" output_pl
set_directive_dataflow "multiHeadAttentionForward/MHAF_LOOP0"
set_directive_unroll "softmaxForward/SF_LOOP9"
set_directive_unroll "linearForward/LF_LOOP1"
set_directive_unroll "scaleDotSelfAttentionForward/SDSAF_LOOP4"
