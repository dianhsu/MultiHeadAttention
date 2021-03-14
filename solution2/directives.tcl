############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
############################################################
set_directive_top -name project_top "project_top"
set_directive_array_partition -type complete -dim 1 "project_top" param.weights
set_directive_array_partition -type complete -dim 1 "project_top" param.bias
set_directive_array_partition -type complete -dim 1 "project_top" input_pl
set_directive_array_partition -type complete -dim 1 "project_top" output_pl
set_directive_unroll "Linear::forward/forward_label0"
set_directive_unroll "Linear::forward/forward_label1"
set_directive_dataflow "Linear::forward"
