############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
############################################################
open_project mha
set_top project_top
add_files mha/attention.h
add_files mha/dropout.h
add_files mha/linear.h
add_files mha/main.cpp
add_files mha/main.h
add_files mha/softmax.h
add_files -tb mha/main_tb.cpp
open_solution "solution1" -flow_target vivado
set_part {xczu9eg-ffvb1156-2-e}
create_clock -period 10 -name default
source "./mha/solution1/directives.tcl"
csim_design
csynth_design
cosim_design
export_design -format ip_catalog
