############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
############################################################
open_project Multi_Head_Attention
set_top project_top
add_files softmax.h
add_files main.h
add_files main.cpp
add_files linear.h
add_files dropout.h
add_files attention.h
add_files -tb main_tb.cpp -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
open_solution "solution2" -flow_target vivado
set_part {xczu9eg-ffvb1156-2-e}
create_clock -period 10 -name default
#source "./Multi_Head_Attention/solution2/directives.tcl"
csim_design
csynth_design
cosim_design
export_design -format ip_catalog
