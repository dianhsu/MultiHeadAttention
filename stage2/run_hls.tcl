############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
############################################################
open_project hls
set_top project_top
add_files attention.h
add_files dropout.h
add_files linear.h
add_files main.cpp
add_files main.h
add_files softmax.h
add_files -tb main_tb.cpp
open_solution "solution" -flow_target vivado
set_part {xcvu19p-fsva3824-2-e}
create_clock -period 10 -name default
source "./directives.tcl"
csim_design
#csynth_design
#cosim_design
#export_design -format ip_catalog
exit