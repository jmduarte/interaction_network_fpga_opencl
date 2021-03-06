#!/bin/bash

unset CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA
unset CL_CONTEXT_EMULATOR_DEVICE_ALTERA

cd bin

aocl program acl0 gnn.aocx
aocl program acl1 gnn.aocx

echo "Pt: 1.5 GeV"
./gnn_fpga /tigress/aheintz/data/model_weights_LP_1p5.hdf5 /tigress/aheintz/data/test_LP_1p5 100 0
