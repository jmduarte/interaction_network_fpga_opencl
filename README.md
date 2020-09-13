# interaction_network_fpga_opencl

This repository is an exploratory implementation of an Interaction Network for particle track reconstruction on FPGAs using OpenCL.

The network is described in the following repository: https://github.com/savvy379/princeton_gnn_tracking

Several bash scripts are provided to easily set up the necessary environment and compile the code on the FPGAs. The only change that needs to be made is to re-set data and local path locations inside each script.

To set up the environment (must be done before compiling or running the code), run the following:

> source environmentSetUp.sh

Because it takes hours to compile for an FPGA, it is recommended to first compile the code on an emulator. To compile the code on the emulator, run the emulator_setup.sh script:

> bash emulator_setup.sh

To run the code on the emulator, run the emulator_run.sh script:

> bash emulator_run.sh

To compile the code on the fpga, run the fpga_setup.sh script:

> bash fpga_setup.sh

To run the code on the emulator, run the fpga_run.sh script:

> bash fpga_run.sh

