# RTL Testbench

This section describes the implementation of Snitch's RTL testbench, as used in QuestaSim and VCS simulations.

## Top level

The top level module of the RTL testbench is defined in [tb_bin.sv](https://github.com/pulp-platform/snitch_cluster/blob/main/target/common/test/tb_bin.sv).

This module does a few simple things:
- instantiate the DUT-specific testharness
- generate reset and clock signals
- manage the interaction with the DPI-C simulation environment

As described in more detail in the next section, this last step involves polling the DPI-C simulation environment for the simulated Snitch binary's return code. Here, once the return code is received, an error is raised if it is non-zero and the simulation is terminated.

## DPI-C simulation environment

The `fesvr_tick()` function represents the entry point to the DPI-C simulation environment. It is defined in [rtl_lib.cc](https://github.com/pulp-platform/snitch_cluster/blob/main/target/common/test/rtl_lib.cc). On the first invocation it creates a simulation  object, i.e. an instance `s` of the `Sim` class. This class is defined in [sim.hh](https://github.com/pulp-platform/snitch_cluster/blob/main/target/common/test/sim.hh). Implementations for some of its class methods can be found in [common_lib.cc](https://github.com/pulp-platform/snitch_cluster/blob/main/target/common/test/common_lib.cc) and [rtl_lib.cc](https://github.com/pulp-platform/snitch_cluster/blob/main/target/common/test/rtl_lib.cc). Successive invocations of `fesvr_tick()` merely invoke the simulation object's `run()` method, which polls a specific 32-bit memory location for the simulated Snitch binary's return code. For the simulation to terminate, a `1` must be written to the LSB of this memory location by the simulated Snitch binary, while the upper bits carry the binary's return code.

The `Sim` class additionally defines the `ipc` class variable. This variable is an instance of the `IpcIface` class defined in [ipc.hh](https://github.com/pulp-platform/snitch_cluster/blob/main/target/common/test/ipc.hh) and [ipc.cc](https://github.com/pulp-platform/snitch_cluster/blob/main/target/common/test/ipc.cc). If the simulation is launched with the `--ipc` flag, the `IpcIface` constructor will spawn a new thread to handle inter-process communication.
