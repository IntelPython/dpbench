[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://github.com/IntelPython/dpbench/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/IntelPython/dpbench/actions/workflows/pre-commit.yml)

# DPBench - Numba/Native Benchmarks

This repository contains a set of benchmarks that are used for evaluating the performance Numba's JIT compilation functionality for Intel GPUs. The repository is structured as follows.
* __numba__ : Contains Numba implementations of the benchmarks. Each benchmark directory contains two sub-directories - CPU and GPU. These sub-directories contain the CPU and GPU Numba implementations of the benchmark, respectively.
* __native__ : Contains C++/OpenMP implementations of the benchmarks. The CPU implementations (in CPU sub-directory) use OpenMP parallel semantics. The GPU implementations use OpenMP offload
* __native_dpcpp__ : Contains DPC++ implementations of the benchmarks.
* __dpnp__ : Contains dpnp implementations for a subset of the benchmarks

In addition to the implementations, this repository contains a set of Python scripts to exercise the implementations. These Python scripts provide mechanisms to the user to build and run benchmark programs. The Python scripts can plot bar graphs showing the performance throughput of the benchmarks for the executed implementations.

The primary interface to running the benchmarks is `automate_run.py` script. It accepts the following options:
* _-r, --run_ : "execute" the benchmark/s or "plot" performance data to generate graphs (Default: "all" if option unspecified. Runs both)
* _-ws_ : name/s of benchmark/s to execute or "all" to execute every benchmark (Default: "all" if option unspecified)
* _-i, --impl_ : execute "native" or "numba" or "dpnp" or "native_dpcpp" implementation (Default: "all" if option unspecified. Runs both implementations)
* _-k_ : execute dppy.kernel implementation if available. This option can be used only if "-i" is set to "numba"
* _-p, --platform_ : execute "cpu" or "gpu" implementation (Default: "all" if option unspecified. Runs both)
* _-a, --analysis_ : selects the type of execution. Currently we support four analysis options. "test" runs the benchmark with the smallest input and is suitable for testing the functionality of the benchmark. "perf" runs the benchmark on varying inputs and generates performance data. "vtune" and "advisor" run the benchmark with Intel VTune and Intel Advisor profiling tools

`python automate_run.py -h` specifies the list of options and the arguments that can be provided to them.

Note: To obtain GPU roofline graph using Intel Advisor, set the value of the _dev.i915.perf_stream_paranoid_ sysctl option to 0 using `sudo sysctl -w dev.i915.perf_stream_paranoid=0`. This command makes a temporary-only change that is lost on the next reboot. Hence, every time the machine is rebooted the command needs to be executed. More details on obtaining GPU Roofline using Intel Advisor can be found at [this link](https://software.intel.com/content/www/us/en/develop/documentation/get-started-with-advisor/top/offload-advisor-workflow/identify-gpu-performance-bottlenecks-using-gpu-roofline.html).

## Examples of running the benchmarks

1. To generate performance data, plot graph, VTune profile and Advisor roofline graph for CPU and GPU implementations of all benchmarks

        $ python automate_run.py

2. To generate performance data for numba implementations only (CPU and GPU)

        $ python automate_run.py -r execute -i numba -a perf

3. To generate advisor roofline graph for native GPU implementations of all benchmarks

        $ python automate_run.py -a advisor -i native -p gpu

   Roofline graph for each benchmark can be found at `<path/to/native/benchmark/directory>/GPU/roofline/roofline.html`.

   Note: To obtain GPU roofline graph using Intel Advisor, ensure the value of _dev.i915.perf_stream_paranoid_ sysctl option is set to 0. If not set to 0, use `sudo sysctl -w dev.i915.perf_stream_paranoid=0` to set it to 0

4. Generate VTune profile for kmeans and pairwise_distance benchmarks numba CPU implementations

        $ python automate_run.py -a vtune -i numba -p cpu -ws kmeans pairwise_distance

5. Run "test" version of l2_distance Numba GPU benchmark

        $ python automate_run.py -ws l2_distance -a test -p gpu -i numba

6. Plot graph from all performance data (No execution)

        $ python automate_run.py -r plot

6. Plot graph to compare specific benchmark's numba performance (cpu vs gpu)

        $ python automate_run.py -r plot -ws kmeans -i numba

6. Plot graph to compare a set of benchmarks' gpu performance (numba vs native)

        $ python automate_run.py -r plot -ws kmeans blackscholes l2_distance -p gpu
