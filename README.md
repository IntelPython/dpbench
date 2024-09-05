<!--
SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation

SPDX-License-Identifier: Apache-2.0
-->

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://github.com/IntelPython/dpbench/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/IntelPython/dpbench/actions/workflows/pre-commit.yml)

# DPBench - Benchmarks to evaluate Data-Parallel Extensions for Python

* **\<benchmark\>\_numba\_\<mode\>.py** : This file contains Numba implementations of the benchmarks. There are three modes: nopython-mode, nopython-mode-parallel and nopython-mode-parallel-range.
* **\<benchmark\>\_numba_dpex\_\<mode\>.py** : This file contains Numba-Dpex implementations of the benchmarks. There are three modes: kernel-mode, numpy-mode and prange-mode.
* **\<benchmark\>\_dpnp\_\<mode\>.py** : This file contains dpnp implementations of the benchmarks.
* **\<benchmark\>\_native_ext/\<benchmark\>\_sycl/_\<benchmark\>_kernel.hpp** : This file contains native dpcpp implementations of the benchmarks.
* **\<benchmark\>\_numpy.py** : This file contains numpy implementations of the benchmarks. It should take benefits of numpy arrays and should avoid loops over arrays.
* **\<benchmark\>\_python.py** : This file contains naive python implementations of the benchmarks. Should be run only for small presets, otherwise it will take long execution time.
* **\<benchmark\>\_numba_mlir\_\<mode\>.py** : This file contains Numba-MLIR implementations of the benchmarks. There are three modes: kernel-mode, numpy-mode and prange-mode. Experimental.

## Examples of setting up and running the benchmarks

### Using prebuilt version

1. Create conda environment

    ```bash
    conda create -n dpbench dpbench -c dppy/label/dev -c conda-forge -c https://software.repos.intel.com/python/conda -c nodefaults --override-channels
    conda activate dpbench
    ```

2. Run specific benchmark, e.g. black_scholes

    ```bash
    dpbench -b black_scholes run
    ```

### Build from source (for development)

1. Clone the repository

    ```bash
    git clone https://github.com/IntelPython/dpbench
    cd dpbench
    ```

2. Setting up conda environment and installing dependencies:

    ```bash
    conda env create -n dpbench -f ./environments/conda.yml
    ```

    If you want to build sycl benchmarks as well:
    ```bash
    conda env create -n dpbench -f ./environments/conda-linux-sycl.yml
    ```

3. Build DPBench

    ```bash
    pip install --no-index --no-deps --no-build-isolation -e . -v
    ```

    Alternatively you can build it with `setup.py`, but pip version is preferable:

    ```bash
    python setup.py develop
    ```

    For sycl build use:
    ```bash
    CC=icx CXX=icpx DPBENCH_SYCL=1 pip install --no-index --no-deps --no-build-isolation -e . -v
    ```

    or

    ```bash
    CC=icx CXX=icpx DPBENCH_SYCL=1 python setup.py develop
    ```

4. Run specific benchmark, e.g. black_scholes

    ```bash
    dpbench -b black_scholes run
    ```

### Usage

1. Run all benchmarks

    ```bash
    dpbench -a run
    ```

2. Generate report

    ```bash
    dpbench report
    ```

3. Device Customization

   If a framework is SYCL based, an extra configuration option
   `sycl_device` may be set in the framework config file or by passing
   `--sycl-device` argument to `dpbench run` to control what device
   the framework uses for execution. The `sycl_device` value should be
   a legal [SYCL device filter
   ](https://intel.github.io/llvm-docs/EnvironmentVariables.html#sycl_device_filter)
   string. The dpcpp, dpnp, and numba_dpex frameworks support the
   sycl_device option.

   Here is an example:

    ```shell
    dpbench -b black_scholes -i dpnp run --sycl-device=level_zero:gpu:0
    ```

4. All available options are available using `dpbench --help` and `dpbench <command> --help`:

    ```
    usage: dpbench [-h] [-b [BENCHMARKS]] [-i [IMPLEMENTATIONS]] [-a | --all-implementations | --no-all-implementations] [--version] [-r [RUN_ID]] [--last-run | --no-last-run] [-d [RESULTS_DB]]
               [--log-level [{critical,fatal,error,warning,info,debug}]]
               {run,report,config} ...

    positional arguments:
    {run,report,config}

    options:
    -h, --help            show this help message and exit
    -b [BENCHMARKS], --benchmarks [BENCHMARKS]
                            Comma separated list of benchmarks. Leave empty to load all benchmarks.
    -i [IMPLEMENTATIONS], --implementations [IMPLEMENTATIONS]
                            Comma separated list of implementations. Use --all-implementations to load all available implementations.
    -a, --all-implementations, --no-all-implementations
                            If set, all available implementations will be loaded.
    --version             show program's version number and exit
    -r [RUN_ID], --run-id [RUN_ID]
                            run_id to perform actions on. Use --last-run to use latest available run, or leave empty to create new one.
    --last-run, --no-last-run
                            Sets run_id to the latest run_id from the database.
    -d [RESULTS_DB], --results-db [RESULTS_DB]
                            Path to a database to store results.
    --log-level [{critical,fatal,error,warning,info,debug}]
                            Log level.
    ```

    ```
    usage: dpbench run [-h] [-p [{S,M16Gb,M,L}]] [-s | --validate | --no-validate] [--dpbench | --no-dpbench] [--experimental-npbench | --no-experimental-npbench] [--experimental-polybench | --no-experimental-polybench]
                   [--experimental-rodinia | --no-experimental-rodinia] [-r [REPEAT]] [-t [TIMEOUT]] [--precision [{single,double}]] [--print-results | --no-print-results] [--save | --no-save] [--sycl-device [SYCL_DEVICE]]
                   [--skip-expected-failures | --no-skip-expected-failures]

    Subcommand to run benchmark executions.

    options:
    -h, --help            show this help message and exit
    -p [{S,M16Gb,M,L}], --preset [{S,M16Gb,M,L}]
                            Preset to use for benchmark execution.
    -s, --validate, --no-validate
                            Set if the validation will be run for each benchmark.
    --dpbench, --no-dpbench
                            Set if run dpbench benchmarks.
    --experimental-npbench, --no-experimental-npbench
                            Set if run npbench benchmarks.
    --experimental-polybench, --no-experimental-polybench
                            Set if run polybench benchmarks.
    --experimental-rodinia, --no-experimental-rodinia
                            Set if run rodinia benchmarks.
    -r [REPEAT], --repeat [REPEAT]
                            Number of repeats for each benchmark.
    -t [TIMEOUT], --timeout [TIMEOUT]
                            Timeout time in seconds for each benchmark execution.
    --precision [{single,double}]
                            Data precision to use for array initialization.
    --print-results, --no-print-results
                            Show the result summary or not
    --save, --no-save     Either to save execution into database.
    --sycl-device [SYCL_DEVICE]
                            Sycl device to overwrite for framework configurations.
    --skip-expected-failures, --no-skip-expected-failures
                            Either to save execution into database.
    ```

    ```
    usage: dpbench report [--comparisons [COMPARISON_PAIRS]] [--csv]

    Subcommand to generate a summary report from the local DB

    options:
    -c, --comparisons [COMPARISON_PAIRS]
                            Comma separated list of implementation pairs to be compared
    --csv
                            Sets the general summary report to output in CSV format (default: False)
    ```

### Performance Measurement

For each benchmark, we measure the execution time of the
computationally intesive part, but not the intialization or
shutdown. We provide three inputs (a.k.a presets) for each benchmark.

* **S** - Minimal input to verify that programs are executable
* **M** - Medium-sized input for performance measurements on client devices
* **L** - Large-sized input for performance measurements on servers

As a rough guideline for selecting input sizes, **S** inputs need to
be small enough for python and numpy implementations to execute in
<100ms. **M** and **L** inputs need to be large enough to obtain
useful performance insights on client and servers devices,
respectively. Also, note that the python and numpy implementations are
not expected to work with **M** and **L** inputs.
