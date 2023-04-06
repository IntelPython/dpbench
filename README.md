[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://github.com/IntelPython/dpbench/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/IntelPython/dpbench/actions/workflows/pre-commit.yml)

# DPBench - Numba/Numba-Dpex/dpnp/DPCPP Benchmarks

* __*_numba_*.py__ : This file contains Numba implementations of the benchmarks. There are three modes: nopython-mode, nopython-mode-parallel and nopython-mode-parallel-range.
* __*_numba_dpex_*.py__ : This file contains Numba-Dpex implementations of the benchmarks. There are three modes: kernel-mode, numpy-mode and prange-mode.
* __*_dpnp_*.py__ : This file contains dpnp implementations of the benchmarks.
* __*_native_ext/_sycl/__kernel_*.hpp__ : This file contains native dpcpp implementations of the benchmarks.

## Examples of setting up and running the benchmarks
1. Setting up conda environment and installing dependencies:

        $ conda create -n dpbench-dev
        $ conda activate dpbench-dev
        $ conda install python
        $ conda install -c intel tbb dpcpp_linux-64
        $ conda install numpy numba cython cmake ninja scikit-build pandas
        $ conda install scipy scikit-learn pybind11
        # do not miss the quotes!
        $ conda install -c pkgs/main libgcc-ng">=11.2.0" libstdcxx-ng">=11.2.0" libgomp">=11.2.0"
        $ conda install -c dppy/label/dev -c intel -c main dpctl numba-dpex dpnp

2. Build and run DPBench
    - To build:
        ```bash
        $  CC=icx CXX=icpx python setup.py develop -- -Dpybind11_DIR=$(python -m pybind11 --cmakedir) -DDPCTL_MODULE_PATH=$(python -m dpctl --cmakedir)
        ```
    - To run, taking black_scholes for example:
        ```bash
        $  python -c "import dpbench; dpbench.run_benchmark(\"black_scholes\")" 2> /dev/null
        ```
    - Similarly, to run all the cases in DPBench:
        ```bash
        $  python -c "import dpbench; dpbench.run_benchmarks()" 2> /dev/null
        ```

3. Device Customization

   If a framework is SYCL based, an extra configuration option `sycl_device` may be set in the
   framework JSON file to control what device the framework uses for execution. The `sycl_device`
   value should be a legal
   [SYCL device filter ](https://intel.github.io/llvm-docs/EnvironmentVariables.html#sycl_device_filter)
   string. The dpcpp, dpnp, and numba_dpex frameworks support the sycl_device option.

   Here is an example:

    ```json
        {
            "framework": {
                "simple_name": "dpcpp",
                "full_name": "dpcpp",
                "prefix": "dp",
                "postfix": "dpcpp",
                "class": "DpcppFramework",
                "arch": "gpu",
                "sycl_device": "level_zero:gpu:0"
            }
        }
    ```

    > **_NOTE:_**  The `arch` option is deprecated and not used by dpbench.

   To run with custimized framework JSON file, pass it as an argument to the `run_benchmark` or
   `run_benchmarks` functions.

    ```bash
    $ python -c "import dpbench; dpbench.run_benchmark(\"black_scholes\", "<absolute path to json file>")"
    ```
