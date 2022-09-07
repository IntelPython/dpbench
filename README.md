[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://github.com/IntelPython/dpbench/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/IntelPython/dpbench/actions/workflows/pre-commit.yml)

# DPBench - Numba/Numba-Dpex/DPCPP Benchmarks

* ___*_numba_*.py__ : This file contains Numba implementations of the benchmarks. There are three modes: nopython-mode, nopython-mode-parallel and nopython-mode-parallel-range.
* __*_numba_dpex_*.py__ : This file contains Numba-Dpex implementations of the benchmarks. There are three modes: kernel-mode, numpy-mode and prange-mode.
* __*_native_ext/__sycl/__kernel_*.hpp__ : This file contains native dpcpp implementations of the benchmarks.

## Examples of setting up and running the benchmarks
1. Setting up conda environment and installing dependencies:

        $ conda create -n dpbench
        $ conda activate dpbench
        $ conda install python=3.9 numpy cython cmake ninja scikit-build conda-forge::gtest conda-forge::gmock pytest
        $ conda install dpnp -c dppy/label/dev -c /opt/intel/oneapi/conda_channel --override-channels
        $ conda install numba scipy scikit-learn spirv-tool
        $ conda install pybind11
        $ conda install dpcpp_linux-64 -c intel

2. Build Numba-Dpex

        $ git clone https://github.com/IntelPython/numba-dpex.git
        $ cd numba-dpex/
        $ PATH=$(dirname $(which icx))/../bin-llvm:$PATH python setup.py develop
        $ cd ..

3. Build and run DPBench

        #To build:
        $  CC=icx CXX=icpx python setup.py develop -- -Dpybind11_DIR=$(python -m pybind11 --cmakedir) -DDPCTL_MODULE_PATH=$(python -m dpctl --cmakedir)
        #To run, taking black_scholes for example:
        $  python -c "import dpbench; dpbench.run_benchmark(\"black_scholes\")"
