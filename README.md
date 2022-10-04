[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://github.com/IntelPython/dpbench/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/IntelPython/dpbench/actions/workflows/pre-commit.yml)

# DPBench - Numba/Numba-Dpex/DPCPP Benchmarks

* ___*_numba_*.py__ : This file contains Numba implementations of the benchmarks. There are three modes: nopython-mode, nopython-mode-parallel and nopython-mode-parallel-range.
* __*_numba_dpex_*.py__ : This file contains Numba-Dpex implementations of the benchmarks. There are three modes: kernel-mode, numpy-mode and prange-mode.
* __*_native_ext/__sycl/__kernel_*.hpp__ : This file contains native dpcpp implementations of the benchmarks.

## Examples of setting up and running the benchmarks
1. Setting up conda environment and installing dependencies:

        $ conda create -n dpbench-dev
        $ conda activate dpbench-dev
        $ conda install python=3.9 
        $ conda install -c intel tbb=2021.6.0 dpcpp_linux-64
        $ conda install numpy numba cython cmake ninja scikit-build pandas
        $ conda install scipy spirv-tools scikit-learn pybind11
        # do not miss the quotes!
        $ conda install -c pkgs/main libgcc-ng">=11.2.0" libstdcxx-ng">=11.2.0" libgomp">=11.2.0"
        $ conda install -c dppy/label/dev -c intel dpctl=0.13.0 numba-dpex=0.18.1 dpnp=0.10.1

2. Build Numba-Dpex (If you want to use your own numba-dpex)

        $ git clone https://github.com/IntelPython/numba-dpex.git
        $ cd numba-dpex/
        $ PATH=$(dirname $(which icx))/../bin-llvm:$PATH python setup.py develop
        $ cd ..

3. Build and run DPBench
    - To build:
        ```bash
        $  CC=icx CXX=icpx python setup.py develop -- -Dpybind11_DIR=$(python -m pybind11 --cmakedir) -DDPCTL_MODULE_PATH=$(python -m dpctl --cmakedir)
        ```
    - To run, taking black_scholes for example:
        ```bash
        $  python -c "import dpbench; dpbench.run_benchmark(\"black_scholes\")"
        ```

4. Device Customization

   Device can be selected via providing customized arch in framework Json file. Here is an example:
    ```json
        {
            "framework": {
                "simple_name": "dpcpp",
                "full_name": "dpcpp",
                "prefix": "dp",
                "postfix": "dpcpp",
                "class": "DpcppFramework",
                "arch": "gpu"
            }
        }
    ```

   To run with custimized Json file:
    ```bash
    $ python -c "import dpbench; dpbench.run_benchmark(\"black_scholes\", "<absolute path to json file>")"
    ```
