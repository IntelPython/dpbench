#!/bin/bash -x

export PYBIND11_INCLUDES=$(python3 -m pybind11 --includes)
export DPCTL_INCLUDE_DIR=$(python -c "import dpctl; print(dpctl.get_include())")
export DPCTL_LIB_DIR=${DPCTL_INCLUDE_DIR}/..
export PY_EXT_SUFFIX=$(python3-config --extension-suffix)

dpcpp -Wall -O3 -DSYCL_USE_NATIVE_FP_ATOMICS -D__DO_FLOAT__ -shared -fPIC ${PYBIND11_INCLUDES} -I${DPCTL_INCLUDE_DIR} gpairs.cpp -o sycl_gpairs${PY_EXT_SUFFIX}
