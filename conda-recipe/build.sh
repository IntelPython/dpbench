#!/bin/bash

# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

# Intel LLVM must cooperate with compiler and sysroot from conda
echo "--gcc-toolchain=${BUILD_PREFIX} --sysroot=${BUILD_PREFIX}/${HOST}/sysroot -target ${HOST}" > icpx_for_conda.cfg
export ICPXCFG="$(pwd)/icpx_for_conda.cfg"
export ICXCFG="$(pwd)/icpx_for_conda.cfg"

export CMAKE_GENERATOR="Ninja"
export DPBENCH_SYCL=1

if [ -e "_skbuild" ]; then
    ${PYTHON} setup.py clean --all
fi

SKBUILD_ARGS="-- -DCMAKE_C_COMPILER:PATH=icx -DCMAKE_CXX_COMPILER:PATH=icpx -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON"

# Build wheel package
if [ "$CONDA_PY" == "36" ]; then
    WHEELS_BUILD_ARGS="-p manylinux1_x86_64"
else
    WHEELS_BUILD_ARGS="-p manylinux2014_x86_64"
fi
if [ -n "${WHEELS_OUTPUT_FOLDER}" ]; then
    $PYTHON setup.py install bdist_wheel ${WHEELS_BUILD_ARGS} ${SKBUILD_ARGS}
    cp dist/dpnp*.whl ${WHEELS_OUTPUT_FOLDER}
else
    $PYTHON setup.py install ${SKBUILD_ARGS}
fi
