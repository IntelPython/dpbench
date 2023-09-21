#!/bin/bash -x

# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

# Intel LLVM must cooperate with compiler and sysroot from conda
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${BUILD_PREFIX}/lib"

echo "--gcc-toolchain=${BUILD_PREFIX} --sysroot=${BUILD_PREFIX}/${HOST}/sysroot -target ${HOST}" > icpx_for_conda.cfg
export ICPXCFG="$(pwd)/icpx_for_conda.cfg"
export ICXCFG="$(pwd)/icpx_for_conda.cfg"

export DPBENCH_SYCL=1
export CMAKE_GENERATOR="Ninja"
export CC=icx
export CXX=icpx

if [ -e "_skbuild" ]; then
    ${PYTHON} setup.py clean --all
fi

# TODO: switch to pip build. Currently results in broken binary on Windows
# $PYTHON -m pip install --no-index --no-deps --no-build-isolation . -v

# Build wheel package
if [ -n "${WHEELS_OUTPUT_FOLDER}" ]; then
    $PYTHON setup.py install --single-version-externally-managed --record=record.txt bdist_wheel -p manylinux2014_x86_64
    mkdir -p ${WHEELS_OUTPUT_FOLDER}
    cp dist/dpbench*.whl ${WHEELS_OUTPUT_FOLDER}
else
    $PYTHON setup.py install --single-version-externally-managed --record=record.txt
fi
