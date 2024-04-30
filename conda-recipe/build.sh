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

# TODO: switch to pip build. Currently results in broken binary on Windows
# $PYTHON -m pip install --no-index --no-deps --no-build-isolation . -v

# -wnx flags mean: --wheel --no-isolation --skip-dependency-check
${PYTHON} -m build -w -n -x
${PYTHON} -m wheel tags --remove --build "$GIT_DESCRIBE_NUMBER" \
    --platform-tag manylinux2014_x86_64 dist/dpbench*.whl
${PYTHON} -m pip install dist/dpbench*.whl \
    --no-build-isolation \
    --no-deps \
    --only-binary :all: \
    --no-index \
    --prefix ${PREFIX} \
    -vv

# Must be consistent with pyproject.toml project.scritps. Currently pip does
# not allow to ignore scripts installation, so we have to remove them manually.
# https://github.com/pypa/pip/issues/3980
# We have to let conda-build manage it for use in order to set proper python
# path.
# https://docs.conda.io/projects/conda-build/en/stable/resources/define-metadata.html#python-entry-points
rm ${PREFIX}/bin/dpbench

# Copy wheel package
if [[ -v WHEELS_OUTPUT_FOLDER ]]; then
    cp dist/dpbench*.whl "${WHEELS_OUTPUT_FOLDER[@]}"
fi
