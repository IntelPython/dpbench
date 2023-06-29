# Copyright 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache 2.0

#!/usr/bin/env python

import os

from setuptools import find_packages

if not os.getenv("DPBENCH_SYCL"):
    from setuptools import setup

    cmake_args = None
else:
    import dpctl
    import pybind11
    from skbuild import setup

    pybind11_cmake_dir = pybind11.get_cmake_dir()
    dpctl_include_dir = dpctl.get_include()
    dpctl_cmake_dir = (
        dpctl_include_dir.removesuffix("/include") + "/resource/cmake"
    )
    cmake_args = [
        "-Dpybind11_DIR=" + pybind11_cmake_dir,
        "-DDpctl_INCLUDE_DIRS=" + dpctl_include_dir,
        "-DDPCTL_MODULE_PATH=" + dpctl_cmake_dir,
    ]


setup(
    # https://github.com/pypa/packaging-problems/issues/606
    url="https://github.com/IntelPython/dpbench",
    packages=(
        find_packages(include=["*"])
        + find_packages(where="./dpbench/benchmarks/*/*")
    ),
    include_package_data=True,
    package_data={
        "dpbench.migrations": ["alembic.ini"],
        "dpbench.configs": ["*/*.toml", "*.toml"],
    },
    cmake_args=cmake_args,
)
