# Copyright 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache 2.0

#!/usr/bin/env python

import os

import versioneer
from setuptools import find_packages

if not os.getenv("DPBENCH_SYCL"):
    from setuptools import setup

    cmake_args = None
else:
    from skbuild import setup
    from skbuild.platform_specifics import windows

    cmake_args = ["-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON"]

    # Monkey patch msvc compiler environment, so scikit-build does not overwrite
    # it. Make sure to set desired environment using:
    # > vcvars64.bat -vcvars_ver=<vcvars_ver>
    windows._get_msvc_compiler_env = lambda v, t: windows.CachedEnv()

setup(
    # https://github.com/pypa/packaging-problems/issues/606
    url="https://github.com/IntelPython/dpbench",
    packages=(
        find_packages(include=["*"])
        + find_packages(where="./dpbench/benchmarks/*/*")
    ),
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    include_package_data=True,
    package_data={
        "dpbench.migrations": ["alembic.ini"],
        "dpbench.configs": ["*/*.toml", "*.toml"],
    },
    cmake_args=cmake_args,
)
