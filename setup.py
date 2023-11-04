# Copyright 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache 2.0

#!/usr/bin/env python

import os

import versioneer
from setuptools import find_namespace_packages, find_packages

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
    # TODO: monkey patch abstract.CMakePlatform.compile_test_cmakelist or
    #  abstract.CMakePlatform.get_best_generator instead to chose right
    #  complier. It will produce more stable output. We may avoid setting up
    #  MSVC environment manually this way.
    windows._get_msvc_compiler_env = lambda v, t: windows.CachedEnv()

setup(
    # https://github.com/pypa/packaging-problems/issues/606
    url="https://github.com/IntelPython/dpbench",
    packages=(
        find_packages(include=["dpbench*"], exclude=["dpbench.benchmarks*"])
        + find_namespace_packages(include=["dpbench.benchmarks*"])
        + find_namespace_packages(include=["dpbench.configs*"])
    ),
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    include_package_data=True,
    package_data={
        "dpbench.migrations": ["alembic.ini"],
        "dpbench.configs": [
            "*.toml",
            "bench_info/*.toml",
            "bench_info/polybench/*.toml",
            "bench_info/polybench/stencils/*.toml",
            "bench_info/polybench/datamining/*.toml",
            "bench_info/polybench/linear-algebra/*.toml",
            "bench_info/polybench/linear-algebra/kernels/*.toml",
            "bench_info/polybench/linear-algebra/solvers/*.toml",
            "bench_info/polybench/linear-algebra/blas/*.toml",
            "bench_info/polybench/medley/*.toml",
            "bench_info/npbench/*.toml",
            "bench_info/rodinia/*.toml",
            "framework_info/*.toml",
        ],
    },
    cmake_args=cmake_args,
)
