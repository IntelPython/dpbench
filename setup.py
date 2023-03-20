# Copyright 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache 2.0

#!/usr/bin/env python

import os
import shutil
import subprocess

import dpctl
from setuptools import find_packages
from skbuild import setup

import dpbench.benchmarks


# Init an implementation directory with .py files in it
def init_implementation_dir(benchmark_path, p, do_imports=True, imp_ext=True):
    implementation_ext = [
        "cupy",
        "dace",
        "legate",
        "numba_n",
        "numba_np",
        "numba_npr",
        "numba_o",
        "numba_op",
        "numpy",
        "pythran",
    ]
    f = open(os.path.join(benchmark_path, "__init__.py"), "w")
    if do_imports:
        f.write("from .{0:s} import {0:s} as {0:s}\n".format(p))
        for ext in implementation_ext:
            if os.path.exists(
                os.path.join(benchmark_path, "{0:s}_{1:s}.py".format(p, ext))
            ):
                f.write(
                    "from .{0:s}_{1:s} import {0:s} as {0:s}_{1:s}\n".format(
                        p, ext
                    )
                )
        f.write("\n\n")

    f.write("__all__ = [\n")
    if imp_ext:
        f.write('    "{0:s}",\n'.format(p))
        for ext in implementation_ext:
            if os.path.exists(
                os.path.join(benchmark_path, "{0:s}_{1:s}.py".format(p, ext))
            ):
                f.write('    "{0:s}_{1:s}",\n'.format(p, ext))
    else:
        for d in os.listdir(benchmark_path):
            sub_path = os.path.join(benchmark_path, d)
            if os.path.isdir(sub_path):
                f.write('    "{:s}"\n'.format(d))
    f.write("]\n")
    f.close()


# decide CC/CXX paths
ccpath = shutil.which("icx")
cxxpath = shutil.which("icpx")
if ccpath or cxxpath:
    os.environ["CC"] = ccpath
    os.environ["CXX"] = cxxpath
    print("CC={:s}".format(os.environ["CC"]))
    print("CXX={:s}".format(os.environ["CXX"]))
else:
    raise SystemError(
        "icx/icpx compilers don't exist in the system, "
        + "install Intel OneAPI or dpcpp_linux-64 (through conda)."
    )


# Get cmakedirs
dpctl_include_dir = dpctl.get_include()
pybind_cmake_path = subprocess.check_output(
    ["python", "-m", "pybind11", "--cmakedir"], encoding="utf-8"
).split()[0]
dpctl_cmake_path = subprocess.check_output(
    ["python", "-m", "dpctl", "--cmakedir"], encoding="utf-8"
).split()[0]

# update npbench submodule
subprocess.check_call(["git", "submodule", "update", "--init"])

# setup __init__.py for each benchmark module
npbench_path = os.path.join(dpbench.benchmarks.__path__[0], "npbench")
f = open(os.path.join(npbench_path, "__init__.py"), "w")
f.close()

npbench_benchmark_path = os.path.join(npbench_path, "npbench/benchmarks")
f = open(os.path.join(npbench_benchmark_path, "__init__.py"), "w")
f.close()

for p in os.listdir(npbench_benchmark_path):
    if p == "polybench" or p == "pythran" or p == "weather_stencils":
        benchmark_path = os.path.join(npbench_benchmark_path, p)
        if os.path.isdir(benchmark_path):
            init_implementation_dir(
                benchmark_path, p, do_imports=False, imp_ext=False
            )
        for p_ in os.listdir(benchmark_path):
            benchmark_path_ = os.path.join(benchmark_path, p_)
            if os.path.isdir(benchmark_path_):
                init_implementation_dir(benchmark_path_, p_)
    else:
        benchmark_path = os.path.join(npbench_benchmark_path, p)
        if os.path.isdir(benchmark_path):
            init_implementation_dir(benchmark_path, p)

# setup
setup(
    name="dpbench",
    version="0.0.1",
    url="https://https://github.com/IntelPython/dpbench",
    author="Intel Corp.",
    author_email="diptorup.deb@intel.com",
    description="dpBench",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2.0 License",
        "Operating System :: Linux",
    ],
    packages=(
        find_packages(include=["*"])
        + find_packages(where="./dpbench/benchmarks/*/*")
    ),
    python_requires=">=3.8",
    include_package_data=True,
    install_requires=[
        "numpy",
        "numba",
    ],
    cmake_args=[
        "-DDpctl_INCLUDE_DIRS=" + dpctl_include_dir,
        "-Dpybind11_DIR=" + pybind_cmake_path,
        "-DDPCTL_MODULE_PATH=" + dpctl_cmake_path,
    ],
)
