# Copyright 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache 2.0

#!/usr/bin/env python

import dpctl
from setuptools import find_packages
from skbuild import setup

dpctl_include_dir = dpctl.get_include()

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
    cmake_args=["-DDpctl_INCLUDE_DIRS=" + dpctl_include_dir],
)
