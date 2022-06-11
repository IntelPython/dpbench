# Copyright 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache 2.0

from skbuild import setup
from setuptools import find_packages
import dpctl

dpctl_include_dir = dpctl.get_include()
setup(
    name="black_scholes_sycl_native_ext",
    version="0.0.1",
    cmake_args=["-DDpctl_INCLUDE_DIRS=" + dpctl_include_dir],
    description="SYCL implementation for Black-Scholes",
    author="Intel Scripting",
    license="Apache 2.0",
    packages=find_packages(include=["*"]),
    include_package_data=True,
)
