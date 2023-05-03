# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Config sub-package with benchmark, frameworks settings, etc.

The config sub-package is a module that contains a set of data classes that
define a benchmarking framework's configuration. The data classes are designed
to provide a structured way to define and store benchmark data.
"""


from .benchmark import Benchmark, BenchmarkImplementation
from .config import Config
from .framework import Framework
from .implementation_postfix import Implementation
from .reader import read_configs

"""Use this variable for reading configurations"""
GLOBAL: Config = Config()

__all__ = [
    "GLOBAL",
    "Benchmark",
    "BenchmarkImplementation",
    "Config",
    "Framework",
    "Implementation",
]
