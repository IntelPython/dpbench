# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0

from .benchmark import Benchmark
from .framework import Framework, generate_framework
from .numba_framework import NumbaFramework
from .dpnp_framework import DpnpFramework
from .numba_dppy_framework import NumbaDppyFramework
from .test import Test
from .utilities import (
    benchmark,
    create_connection,
    str2bool,
    time_to_ms,
    validate,
)

__all__ = [
    "Benchmark",
    "Framework",
    "NumbaFramework",
    "Test",
    "create_connection",
    "generate_framework",
    "benchmark",
    "validate",
    "time_to_ms",
    "str2bool",
]
