# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration root related configuration classes."""

from dataclasses import dataclass

from .benchmark import Benchmark
from .framework import Framework
from .implementation_postfix import Implementation


@dataclass
class Config:
    """Root of the configuration."""

    frameworks: list[Framework]
    benchmarks: list[Benchmark]
    implementations: list[Implementation]
    dtypes: dict[str, dict[str, str]]
