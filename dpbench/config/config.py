# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration root related configuration classes."""

from dataclasses import dataclass, field

from .benchmark import Benchmark
from .framework import Framework
from .implementation_postfix import Implementation


@dataclass
class Config:
    """Root of the configuration."""

    frameworks: list[Framework] = field(default_factory=list)
    benchmarks: list[Benchmark] = field(default_factory=list)
    # deprecated. Use frameworks.postfixes instead
    implementations: list[Implementation] = field(default_factory=list)
    dtypes: dict[str, dict[str, str]] = field(default_factory=dict)
