# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Module related configuration classes."""

from dataclasses import dataclass


@dataclass
class Module:
    """Benchmark set configuration."""

    benchmark_configs_path: str = ""
    benchmark_configs_recursive: bool = False
    framework_configs_path: str = ""
    impl_postfix_path: str = ""

    benchmarks_module: str = ""
    path: str = ""
