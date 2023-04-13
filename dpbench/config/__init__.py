# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Config sub-package with benchmark, frameworks settings, etc.

The config sub-package is a module that contains a set of data classes that
define a benchmarking framework's configuration. The data classes are designed
to provide a structured way to define and store benchmark data.
"""

import json
import os

from .benchmark import Benchmark
from .config import Config
from .framework import Framework
from .implementaion_postfix import Implementation


def read_configs(dirname: str = os.path.dirname(__file__)) -> Config:
    """Read all configuration files and populates those settings into Config."""
    C: Config = Config([], [], [])

    impl_postfix_file = os.path.join(dirname, "../configs/impl_postfix.json")
    bench_info_dir = os.path.join(dirname, "../configs/bench_info")
    framework_info_dir = os.path.join(dirname, "../configs/framework_info")

    for bench_info_file in os.listdir(bench_info_dir):
        if not bench_info_file.endswith(".json"):
            continue

        bench_info_file = os.path.join(bench_info_dir, bench_info_file)

        with open(bench_info_file) as file:
            file_contents = file.read()

        bench_info = json.loads(file_contents)
        benchmark = Benchmark.from_dict(bench_info.get("benchmark"))
        C.benchmarks.append(benchmark)

    for framework_info_file in os.listdir(framework_info_dir):
        if not framework_info_file.endswith(".json"):
            continue

        framework_info_file = os.path.join(
            framework_info_dir, framework_info_file
        )

        with open(framework_info_file) as file:
            file_contents = file.read()

        framework_info = json.loads(file_contents)
        framework_dict = framework_info.get("framework")
        if framework_dict:
            framework = Framework.from_dict(framework_dict)
            C.frameworks.append(framework)

    with open(impl_postfix_file) as file:
        file_contents = file.read()

    implementaion_postfixes = json.loads(file_contents)
    for impl in implementaion_postfixes:
        implementation = Implementation.from_dict(impl)
        C.implementations.append(implementation)

    return C


"""Use this variable for reading configurations"""
C: Config = read_configs()
