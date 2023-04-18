# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Set of functions to read configuration files."""

import importlib
import json
import os

from .benchmark import Benchmark, BenchmarkImplementation
from .config import Config
from .framework import Framework
from .implementation_postfix import Implementation


def read_configs(dirname: str = os.path.dirname(__file__)) -> Config:
    """Read all configuration files and populate those settings into Config.

    Args:
        dirname: Path to the directory with configuration files.

    Returns:
        Configuration object with populated configurations.
    """
    config: Config = Config([], [], [])

    impl_postfix_file = os.path.join(dirname, "../configs/impl_postfix.json")
    bench_info_dir = os.path.join(dirname, "../configs/bench_info")
    framework_info_dir = os.path.join(dirname, "../configs/framework_info")

    read_benchmarks(config, bench_info_dir)
    read_frameworks(config, framework_info_dir)
    read_implementation_postfixes(config, impl_postfix_file)

    for benchmark in config.benchmarks:
        read_benchmark_implementations(benchmark, config.implementations)

    return config


def read_benchmarks(config: Config, bench_info_dir: str):
    """Read and populate benchmark configuration files.

    Args:
        config: Configuration object where settings should be populated.
        bench_info_dir: Path to the directory with configuration files.

    Returns: nothing.
    """
    for bench_info_file in os.listdir(bench_info_dir):
        if not bench_info_file.endswith(".json"):
            continue

        bench_info_file = os.path.join(bench_info_dir, bench_info_file)

        with open(bench_info_file) as file:
            file_contents = file.read()

        bench_info = json.loads(file_contents)
        benchmark = Benchmark.from_dict(bench_info.get("benchmark"))
        config.benchmarks.append(benchmark)


def read_frameworks(config: Config, framework_info_dir: str) -> None:
    """Read and populate framework configuration files.

    Args:
        config: Configuration object where settings should be populated.
        framework_info_dir: Path to the directory with configuration files.

    Returns: nothing.
    """
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
            config.frameworks.append(framework)


def read_implementation_postfixes(
    config: Config, impl_postfix_file: str
) -> None:
    """Read and populate implementation postfix configuration file.

    Args:
        config: Configuration object where settings should be populated.
        impl_postfix_file: Path to the configuration file.

    Returns: nothing.
    """
    with open(impl_postfix_file) as file:
        file_contents = file.read()

    implementation_postfixes = json.loads(file_contents)
    for impl in implementation_postfixes:
        implementation = Implementation.from_dict(impl)
        config.implementations.append(implementation)


def read_benchmark_implementations(
    config: Benchmark, implementations: Implementation
) -> None:
    """Read and discover implementation modules and functions.

    Args:
        config: Benchmark configuration object where settings should be
            populated.
        implementations: Prepopulated list of implementations.

    Returns: nothing.

    Raises:
        RuntimeError: Implementation file does not match any known postfix.
    """
    if config.implementations:
        return

    mod = importlib.import_module("dpbench.benchmarks." + config.module_name)

    modules: list[str] = [
        m
        for m in mod.__loader__.get_resource_reader().contents()
        if m.startswith(config.module_name)
    ]

    for module in modules:
        postfix = ""
        module_name = ""

        if module.endswith(".py"):
            module_name = module[:-3]
            postfix = module_name[len(config.module_name) + 1 :]
        elif module.endswith("sycl_native_ext"):
            module_name = (
                f"{module}.{config.module_name}_sycl._{config.module_name}_sycl"
            )
            postfix = "sycl"

        if config.init.module_name.endswith(module_name):
            continue

        if postfix not in [impl.postfix for impl in implementations]:
            raise RuntimeError(
                f"Could not recognize postfix {postfix} as known postfix for"
                + f" file {module} in {config.module_name}"
            )

        func_name: str = None
        package_path: str = (
            f"dpbench.benchmarks.{config.module_name}.{module_name}"
        )

        try:
            impl_mod = importlib.import_module(package_path)

            for func in [config.module_name, f"{config.module_name}_{postfix}"]:
                if hasattr(impl_mod, func):
                    func_name = func
                    break
        except Exception:
            continue

        config.implementations.append(
            BenchmarkImplementation(
                postfix=postfix,
                func_name=func_name,
                module_name=module_name,
                package_path=package_path,
            )
        )
