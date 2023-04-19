# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Set of functions to read configuration files."""

import importlib
import json
import logging
import os
import pkgutil
import re
import sys
from typing import Callable

from .benchmark import Benchmark, BenchmarkImplementation, Presets
from .config import Config
from .framework import Framework
from .implementation_postfix import Implementation
from .module import Module


def read_configs(
    benchmarks: list[str] = None,
    postfixes: list[str] = None,
) -> Config:
    """Read all configuration files and populate those settings into Config.

    Args:
        benchmarks: list of benchmarks to load. None means all.
        postfixes: list of benchmark postfixes to load. None means all.

    Returns:
        Configuration object with populated configurations.
    """
    config: Config = Config([], [], [])

    dirname: str = os.path.dirname(__file__)

    modules: list[Module] = [
        Module(
            benchmark_configs_path=os.path.join(
                dirname, "../configs/bench_info"
            ),
            benchmarks_module="dpbench.benchmarks",
            framework_configs_path=os.path.join(
                dirname, "../configs/framework_info"
            ),
            impl_postfix_path=os.path.join(
                dirname, "../configs/impl_postfix.json"
            ),
        ),
    ]

    no_dpbench = os.getenv("NO_DPBENCH")
    if no_dpbench:
        modules[0].benchmark_configs_path = ""

    npbench_root = os.getenv("NPBENCH_ROOT")
    if npbench_root:
        modules.append(
            Module(
                benchmark_configs_path=os.path.join(npbench_root, "bench_info"),
                benchmarks_module="npbench.benchmarks",
                path=npbench_root,
            )
        )

    for mod in modules:
        if mod.benchmark_configs_path != "":
            read_benchmarks(
                config,
                mod.benchmark_configs_path,
                parent_package=mod.benchmarks_module,
                benchmarks=benchmarks,
            )
        if mod.framework_configs_path != "":
            read_frameworks(config, mod.framework_configs_path)
        if mod.impl_postfix_path != "":
            read_implementation_postfixes(config, mod.impl_postfix_path)
        if mod.path != "":
            sys.path.append(mod.path)

    for benchmark in config.benchmarks:
        postfixes_tmp = postfixes
        if postfixes_tmp is None:
            postfixes_tmp = [impl.postfix for impl in config.implementations]
        read_benchmark_implementations(
            benchmark,
            config.implementations,
            postfixes=postfixes_tmp,
        )

    if npbench_root:
        fix_npbench_configs(config.benchmarks)

    return config


def read_benchmarks(
    config: Config,
    bench_info_dir: str,
    parent_package: str = "dpbench.benchmarks",
    benchmarks: list[str] = None,
):
    """Read and populate benchmark configuration files.

    Args:
        config: Configuration object where settings should be populated.
        bench_info_dir: Path to the directory with configuration files.
        parent_package: Package that contains benchmark packages.
        benchmarks: list of benchmarks to load. None means all.

    Returns: nothing.
    """
    for bench_info_file in os.listdir(bench_info_dir):
        if not bench_info_file.endswith(".json"):
            continue

        if benchmarks and not bench_info_file[:-5] in benchmarks:
            continue

        bench_info_file = os.path.join(bench_info_dir, bench_info_file)

        with open(bench_info_file) as file:
            file_contents = file.read()

        bench_info = json.loads(file_contents)
        benchmark = Benchmark.from_dict(bench_info.get("benchmark"))
        if benchmark.package_path == "":
            rel_path = benchmark.relative_path.replace("/", ".")

            if benchmark.relative_path == "":
                rel_path = benchmark.module_name

            benchmark.package_path = f"{parent_package}.{rel_path}"
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


def setup_init(config: Benchmark, modules: list[str]) -> None:
    """Read and discover initialization module and function.

    Args:
        config: Benchmark configuration object where settings should be
            populated.
        modules: List of available modules for the benchmark to find init.

    Returns: nothing.
    """
    if config.init is None:
        return

    init_module = None
    if config.module_name in modules:
        init_module = config.module_name
    elif config.module_name + "_initialize" in modules:
        init_module = config.module_name + "_initialize"

    if init_module:
        if config.init.module_name == "":
            config.init.module_name = init_module
        if config.init.package_path == "":
            config.init.package_path = config.package_path + "." + init_module
        if config.init.func_name == "":
            config.init.func_name = "initialize"

        impl_mod = importlib.import_module(config.init.package_path)

        if not hasattr(impl_mod, config.init.func_name):
            print(
                f"WARNING: could not find init function for {config.module_name}"
            )


def read_benchmark_implementations(
    config: Benchmark,
    known_implementations: list[Implementation],
    postfixes: list[str] = None,
) -> None:
    """Read and discover implementation modules and functions.

    Args:
        config: Benchmark configuration object where settings should be
            populated.
        postfixes: List of postfixes to import. Set it to None to import all
            available implementations. It does not affect initialization import.
        implementations: Prepopulated list of implementations.

    Returns: nothing.

    Raises:
        RuntimeError: Implementation file does not match any known postfix.
    """
    if config.implementations:
        return

    try:
        mod = importlib.import_module(config.package_path)
    except ModuleNotFoundError:
        logging.warning(f"Module not found: {config.package_path}")
        return

    modules: list[str] = [
        name
        for _, name, _ in pkgutil.iter_modules(
            mod.__spec__.submodule_search_locations
        )
    ]

    setup_init(config, modules)

    for module in modules:
        postfix = ""
        module_name = ""

        if module.endswith("sycl_native_ext"):
            module_name = (
                f"{module}.{config.module_name}_sycl._{config.module_name}_sycl"
            )
            postfix = "sycl"
        else:
            module_name = module
            postfix = module[len(config.module_name) + 1 :]

        if postfixes and postfix not in postfixes:
            continue

        if config.init and config.init.module_name.endswith(module_name):
            continue

        if postfix not in [impl.postfix for impl in known_implementations]:
            logging.warning(f"Skipping postfix: {module}")
            continue

        func_name: str = None
        package_path: str = f"{config.package_path}.{module_name}"

        try:
            impl_mod = importlib.import_module(package_path)

            for func in [
                config.module_name,
                f"{config.module_name}_{postfix}",
                "kernel",
                re.sub(r"[0-9]", "", config.module_name),
            ]:
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


def get_benchmark_index(configs: list[Benchmark], module_name: str) -> int:
    """Finds configuration index by module name."""
    return next(
        (
            i
            for i, config in enumerate(configs)
            if config.module_name == module_name
        ),
        None,
    )


def fix_npbench_configs(configs: list[Benchmark]):
    """Applies configuration fixes for some npbench benchmarks.

    Fixes required due to the difference in framework implementations.
    """
    index = get_benchmark_index(configs, "mandelbrot1")
    if index is not None:
        configs[index] = modify_args(
            configs[index], modifier=lambda s: s.lower()
        )

    index = get_benchmark_index(configs, "mandelbrot2")
    if index is not None:
        configs[index] = modify_args(
            configs[index],
            modifier=lambda s: "itermax" if s == "maxiter" else s.lower(),
        )

    index = get_benchmark_index(configs, "conv2d")
    if index is not None:
        config = configs[index]

        config.module_name = "conv2d_bias"
        configs[index] = config

        for impl in config.implementations:
            impl.func_name = "conv2d_bias"

    index = get_benchmark_index(configs, "nbody")
    if index is not None:
        configs[index].output_args.append("pos")
        configs[index].output_args.append("vel")

    index = get_benchmark_index(configs, "scattering_self_energies")
    if index is not None:
        configs[index].output_args.append("Sigma")

    index = get_benchmark_index(configs, "correlation")
    if index is not None:
        configs[index].output_args.append("data")

    index = get_benchmark_index(configs, "doitgen")
    if index is not None:
        configs[index].output_args.append("A")


def modify_args(config: Benchmark, modifier: Callable[[str], str]) -> Benchmark:
    """Applies modifier to function argument names.

    Current implementation applies modifier to
      - all presets keys, not preset names;
      - all input_args;
      - all array_args;
      - all output_args.
    """
    config.parameters = Presets(
        {
            preset: {modifier(k): v for k, v in parameters.items()}
            for preset, parameters in config.parameters.items()
        }
    )

    config.input_args = [modifier(arg) for arg in config.input_args]
    config.array_args = [modifier(arg) for arg in config.array_args]
    config.output_args = [modifier(arg) for arg in config.output_args]

    return config
