# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Set of functions to read configuration files."""

import importlib
import logging
import os
import pkgutil
import re
import sys

import tomli

from .benchmark import Benchmark, BenchmarkImplementation, Presets
from .config import Config
from .framework import Framework
from .implementation_postfix import Implementation
from .module import Module

_REFERENCE_IMPLEMENTATIONS = {"numpy", "python"}


def read_configs(  # noqa: C901: TODO: move modules into config
    benchmarks: set[str] = None,
    implementations: set[str] = None,
    no_dpbench: bool = False,
    with_npbench: bool = False,
    with_polybench: bool = False,
    with_rodinia: bool = False,
    load_implementations: bool = True,
) -> Config:
    """Read all configuration files and populate those settings into Config.

    Args:
        benchmarks: list of benchmarks to load. None means all.
        postfixes: list of benchmark postfixes to load. None means all.

    Returns:
        Configuration object with populated configurations.
    """
    config: Config = Config()

    dirname: str = os.path.dirname(__file__)

    # TODO: move into config
    modules: list[Module] = [
        Module(
            benchmark_configs_path=os.path.join(
                dirname, "../configs/bench_info"
            ),
            benchmarks_module="dpbench.benchmarks",
            framework_configs_path=os.path.join(
                dirname, "../configs/framework_info"
            ),
            precision_dtypes_path=os.path.join(
                dirname, "../configs/precision_dtypes.toml"
            ),
        ),
    ]

    if no_dpbench:
        modules[0].benchmark_configs_path = ""

    if with_npbench:
        modules.append(
            Module(
                benchmark_configs_path=os.path.join(
                    dirname, "../configs/bench_info/npbench"
                ),
                benchmarks_module="dpbench.benchmarks.npbench",
                path=os.path.join(dirname, "../benchmarks/npbench"),
            )
        )

    if with_polybench:
        modules.append(
            Module(
                benchmark_configs_path=os.path.join(
                    dirname, "../configs/bench_info/polybench"
                ),
                benchmark_configs_recursive=True,
                benchmarks_module="dpbench.benchmarks.polybench",
                path=os.path.join(dirname, "../benchmarks/polybench"),
            )
        )

    if with_rodinia:
        modules.append(
            Module(
                benchmark_configs_path=os.path.join(
                    dirname, "../configs/bench_info/rodinia"
                ),
                benchmark_configs_recursive=True,
                benchmarks_module="dpbench.benchmarks.rodinia",
                path=os.path.join(dirname, "../benchmarks/rodinia"),
            )
        )

    for mod in modules:
        if mod.benchmark_configs_path != "":
            read_benchmarks(
                config,
                mod.benchmark_configs_path,
                recursive=mod.benchmark_configs_recursive,
                parent_package=mod.benchmarks_module,
                benchmarks=benchmarks,
            )
        if mod.framework_configs_path != "":
            read_frameworks(config, mod.framework_configs_path, implementations)
        if mod.precision_dtypes_path != "":
            read_precision_dtypes(config, mod.precision_dtypes_path)
        if mod.path != "":
            sys.path.append(mod.path)

    for framework in config.frameworks:
        config.implementations += framework.postfixes

    if implementations is None:
        implementations = {impl.postfix for impl in config.implementations}

    if load_implementations:
        for benchmark in config.benchmarks:
            read_benchmark_implementations(
                benchmark,
                implementations,
            )

        config.benchmarks = [
            benchmark
            for benchmark in config.benchmarks
            if len(benchmark.implementations) > 0
        ]

    return config


def read_benchmarks(
    config: Config,
    bench_info_dir: str,
    recursive: bool = False,
    parent_package: str = "dpbench.benchmarks",
    benchmarks: set[str] = None,
):
    """Read and populate benchmark configuration files.

    Args:
        config: Configuration object where settings should be populated.
        bench_info_dir: Path to the directory with configuration files.
        recursive: Either to load configs recursively.
        parent_package: Package that contains benchmark packages.
        benchmarks: list of benchmarks to load. None means all.
    """
    for bench_info_file in os.listdir(bench_info_dir):
        bench_info_file_path = os.path.join(bench_info_dir, bench_info_file)

        if os.path.isdir(bench_info_file_path) and recursive:
            read_benchmarks(
                config=config,
                bench_info_dir=bench_info_file_path,
                recursive=recursive,
                parent_package=parent_package + "." + bench_info_file,
                benchmarks=benchmarks,
            )

        if (
            not os.path.isfile(bench_info_file_path)
            or not bench_info_file.endswith(".toml")
            or (benchmarks and not bench_info_file[:-5] in benchmarks)
        ):
            continue

        with open(bench_info_file_path) as file:
            file_contents = file.read()

        bench_info = tomli.loads(file_contents)
        benchmark = Benchmark.from_dict(bench_info.get("benchmark"))
        if benchmark.package_path == "":
            rel_path = benchmark.relative_path.replace("/", ".")

            if benchmark.relative_path == "":
                rel_path = benchmark.module_name

            benchmark.package_path = f"{parent_package}.{rel_path}"
        config.benchmarks.append(benchmark)


def read_frameworks(
    config: Config,
    framework_info_dir: str,
    implementations: set[str] = None,
) -> None:
    """Read and populate framework configuration files.

    Args:
        config: Configuration object where settings should be populated.
        framework_info_dir: Path to the directory with configuration files.
        implementations: Set of the implementations to load. If framework
            does not have any implementation from this list - it won't be
            loaded. If set None or empty - all frameworks/implementations get
            loaded.
    """
    for framework_info_file in os.listdir(framework_info_dir):
        if not framework_info_file.endswith(".toml"):
            continue

        framework_info_file = os.path.join(
            framework_info_dir, framework_info_file
        )

        with open(framework_info_file) as file:
            file_contents = file.read()

        framework_info = tomli.loads(file_contents)
        framework_dict = framework_info.get("framework")
        if not framework_dict:
            continue
        framework = Framework.from_dict(framework_dict)
        if implementations:
            framework.postfixes = [
                postfix
                for postfix in framework.postfixes
                if postfix.postfix in implementations
                or postfix.postfix in _REFERENCE_IMPLEMENTATIONS
            ]

        if len(framework.postfixes) == 0:
            continue

        config.frameworks.append(framework)


def read_precision_dtypes(config: Config, precision_dtypes_file: str) -> None:
    """Read and populate dtype_obj data types file.

    Args:
        config: Configuration object where settings should be populated.
        precision_dtypes_file: Path to the configuration file.
    """
    with open(precision_dtypes_file) as file:
        file_contents = file.read()

    config.dtypes = tomli.loads(file_contents)


def setup_init(config: Benchmark, modules: list[str]) -> None:
    """Read and discover initialization module and function.

    Args:
        config: Benchmark configuration object where settings should be
            populated.
        modules: List of available modules for the benchmark to find init.
    """
    if config.init is None:
        return

    init_module = None
    if config.module_name in modules:
        init_module = config.module_name
    elif config.short_name in modules:
        init_module = config.short_name
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
            logging.warn(
                f"could not find init function for {config.module_name}"
            )


def discover_module_name_and_postfix(module: str, config: Config):
    """Discover real module name and postfix for the implementation.

    Args:
        module: Name of the root python module (either python file or top level
            folder for sycl).
        config: Module config.

    Returns: (module_name, postfix).
    """
    postfix = ""
    module_name = ""

    if module.endswith("sycl_native_ext"):
        module_name = (
            f"{module}.{config.module_name}_sycl._{config.module_name}_sycl"
        )
        postfix = "sycl"
    else:
        module_name = module
        if module.startswith(config.module_name):
            postfix = module[len(config.module_name) + 1 :]
        elif module.startswith(config.short_name):
            postfix = module[len(config.short_name) + 1 :]

    return module_name, postfix


def read_benchmark_implementations(
    config: Benchmark,
    implementations: set[str] = None,
) -> None:
    """Read and discover implementation modules and functions.

    Args:
        config: Benchmark configuration object where settings should be
            populated.
        implementations: List of postfixes to import. It does not affect
            initialization import.

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
    set_default_reference_implementation_postfix(config, modules)

    for module in modules:
        module_name, postfix = discover_module_name_and_postfix(module, config)

        if (
            implementations
            and (postfix not in implementations)
            and (postfix != config.reference_implementation_postfix)
            or (config.init and config.init.module_name.endswith(module_name))
        ):
            continue

        func_name: str = None
        package_path: str = f"{config.package_path}.{module_name}"

        try:
            impl_mod = importlib.import_module(package_path)

            for func in [
                module,
                f"{module}_{postfix}",
                config.module_name,
                f"{config.module_name}_{postfix}",
                "kernel",
                re.sub(r"[0-9]", "", config.module_name),
            ]:
                if hasattr(impl_mod, func):
                    func_name = func
                    break
        except Exception as e:
            logging.warn(f"Could not import module: {e}")
            import traceback

            traceback.print_exc()
            continue

        config.implementations.append(
            BenchmarkImplementation(
                postfix=postfix,
                func_name=func_name,
                module_name=module_name,
                package_path=package_path,
            )
        )


def set_default_reference_implementation_postfix(
    config: Benchmark,
    modules: set[str] = None,
):
    """Sets reference implementation postfix if not set.

    It will set it to 'numpy' or 'python' with priority to 'numpy' depending on
    the available modules.

    Args:
        config: Benchmark configuration object where settings should be
            populated.
        modules: List of modules in benchmark implementation dir.
    """
    if config.reference_implementation_postfix:
        return

    postfixes = {
        discover_module_name_and_postfix(module, config)[1]
        for module in modules
    }

    for postfix in _REFERENCE_IMPLEMENTATIONS:
        if postfix in postfixes:
            config.reference_implementation_postfix = postfix
            break


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
