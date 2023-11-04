# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Run subcommand package."""

import argparse
import logging

import sqlalchemy

import dpbench.config as cfg
import dpbench.infrastructure as dpbi
from dpbench.infrastructure.benchmark_runner import BenchmarkRunner, RunConfig
from dpbench.infrastructure.datamodel import Postfix, store_postfix
from dpbench.infrastructure.frameworks.fabric import build_framework

from ._namespace import Namespace


def add_run_arguments(parser: argparse.ArgumentParser):
    """Add arguments for the run subcommand.

    Args:
        parser: argument parser where arguments will be populated.
    """
    parser.add_argument(
        "-p",
        "--preset",
        choices=["S", "M16Gb", "M", "L"],
        type=str,
        nargs="?",
        default="S",
        help="Preset to use for benchmark execution.",
    )
    parser.add_argument(
        "-s",
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set if the validation will be run for each benchmark.",
    )
    parser.add_argument(
        "--dpbench",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set if run dpbench benchmarks.",
    )
    parser.add_argument(
        "--npbench",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Set if run npbench benchmarks.",
    )
    parser.add_argument(
        "--polybench",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Set if run polybench benchmarks.",
    )
    parser.add_argument(
        "--rodinia",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Set if run rodinia benchmarks.",
    )
    parser.add_argument(
        "-r",
        "--repeat",
        type=int,
        nargs="?",
        default=10,
        help="Number of repeats for each benchmark.",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        nargs="?",
        default=200.0,
        help="Timeout time in seconds for each benchmark execution.",
    )
    parser.add_argument(
        "--precision",
        choices=["single", "double"],
        type=str,
        nargs="?",
        default=None,
        help="Data precision to use for array initialization.",
    )
    parser.add_argument(
        "--print-results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the result summary or not",
    )
    parser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Either to save execution into database.",
    )
    parser.add_argument(
        "--sycl-device",
        type=str,
        nargs="?",
        default=None,
        help="Sycl device to overwrite for framework configurations.",
    )
    parser.add_argument(
        "--skip-expected-failures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Either to save execution into database.",
    )


def _find_framework_config(implementation: str) -> cfg.Framework:
    framework = None

    for f in cfg.GLOBAL.frameworks:
        for impl in f.postfixes:
            if impl.postfix == implementation:
                framework = f

    return framework


def execute_run(args: Namespace, conn: sqlalchemy.Engine):
    """Execute run sub command.

    Run subcommand reads configuration and runs benchmarks.

    Args:
        args: object with all input arguments.
        conn: database connection.
    """
    cfg.GLOBAL = cfg.read_configs(
        benchmarks=args.benchmarks,
        implementations=set(args.implementations),
        no_dpbench=not args.dpbench,
        with_npbench=args.npbench,
        with_polybench=args.polybench,
        with_rodinia=args.rodinia,
    )

    if args.all_implementations:
        args.implementations = [
            impl.postfix for impl in cfg.GLOBAL.implementations
        ]

    if args.sycl_device:
        for framework in cfg.GLOBAL.frameworks:
            framework.sycl_device = args.sycl_device

    if args.save and args.run_id is None:
        args.run_id = dpbi.create_run(conn)

    runner = BenchmarkRunner()

    implementation_descriptions = {
        impl.postfix: impl.description for impl in cfg.GLOBAL.implementations
    }
    for implementation in args.implementations:
        framework_config = _find_framework_config(implementation)

        if not framework_config:
            logging.error(
                f"Could not find framework for {implementation} implementation"
            )
            continue

        framework = build_framework(framework_config)

        store_postfix(
            conn=conn,
            postfix=Postfix(
                run_id=args.run_id,
                postfix=implementation,
                description=implementation_descriptions[implementation],
                device=framework.device_info,
            ),
        )

    for benchmark in cfg.GLOBAL.benchmarks:
        print("")
        print(
            f"================ Benchmark {benchmark.name} ({benchmark.module_name}) ========================"
        )
        print("")

        for implementation in args.implementations:
            framework = _find_framework_config(implementation)

            if not framework:
                continue

            logging.info(
                f"Running {benchmark.module_name} ({implementation}) on {framework.simple_name}"
            )

            runner.run_benchmark_and_save(
                RunConfig(
                    conn=conn,
                    benchmark=benchmark,
                    framework=framework,
                    implementation=implementation,
                    preset=args.preset,
                    repeat=args.repeat,
                    validate=args.validate,
                    timeout=args.timeout,
                    precision=args.precision,
                    print_results=args.print_results,
                    run_id=args.run_id,
                    skip_expected_failures=args.skip_expected_failures,
                )
            )

    runner.close_connections()
