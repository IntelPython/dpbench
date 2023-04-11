# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
import logging
import os
import pathlib
import pkgutil
import sqlite3
from datetime import datetime

import sqlalchemy

import dpbench.benchmarks as dp_bms
import dpbench.config as cfg
import dpbench.infrastructure as dpbi
from dpbench.infrastructure.enums import ErrorCodes
from dpbench.infrastructure.frameworks.framework import Framework


def _format_ns(time_in_ns):
    time = int(time_in_ns)
    assert time >= 0
    suff = [("s", 1000_000_000), ("ms", 1000_000), ("\u03BCs", 1000), ("ns", 0)]
    for s, scale in suff:
        if time >= scale:
            scaled_time = float(time) / scale if scale > 0 else time
            return f"{scaled_time}{s} ({time} ns)"


def _print_results(result: dpbi.BenchmarkResults, framework: Framework):
    print(
        "================ implementation "
        + result.impl_postfix
        + " ========================\n"
        + "implementation:",
        result.impl_postfix,
    )

    if result.error_state == ErrorCodes.SUCCESS:
        print("framework:", framework.fname)
        print("framework version:", framework.version())
        print("setup time:", _format_ns(result.setup_time))
        print("warmup time:", _format_ns(result.warmup_time))
        print("teardown time:", _format_ns(result.teardown_time))
        print("max execution times:", _format_ns(result.max_exec_time))
        print("min execution times:", _format_ns(result.min_exec_time))
        print("median execution times:", _format_ns(result.median_exec_time))
        print("repeats:", result.repeats)
        print("preset:", result.preset)
        print("validated:", result.validation_state)
    else:
        print("error states:", result.error_state)
        print("error msg:", result.error_msg)


def get_benchmark(
    benchmark: cfg.Benchmark = None,
    benchmark_name: str = "",
) -> cfg.Benchmark:
    """Returns benchmark config if it is not none, otherwise returns benchmark
    config by name."""
    if benchmark is not None:
        return benchmark

    return next(
        b for b in cfg.GLOBAL.benchmarks if b.module_name == benchmark_name
    )


def run_benchmark(
    bname: str = "",
    benchmark: cfg.Benchmark = None,
    implementation_postfix=None,
    preset="S",
    repeat=10,
    validate=True,
    timeout=200.0,
    precision=None,
    conn=None,
    print_results=True,
    run_id: int = None,
):
    """Run specific benchmark.

    Args:
        bname (str, semi-optional): Name of the benchmark. Either name, either
            configuration must be provided.
        benchmark (Benchmark, semi-optional): Benchmark configuration. Either
            name, either configuration must be provided.
        implementation_postfix: (str, optional): Implementation postfixes
            to be executed. If not provided, all possible implementations will
            be executed.
        preset (str, optional): Problem size. Defaults to "S".
        repeat (int, optional): Number of repetitions. Defaults to 1.
        validate (bool, optional): Whether to validate against NumPy.
            Defaults to True.
        timeout (float, optional): Timeout setting. Defaults to 10.0.
        precision (str, optional): Precision to set for input types. If not provided,
            precision used in benchmark initialization is retained.
        conn: connection to database. If not provided results won't be stored.
        print_results (bool, optional): Either print results. Defaults to True.
        run_id (int, optional): Either store result to specific run_id.
            If not provided, new run_id will be created.

    Returns: nothing.
    """
    bench_cfg = get_benchmark(benchmark=benchmark, benchmark_name=bname)
    bname = bench_cfg.name
    print("")
    print("================ Benchmark " + bname + " ========================")
    print("")
    bench = None

    try:
        bench = dpbi.Benchmark(bench_cfg)
    except Exception:
        logging.exception(
            "Skipping the benchmark execution due to the following error: "
        )
        return

    try:
        results = bench.run(
            implementation_postfix=implementation_postfix,
            preset=preset,
            repeat=repeat,
            validate=validate,
            timeout=timeout,
            precision=precision,
            conn=conn,
            run_id=run_id,
        )
        if print_results:
            for result, framework in results:
                _print_results(result, framework)

    except Exception:
        logging.exception(
            "Benchmark execution failed due to the following error: "
        )
        return


def run_benchmarks(
    conn: sqlalchemy.Engine,
    preset="S",
    repeat=10,
    validate=True,
    timeout=200.0,
    precision=None,
    print_results=True,
    run_id=None,
    implementations: list[str] = None,
):
    """Run all benchmarks in the dpbench benchmark directory

    Args:
        preset (str, optional): Problem size. Defaults to "S".
        repeat (int, optional): Number of repetitions. Defaults to 1.
        validate (bool, optional): Whether to validate against NumPy.
            Defaults to True.
        timeout (float, optional): Timeout setting. Defaults to 10.0.
        precision (str, optional): Precision to set for input types. If not provided,
            precision used in benchmark initialization is retained.
        print_results (bool, optional): Either print results. Defaults to True.
        run_id (int, optional): Either store result to specific run_id.
            If not provided, new run_id will be created.
        implementations: (list[str], optional): List of implementation postfixes
            to be executed. If not provided, all possible implementations will
            be executed.

    Returns: nothing.
    """

    print("===============================================================")
    print("")
    print("***Start Running DPBench***")

    if not implementations:
        implementations = [impl.postfix for impl in cfg.GLOBAL.implementations]

    for b in cfg.GLOBAL.benchmarks:
        for impl in implementations:
            run_benchmark(
                benchmark=b,
                implementation_postfix=impl,
                preset=preset,
                repeat=repeat,
                validate=validate,
                timeout=timeout,
                precision=precision,
                conn=conn,
                print_results=print_results,
                run_id=run_id,
            )

    print("")
    print("===============================================================")
    print("")
    print("***All the Tests are Finished. DPBench is Done.***")
    print("")
    print("===============================================================")
    print("")

    if print_results:
        print_report(conn, run_id=run_id, implementations=implementations)


def print_report(
    conn: sqlalchemy.Engine,
    run_id: int,
    implementations: set[str],
    comparison_pairs: list[tuple[str, str]] = [],
):
    if not implementations:
        implementations = {impl.postfix for impl in cfg.GLOBAL.implementations}

    implementations = list(implementations)
    implementations.sort()

    dpbi.generate_impl_summary_report(
        conn, run_id=run_id, implementations=implementations
    )

    dpbi.generate_performance_report(
        conn,
        run_id=run_id,
        implementations=implementations,
        headless=True,
    )

    dpbi.generate_comparison_report(
        conn,
        run_id=run_id,
        implementations=implementations,
        comparison_pairs=comparison_pairs,
        headless=True,
    )

    unexpected_failures = dpbi.get_unexpected_failures(conn, run_id=run_id)

    if len(unexpected_failures) > 0:
        raise ValueError(
            f"Unexpected benchmark implementations failed: {unexpected_failures}.",
        )
