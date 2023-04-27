# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
import logging
import os
import pathlib
import pkgutil
from datetime import datetime

import dpbench.benchmarks as dp_bms
import dpbench.config as cfg
import dpbench.infrastructure as dpbi
from dpbench.infrastructure.enums import ErrorCodes


def _format_ns(time_in_ns):
    time = int(time_in_ns)
    assert time >= 0
    suff = [("s", 1000_000_000), ("ms", 1000_000), ("\u03BCs", 1000), ("ns", 0)]
    for s, scale in suff:
        if time >= scale:
            scaled_time = float(time) / scale if scale > 0 else time
            return f"{scaled_time}{s} ({time} ns)"


def _print_results(result: dpbi.BenchmarkResults):
    print(
        "================ implementation "
        + result.benchmark_impl_postfix
        + " ========================\n"
        + "implementation:",
        result.benchmark_impl_postfix,
    )

    if result.error_state == ErrorCodes.SUCCESS:
        print("framework:", result.framework_name)
        print("framework version:", result.framework_version)
        print("setup time:", _format_ns(result.setup_time))
        print("warmup time:", _format_ns(result.warmup_time))
        print("teardown time:", _format_ns(result.teardown_time))
        print("max execution times:", _format_ns(result.max_exec_time))
        print("min execution times:", _format_ns(result.min_exec_time))
        print("median execution times:", _format_ns(result.median_exec_time))
        print("repeats:", result.num_repeats)
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
    conn=None,
    print_results=True,
    run_id: int = None,
):
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
            conn=conn,
            run_id=run_id,
        )
        if print_results:
            for result in results:
                _print_results(result)

    except Exception:
        logging.exception(
            "Benchmark execution failed due to the following error: "
        )
        return


def run_benchmarks(
    preset="S",
    repeat=10,
    validate=True,
    timeout=200.0,
    dbfile=None,
    print_results=True,
    run_id=None,
):
    """Run all benchmarks in the dpbench benchmark directory
    Args:
        bconfig_path (str, optional): Path to benchmark configurations.
        Defaults to None.
        preset (str, optional): Problem size. Defaults to "S".
        repeat (int, optional): Number of repetitions. Defaults to 1.
        validate (bool, optional): Whether to validate against NumPy.
        Defaults to True.
        timeout (float, optional): Timeout setting. Defaults to 10.0.
    """

    print("===============================================================")
    print("")
    print("***Start Running DPBench***")
    if not dbfile:
        dbfile = "results.db"

    dpbi.create_results_table()
    conn = dpbi.create_connection(db_file=dbfile)
    if run_id is None:
        run_id = dpbi.create_run(conn)

    for b in cfg.GLOBAL.benchmarks:
        for impl in cfg.GLOBAL.implementations:
            run_benchmark(
                benchmark=b,
                implementation_postfix=impl.postfix,
                preset=preset,
                repeat=repeat,
                validate=validate,
                timeout=timeout,
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

    dpbi.generate_impl_summary_report(conn, run_id=run_id)

    return dbfile
