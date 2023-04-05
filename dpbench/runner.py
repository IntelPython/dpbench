# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
import logging
import pathlib
import pkgutil
from datetime import datetime

import dpbench.benchmarks as dp_bms
import dpbench.infrastructure as dpbi
from dpbench.infrastructure.enums import ErrorCodes


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
        print("setup time:", result.setup_time)
        print("warmup time:", result.warmup_time)
        print("teardown time:", result.teardown_time)
        print("max execution times:", result.max_exec_time)
        print("min execution times:", result.min_exec_time)
        print("median execution times:", result.median_exec_time)
        print("repeats:", result.num_repeats)
        print("preset:", result.preset)
        print("validated:", result.validation_state.name)
    else:
        print("error states:", result.error_state)
        print("error msg:", result.error_msg)


def list_available_benchmarks():
    """Return the list of available benchmarks that ae in the
    dpbench.benchmarks module.
    """

    submods = [
        submod.name
        for submod in pkgutil.iter_modules(dp_bms.__path__)
        if submod.ispkg
    ]

    return submods


def list_possible_implementations() -> list[str]:
    """Returns list of implementation postfixes, which are keys in
    configs/impl_postfix.json.
    """
    parent_folder = pathlib.Path(__file__).parent.absolute()
    impl_postfix_json = parent_folder.joinpath("configs", "impl_postfix.json")

    try:
        with open(impl_postfix_json) as json_file:
            return [entry["impl_postfix"] for entry in json.load(json_file)]
    except Exception:
        logging.exception(
            "impl postfix JSON file {b} could not be opened.".format(
                b="impl_post_fix.json"
            )
        )
        raise


def run_benchmark(
    bname,
    implementation_postfix=None,
    fconfig_path=None,
    bconfig_path=None,
    preset="S",
    repeat=10,
    validate=True,
    timeout=1000.0,
    conn=None,
    run_datetime=None,
    print_results=True,
):
    print("")
    print("================ Benchmark " + bname + " ========================")
    print("")
    bench = None

    allowed_impl_postfixes = list_possible_implementations()

    try:
        benchmod = importlib.import_module("dpbench.benchmarks." + bname)
        bench = dpbi.Benchmark(
            benchmod,
            bconfig_path=bconfig_path,
            allowed_implementation_postfixes=allowed_impl_postfixes,
        )
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
            run_datetime=run_datetime,
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
    fconfig_path=None,
    bconfig_path=None,
    preset="S",
    repeat=10,
    validate=True,
    timeout=1000.0,
    dbfile=None,
    print_results=True,
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
    datetime_str = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    if not dbfile:
        dbfile = "results_" + datetime_str + ".db"

    conn = dpbi.create_connection(db_file=dbfile)
    dpbi.create_results_table(conn)

    impl_postfixes = list_possible_implementations()

    for b in list_available_benchmarks():
        for impl in impl_postfixes:
            run_benchmark(
                bname=b,
                implementation_postfix=impl,
                fconfig_path=fconfig_path,
                bconfig_path=bconfig_path,
                preset=preset,
                repeat=repeat,
                validate=validate,
                timeout=timeout,
                conn=conn,
                run_datetime=datetime_str,
                print_results=print_results,
            )

    print("")
    print("===============================================================")
    print("")
    print("***All the Tests are Finished. DPBench is Done.***")
    print("")
    print("===============================================================")
    print("")

    dpbi.generate_impl_summary_report(dbfile)

    return dbfile
