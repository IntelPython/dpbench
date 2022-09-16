# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache 2.0 License

import importlib
import pkgutil
import warnings

import dpbench.benchmarks as dp_bms
import dpbench.infrastructure as dpbi


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


def run_benchmark(
    bname,
    implementation_postfix=None,
    fconfig_path=None,
    bconfig_path=None,
    preset="S",
    repeat=10,
    validate=True,
    timeout=200.0,
):
    print("")
    print("================ Benchmark " + bname + " ========================")
    print("")

    try:
        benchmod = importlib.import_module("dpbench.benchmarks." + bname)
        bench = dpbi.Benchmark(benchmod, bconfig_path=bconfig_path)
    except Exception as e:
        warnings.warn(
            "Skipping the benchmark execution due to the following error: "
            + e.__str__
        )
        return

    try:
        bench.run(
            implementation_postfix=implementation_postfix,
            preset=preset,
            repeat=repeat,
            validate=validate,
            timeout=timeout,
        )
    except Exception as e:
        warnings.warn(
            "Benchmark execution failed due to the following error: "
            + e.__str__
        )
        return


def run_benchmarks(
    fconfig_path=None,
    bconfig_path=None,
    preset="S",
    repeat=1,
    validate=True,
    timeout=10.0,
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

    for b in list_available_benchmarks():
        run_benchmark(
            bname=b,
            fconfig_path=fconfig_path,
            bconfig_path=bconfig_path,
            preset=preset,
            repeat=repeat,
            validate=validate,
            timeout=timeout,
        )

    print("")
    print("===============================================================")
    print("")
    print("***All the Tests are Finished. DPBench is Done.***")
    print("")
    print("===============================================================")
    print("")


def all_benchmarks_passed_validation(dbfile):
    """Checks the results table of the output database to confirm if all
    benchmarks passed validation in the last run.
    Args:
        dbfile (str): Name of database with dpbench results
    """

    summary = (
        "SELECT "
        + "MAX(id),"
        + "benchmark,"
        + "framework,"
        + "version,"
        + "details,"
        + "IIF(validated == 1, 'PASS', 'FAIL' ) AS result "
        + "FROM results "
        + "GROUP BY benchmark, framework, version, details, result "
        + "ORDER BY benchmark, framework;"
    )

    failed_benchmark_summary = (
        "SELECT "
        + "MAX(id),"
        + "benchmark,"
        + "framework,"
        + "version,"
        + "details,"
        + "IIF(validated == 1, 'PASS', 'FAIL' ) AS result "
        + "FROM results "
        + "WHERE validated = 0 "
        + "GROUP BY benchmark, framework, version, details, result;"
    )

    conn = dpbi.create_connection(dbfile)
    cur = conn.cursor()

    data = cur.execute(summary)
    print("Summary")
    print("==============================================")
    for row in data:
        print(row)
    print("==============================================")

    data = cur.execute(failed_benchmark_summary)
    fails = [row for row in data]

    if fails:
        print("Number of failing validations: ", len(fails))
        print("==============================================")
        for fail in fails:
            print(fail)
        print("==============================================")
        return False
    else:
        print("All benchmarks were validated successfully")
        return True
