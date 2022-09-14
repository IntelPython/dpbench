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
    fconfig_path=None,
    bconfig_path=None,
    preset="S",
    repeat=1,
    validate=True,
    timeout=10.0,
):
    print("")
    print("================ Benchmark " + bname + " ========================")
    print("")

    try:
        benchmod = importlib.import_module("dpbench.benchmarks." + bname)
        bench = dpbi.Benchmark(benchmod, bconfig_path=bconfig_path)
    except Exception as e:
        print(e)
        warnings.warn(
            "WARN: Skipping the benchmark "
            + bname
            + ". Could not create a Benchmark object."
        )
        return

    bench_impls = bench.get_impl_fnlist()

    if not bench_impls:
        warnings.warn(
            "WARN: Skipping the benchmark "
            + bname
            + ". No implementations exist for the benchmark."
        )
        return

    fws = set()

    # Create the needed Frameworks by looking at the benchmark
    # implementations
    # FIXME: Get the framework name from the framework JSON in the config
    for bimpl in bench_impls:
        if "_numba" in bimpl[0] and "_dpex" not in bimpl[0]:
            fws.add(dpbi.NumbaFramework("numba"))
        elif "_numpy" in bimpl[0]:
            fws.add(dpbi.Framework("numpy"))
        elif "_python" in bimpl[0]:
            fws.add(dpbi.Framework("python"))
        elif "_dpex" in bimpl:
            fws.add(dpbi.NumbaDpexFramework("numba_dpex", fconfig_path))
        elif "_sycl" in bimpl:
            fws.add(dpbi.DpcppFramework("dpcpp", fconfig_path))
        elif "_dpnp" in bimpl:
            # FIXME: dpnp framework needs to be fixed before uncommenting this
            # step.
            # fws.append(dpbi.DpnpFramework("dpnp"))
            warnings.warn(
                "DPNP Framework is broken, skipping dpnp implementation"
            )
            pass

    # Check if a NumPy implementation of the benchmark is there. The
    # NumPy implementation is used for validations.
    fw_np = [fw for fw in fws if "numpy" in fw.fname or "python" in fw.fname]

    if not fw_np:
        warnings.warn(
            "WARN: Skipping running "
            + bname
            + ". Missing NumPy implementation for the benchmark."
        )
        return

    for fw in fws:
        if fw not in fw_np:
            test = dpbi.Test(bench=bench, frmwrk=fw, npfrmwrk=fw_np[0])
            try:
                test.run(
                    preset=preset,
                    repeat=repeat,
                    validate=validate,
                    timeout=timeout,
                )
            except Exception as e:
                warnings.warn(
                    "ERROR: Failed to test the "
                    + fw.fname
                    + " implementation for "
                    + bname
                    + ". The error is {}".format(e)
                )


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
