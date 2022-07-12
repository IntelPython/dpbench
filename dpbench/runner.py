# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache 2.0 License

import warnings

import pkg_resources

import dpbench.infrastructure as dpbi


def list_available_benchmarks():
    """Return the list of available benchmarks"""

    return pkg_resources.resource_listdir(__name__, "benchmarks")


def run_benchmarks(
    bconfig_path=None, preset="S", repeat=1, validate=True, timeout=10.0
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
    print("***Start Running DPBench***");
    
    for b in list_available_benchmarks():
        print("")
        print("===============================================================")
        print("")
        bdir = "benchmarks/" + b
        if not pkg_resources.resource_isdir(__name__, bdir):
            continue
        bench = None
        try:
            bench = dpbi.Benchmark(bname=b, bconfig_path=bconfig_path)
        except Exception:
            warnings.warn(
                "WARN: Skipping the benchmark "
                + b
                + ". No configuration could not be found."
            )
            continue

        bench_impls = pkg_resources.resource_listdir(__name__, bdir)
        if not bench_impls:
            warnings.warn(
                "WARN: Skipping the benchmark "
                + b
                + ". No implementations exist for the benchmark."
            )
            continue

        fws = set()
        # Create the needed Frameworks by looking at the benchmark
        # implementations
        for bimpl in bench_impls:
            if "_numba" in bimpl and "_dpex" not in bimpl:
                # create a Numba framework
                fws.add(dpbi.NumbaFramework("numba"))
            elif "_numpy" in bimpl:
                fws.add(dpbi.Framework("numpy"))
            elif "_dpex" in bimpl:
                fws.add(dpbi.NumbaDpexFramework("numba_dpex"))
                pass
            elif "_dpnp" in bimpl:
                # fws.append(dpbi.DpnpFramework("dpnp"))
                pass

        # Check if a NumPy implementation of the benchmark is there. The
        # NumPy implementation is used for validations.
        fw_np = [fw for fw in fws if "numpy" in fw.fname]
        if not fw_np:
            warnings.warn(
                "WARN: Skipping running "
                + b
                + ". Missing NumPy implementation for "
                + "the benchmark."
            )
            continue

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
                except Exception:
                    warnings.warn(
                        "ERROR: Failed to test the "
                        + fw.fname
                        + " implementation for "
                        + b
                        + "."
                    )
    print("")
    print("===============================================================")
    print("")
    print("***All the Tests are Finished. DPBench is Done.***");
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
