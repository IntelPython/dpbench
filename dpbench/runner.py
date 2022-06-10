# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache 2.0 License

import pkg_resources
import dpbench.infrastructure as dpbi
import warnings


def list_available_benchmarks():
    """Return the list of available benchmarks"""

    return pkg_resources.resource_listdir(__name__, "benchmarks")


def run_benchmarks(
    bconfig_path=None, preset="S", repeat=1, validate=True, timeout=10.0
):

    for b in list_available_benchmarks():
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

        fws = []
        # Create the needed Frameworks by looking at the benchmark
        # implementations
        for bimpl in bench_impls:
            if "_dppy" in bimpl:
                fws.append(dpbi.NumbaDppyFramework("numba_dppy"))
            elif "_numba" in bimpl:
                # create a Numba framework
                fws.append(dpbi.NumbaFramework("numba"))
            elif "_numpy" in bimpl:
                fws.append(dpbi.Framework("numpy"))
            elif "_dpnp" in bimpl:
                fws.append(dpbi.DpnpFramework("dpnp"))
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
