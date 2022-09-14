# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import pathlib
import subprocess
from typing import Any, Callable, Dict, Sequence, Tuple

from dpbench.infrastructure import Benchmark, Framework, utilities


class DpcppFramework(Framework):
    """A class for reading and processing framework information."""

    def __init__(self, fname: str, fconfig_path: str = None):
        """Reads framework information.
        :param fname: The framework name.
        """

        super().__init__(fname, fconfig_path)

    def copy_to_func(self) -> Callable:
        """Returns the copy-method that should be used
        for copying the benchmark arguments to device."""

        def _copy_to_func_impl(ref_array):
            import dpctl.tensor as dpt

            if ref_array.flags["C_CONTIGUOUS"]:
                order = "C"
            elif ref_array.flags["F_CONTIGUOUS"]:
                order = "F"
            else:
                order = "K"
            return dpt.asarray(
                obj=ref_array,
                dtype=ref_array.dtype,
                device=self.device_filter_string(),
                copy=None,
                usm_type=None,
                sycl_queue=None,
                order=order,
            )

        return _copy_to_func_impl

    def copy_from_func(self) -> Callable:
        """Returns the copy-method that should be used
        for copying results back to NumPy (host) from
        any array created by the framework possibly on
        a device memory domain."""

        import dpctl.tensor as dpt

        return dpt.asnumpy

    def validator(self) -> Callable:
        """ """
        from . import utilities

        def _validator(ref, test):
            import dpctl.tensor as dpt

            np_test = []
            for t in test:
                try:
                    np_test.append(dpt.asnumpy(t))
                except TypeError as e:
                    print(
                        "Failed to validate dpcpp results. Could not convert"
                        + " dpcpp output to numpy ndarray"
                    )
                    return False
            return utilities.validate(ref, np_test, framework=self.fname)

        return _validator

    def implementations(
        self, bench: Benchmark
    ) -> Sequence[Tuple[Callable, str]]:
        """Returns the framework's implementations for a particular benchmark.
        :param bench: A benchmark.
        :returns: A list of the benchmark implementations.
        """

        module_pypath = "dpbench.benchmarks.{r}.{m}".format(
            r=bench.info["relative_path"],
            m=bench.info["module_name"] + "_sycl_native_ext",
        ).replace(".", "/")

        func_str = bench.info["func_name"] + "_sycl"
        module_str = module_pypath.replace("/", ".")
        ldict = dict()
        try:
            exec(
                "from {m} import {f} as impl".format(m=module_str, f=func_str),
                ldict,
            )
        except Exception as e:
            print(
                "Failed to load the {r} {f} implementation.".format(
                    r=self.info["full_name"], f=func_str
                )
            )
            raise e

        return [(ldict["impl"], func_str)]

    def version(self) -> str:
        """Returns the framework version."""
        # hack the dpcpp version, need validate dpcpp available first
        return subprocess.check_output(
            "dpcpp --version | grep -Po '\(.*?\)' | grep '\.'", shell=True
        )
