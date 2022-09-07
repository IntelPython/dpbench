# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: BSD-3-Clause
import pathlib
import sys
from typing import Any, Callable, Dict, Sequence, Tuple

from dpbench.infrastructure import Benchmark, Framework

_impl = {
    "kernel-mode": "k",
    "numpy-mode": "n",
    "prange-mode": "p",
}


class NumbaDpexFramework(Framework):
    """A class for reading and processing framework information."""

    def __init__(self, fname: str, device: str = None):
        """Reads framework information.
        :param fname: The framework name.
        """

        self.device = "default" if device is None else device
        super().__init__(fname)

    def impl_files(self, bench: Benchmark) -> Sequence[Tuple[str, str]]:
        """Returns the framework's implementation files for a particular
        benchmark.
        :param bench: A benchmark.
        :returns: A list of the benchmark implementation files.
        """

        parent_folder = pathlib.Path(__file__).parent.absolute()
        implementations = []
        for impl_name, impl_postfix in _impl.items():
            pymod_path = parent_folder.joinpath(
                "..",
                "..",
                "dpbench",
                "benchmarks",
                bench.info["relative_path"],
                bench.info["module_name"]
                + "_"
                + self.info["postfix"]
                + "_"
                + impl_postfix
                + ".py",
            )
            implementations.append((pymod_path, impl_name))
        return implementations

    def implementations(
        self, bench: Benchmark
    ) -> Sequence[Tuple[Callable, str]]:
        """Returns the framework's implementations for a particular benchmark.
        :param bench: A benchmark.
        :returns: A list of the benchmark implementations.
        """

        module_pypath = "dpbench.benchmarks.{r}.{m}".format(
            r=bench.info["relative_path"].replace("/", "."),
            m=bench.info["module_name"],
        )
        if "postfix" in self.info.keys():
            postfix = self.info["postfix"]
        else:
            postfix = self.fname
        module_str = "{m}_{p}".format(m=module_pypath, p=postfix)
        func_str = bench.info["func_name"]

        implementations = []
        for impl_name, impl_postfix in _impl.items():
            ldict = dict()
            try:
                exec(
                    "from {m}_{p} import {f} as impl".format(
                        m=module_str, p=impl_postfix, f=func_str
                    ),
                    ldict,
                )
                implementations.append((ldict["impl"], impl_name))
            except ImportError:
                continue
            except Exception:
                print(
                    "Failed to load the {r} {f} implementation.".format(
                        r=self.info["full_name"], f=impl_name
                    )
                )
                continue

        return implementations

    def imports(self) -> Dict[str, Any]:
        """Returns a dictionary any modules and methods needed for running
        a benchmark."""
        import dpctl

        # check if dpctl is loaded, if yes, why need return?
        modulename = "dpctl"
        if modulename not in sys.modules:
            print("You have not imported the {} module".format(modulename))
        return {"dpctl": dpctl}

    def exec_str(self, bench: Benchmark, impl: Callable = None):
        """Generates the execution-string that should be used to call
        the benchmark implementation.
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        """

        dpctl_ctx_str = (
            "with dpctl.device_context(dpctl.select_{d}_device()): ".format(
                d=self.device
            )
        )
        arg_str = self.arg_str(bench, impl)
        main_exec_str = "__dpb_result = __dpb_impl({a})".format(a=arg_str)
        return dpctl_ctx_str + main_exec_str
