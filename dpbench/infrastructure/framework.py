# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import pathlib
from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np
import pkg_resources

from dpbench.infrastructure import Benchmark


class Framework(object):
    """A class for reading and processing framework information."""

    def __init__(self, fname: str):
        """Reads framework information.
        :param fname: The framework name.
        """

        self.fname = fname

        parent_folder = pathlib.Path(__file__).parent.absolute()
        frmwrk_filename = "{f}.json".format(f=fname)
        frmwrk_path = parent_folder.joinpath(
            "..", "configs", "framework_info", frmwrk_filename
        )
        try:
            with open(frmwrk_path) as json_file:
                self.info = json.load(json_file)["framework"]
                # print(self.info)
        except Exception as e:
            print(
                "Framework JSON file {f} could not be opened.".format(
                    f=frmwrk_filename
                )
            )
            raise (e)

    def version(self) -> str:
        """Returns the framework version."""
        return pkg_resources.get_distribution(self.fname).version

    def imports(self) -> Dict[str, Any]:
        """Returns a dictionary any modules and methods needed for running
        a benchmark."""
        return {}

    def copy_func(self) -> Callable:
        """Returns the copy-method that should be used
        for copying the benchmark arguments."""
        return np.copy

    def impl_files(self, bench: Benchmark) -> Sequence[Tuple[str, str]]:
        """Returns the framework's implementation files for a particular
        benchmark.
        :param bench: A benchmark.
        :returns: A list of the benchmark implementation files.
        """

        parent_folder = pathlib.Path(__file__).parent.absolute()
        pymod_path = parent_folder.joinpath(
            "..",
            "..",
            "dpbench",
            "benchmarks",
            bench.info["relative_path"],
            bench.info["module_name"] + "_" + self.info["postfix"] + ".py",
        )
        return [(pymod_path, "default")]

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

        return [(ldict["impl"], "default")]

    def args(self, bench: Benchmark, impl: Callable = None):
        """Generates the input arguments that should be used for calling
        the benchmark implementation.
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        """

        return [
            "__dpb_{pr}_{a}".format(pr=self.info["prefix"], a=a)
            if a in bench.info["array_args"]
            else a
            for a in bench.info["input_args"]
        ]

    def out_args(self, bench: Benchmark, impl: Callable = None):
        """Generates the input/output arguments that should be copied during
        the setup.
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        """

        return [
            "__dpb_{pr}_{a}".format(pr=self.info["prefix"], a=a)
            for a in bench.info["array_args"]
        ]

    def arg_str(self, bench: Benchmark, impl: Callable = None):
        """Generates the argument-string that should be used for calling
        the benchmark implementation.
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        """

        input_args = self.args(bench, impl)
        return ", ".join(input_args)

    def out_arg_str(self, bench: Benchmark, impl: Callable = None):
        """Generates the argument-string that should be used during the setup.
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        """

        output_args = self.out_args(bench, impl)
        return ", ".join(output_args)

    def setup_str(self, bench: Benchmark, impl: Callable = None):
        """Generates the setup-string that should be used before calling
        the benchmark implementation.
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        """

        if len(bench.info["array_args"]):
            arg_str = self.out_arg_str(bench, impl)
            copy_args = [
                "__dpb_copy({})".format(a) for a in bench.info["array_args"]
            ]
            return arg_str + " = " + ", ".join(copy_args)
        return "pass"

    def exec_str(self, bench: Benchmark, impl: Callable = None):
        """Generates the execution-string that should be used to call
        the benchmark implementation.
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        """

        arg_str = self.arg_str(bench, impl)
        return "__dpb_result = __dpb_impl({a})".format(a=arg_str)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.fname == other.fname

    def __hash__(self):
        return hash((self.fname))


def generate_framework(fname: str) -> Framework:
    """Generates a framework object with the correct class.
    :param fname: The framework name.
    """

    parent_folder = pathlib.Path(__file__).parent.absolute()
    frmwrk_filename = "{f}.json".format(f=fname)
    frmwrk_path = parent_folder.joinpath(
        "..", "..", "framework_info", frmwrk_filename
    )
    try:
        with open(frmwrk_path) as json_file:
            info = json.load(json_file)["framework"]
    except Exception as e:
        print(
            "Framework JSON file {f} could not be opened.".format(
                f=frmwrk_filename
            )
        )
        raise (e)

    exec("from dpbench.infrastructure import {}".format(info["class"]))
    frmwrk = eval("{}(fname)".format(info["class"]))
    return frmwrk
