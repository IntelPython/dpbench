# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: BSD-3-Clause

import pathlib
import subprocess
from typing import Any, Callable, Dict, Sequence, Tuple
import os
from dpbench.infrastructure import Benchmark, Framework,utilities

class DpcppFramework(Framework):
    """A class for reading and processing framework information."""

    def __init__(self, fname: str, device: str = None):
        """Reads framework information.
        :param fname: The framework name.
        """

        self.device = "default" if device is None else device
        super().__init__(fname)

    def implementations(
        self, bench: Benchmark
    ) -> Sequence[Tuple[Callable, str]]:
        """Returns the framework's implementations for a particular benchmark.
        :param bench: A benchmark.
        :returns: A list of the benchmark implementations.
        """
        #compile code
        module_pypath = "dpbench.benchmarks.{r}.{m}".format(
            r=bench.info["relative_path"],
            m=bench.info["module_name"]+"_sycl_native_ext",
        ).replace(".", "/")
        utilities.chdir(module_pypath)   #maybe replace with subprocess
        os.system("bash ./build.sh")
        #import sycl_xxxx from sycl
        func_str="sycl_"+bench.info["func_name"]
        module_str=module_pypath.replace("/", ".")+"."+func_str
        try:
            exec(
                "from {m} import {f} as {f}".format(m=module_str, f=func_str)
            )
        except Exception as e:
            print(
                "Failed to load the {r} {f} implementation.".format(
                    r=self.info["full_name"], f=func_str
                )
            )
            raise e   
   
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

    def version(self) -> str:
        """Returns the framework version."""
        #hack the dpcpp version, need validate dpcpp available first 
        return subprocess.check_output("dpcpp --version | grep -Po '\(.*?\)' | grep '\.'", shell=True)
    
 