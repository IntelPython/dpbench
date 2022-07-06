# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: BSD-3-Clause

import pathlib
from typing import Callable, Sequence, Tuple

from dpbench.infrastructure import Benchmark, Framework

_impl = {
    "object-mode": "o",
    "object-mode-parallel": "op",
    "object-mode-parallel-range": "opr",
    "nopython-mode": "n",
    "nopython-mode-parallel": "np",
    "nopython-mode-parallel-range": "npr",
}


class NumbaFramework(Framework):
    """A class for reading and processing framework information."""

    def __init__(self, fname: str):
        """Reads framework information.
        :param fname: The framework name.
        """

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

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.fname == other.fname

    def __hash__(self):
        return hash((self.fname))
