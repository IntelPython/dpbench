# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: BSD-3-Clause

import pathlib
from typing import Any, Callable, Dict, Sequence, Tuple

import pkg_resources

from dpbench.infrastructure import Benchmark, Framework


class DpnpFramework(Framework):
    """A class for reading and processing framework information."""

    def __init__(self, fname: str, fconfig_path: str = None):
        """Reads framework information.
        :param fname: The framework name.
        """

        super().__init__(fname, fconfig_path)

    def version(self) -> str:
        """Returns the framework version."""
        return pkg_resources.get_distribution(self.fname).version

    def imports(self) -> Dict[str, Any]:
        """Returns a dictionary any modules and methods needed for running
        a benchmark."""
        import dpctl

        return {"dpctl": dpctl}

    def copy_to_func(self) -> Callable:
        """Returns the copy-method that should be used
        for copying the benchmark arguments."""
        import dpnp

        return dpnp.copy

    def copy_from_func(self) -> Callable:
        """Returns the copy-method that should be used
        for copying the benchmark arguments."""
        import dpnp

        return dpnp.copy

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
        main_exec_str = "__dpb_result = __dpb_impl[4000000,]({a})".format(
            a=arg_str
        )
        return dpctl_ctx_str + main_exec_str
