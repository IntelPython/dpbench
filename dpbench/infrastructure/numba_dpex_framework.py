# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache 2.0

from typing import Any, Callable, Dict

import dpctl

from dpbench.infrastructure import Framework

_impl = {
    "kernel-mode": "k",
    "numpy-mode": "n",
    "prange-mode": "p",
}


class NumbaDpexFramework(Framework):
    """A class for reading and processing framework information."""

    def __init__(self, fname: str, fconfig_path: str = None):
        """Reads framework information.
        :param fname: The framework name.
        """

        super().__init__(fname, fconfig_path)

    def imports(self) -> Dict[str, Any]:
        """Returns a dictionary any modules and methods needed for running
        a benchmark."""
        import dpctl

        return {"dpctl": dpctl}

    def execute(self, impl_fn: Callable, input_args: Dict):

        with dpctl.device_context(self.sycl_device):
            return impl_fn(**input_args)
