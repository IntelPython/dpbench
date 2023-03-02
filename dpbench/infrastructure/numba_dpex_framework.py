# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict

import dpctl

from .framework import Framework


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

        return {"dpctl": dpctl}

    def execute(self, impl_fn: Callable, input_args: Dict):
        """The njit implementations for numba_dpex require calling the
        functions inside a dpctl.device_context contextmanager to trigger
        offload.
        """
        with dpctl.device_context(self.sycl_device):
            return impl_fn(**input_args)
