# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache 2.0


from typing import Any, Callable, Dict

import pkg_resources

from .framework import Framework


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
