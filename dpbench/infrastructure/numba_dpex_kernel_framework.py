# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache 2.0

import logging
from typing import Any, Callable, Dict

import pkg_resources

from .numba_dpex_framework import NumbaDpexFramework


class NumbaDpexKernelFramework(NumbaDpexFramework):
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
                device=self.sycl_device,
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

    def execute(self, impl_fn: Callable, input_args: Dict):
        """Numba_dpex kernels support directly passing in
        dpctl.tensor.usm_ndarray and compute follows data. No need for a
        device_context.


        :param impl_fn: A benchmark implementation.
        :param input_args: Parameters to be passed to the kernel
        """
        return impl_fn(**input_args)

    def version(self) -> str:
        """Returns the numba-dpex version."""

        try:
            return pkg_resources.get_distribution("numba_dpex").version
        except pkg_resources.DistributionNotFound:
            logging.exception("No version information exists for framework")
            return "unknown"
