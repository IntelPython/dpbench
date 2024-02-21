# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Callable, Dict

import numpy as np

import dpbench.config as cfg

from .framework import Framework


class NumbaMlirFramework(Framework):
    """A class for reading and processing framework information."""

    def __init__(self, fname: str = None, config: cfg.Framework = None):
        """Reads framework information.
        :param fname: The framework name.
        """

        super().__init__(fname, config)

        self.sycl_device = self.info.sycl_device
        if self.sycl_device:
            import dpctl

            self.device_info = dpctl.SyclDevice(self.sycl_device).name

    def copy_to_func(self) -> Callable:
        """Returns the copy-method that should be used
        for copying the benchmark arguments to device."""

        if self.sycl_device:
            import dpctl.tensor as dpt

            def _copy_to_func_impl(ref_array):
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
        else:
            return np.copy

    def copy_from_func(self) -> Callable:
        """Returns the copy-method that should be used
        for copying results back to NumPy (host) from
        any array created by the framework possibly on
        a device memory domain."""

        if self.sycl_device:
            import dpctl.tensor as dpt

            def cpy(val):
                if isinstance(val, dpt.usm_ndarray):
                    return dpt.asnumpy(val)
                else:
                    return np.asarray(val)

            return cpy
        else:
            return np.copy
