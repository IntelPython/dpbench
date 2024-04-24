# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Callable

import dpctl

import dpbench.config as cfg

from .framework import Framework


class DpcppFramework(Framework):
    """A class for reading and processing framework information."""

    def __init__(self, fname: str = None, config: cfg.Framework = None):
        """Reads framework information.
        :param fname: The framework name.
        """

        super().__init__(fname, config)

        try:
            self.sycl_device = self.info.sycl_device
            self.device_info = dpctl.SyclDevice(self.sycl_device).name
        except KeyError:
            pass
        except dpctl.SyclDeviceCreationError as sdce:
            logging.exception(
                "Could not create a Sycl device using filter {} string".format(
                    self.info.sycl_device
                )
            )
            raise sdce

    def device_filter_string(self) -> str:
        """Returns the sycl device's filter string if the framework has an
        associated sycl device."""

        try:
            return dpctl.SyclDevice(self.device).get_filter_string()
        except Exception:
            logging.exception("No device string exists for device")
            return "unknown"

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
                ref_array,
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

        def _copy_from_func(usm_array):
            import dpctl.tensor as dpt

            if isinstance(usm_array, dpt.usm_ndarray):
                return dpt.asnumpy(usm_array)
            else:
                return usm_array

        return _copy_from_func

    def version(self) -> str:
        """Returns the framework version."""
        # hack the dpcpp version, need validate dpcpp available first
        return self.info.dpcpp_version
