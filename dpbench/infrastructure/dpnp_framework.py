# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Callable

import dpctl

from .framework import Framework


class DpnpFramework(Framework):
    """A class for reading and processing framework information."""

    def __init__(self, fname: str, fconfig_path: str = None):
        """Reads framework information.
        :param fname: The framework name.
        """

        super().__init__(fname, fconfig_path)

        try:
            self.sycl_device = self.info["sycl_device"]
            dpctl.SyclDevice(self.sycl_device)
        except KeyError:
            pass
        except dpctl.SyclDeviceCreationError as sdce:
            logging.exception(
                "Could not create a Sycl device using filter {} string".format(
                    self.info["sycl_device"]
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
        for copying the benchmark arguments."""

        def _copy_to_func_impl(ref_array):
            import dpnp

            if ref_array.flags["C_CONTIGUOUS"]:
                order = "C"
            elif ref_array.flags["F_CONTIGUOUS"]:
                order = "F"
            else:
                order = "K"
            return dpnp.asarray(
                ref_array,
                dtype=ref_array.dtype,
                order=order,
                like=None,
                device=self.sycl_device,
                usm_type=None,
                sycl_queue=None,
            )

        return _copy_to_func_impl

    def copy_from_func(self) -> Callable:
        """Returns the copy-method that should be used
        for copying the benchmark arguments."""
        import dpnp

        return dpnp.asnumpy
