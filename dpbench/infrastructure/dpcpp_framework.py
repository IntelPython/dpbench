# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import subprocess
from typing import Callable

from .framework import Framework


class DpcppFramework(Framework):
    """A class for reading and processing framework information."""

    def __init__(self, fname: str, fconfig_path: str = None):
        """Reads framework information.
        :param fname: The framework name.
        """

        super().__init__(fname, fconfig_path)

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

    def version(self) -> str:
        """Returns the framework version."""
        # hack the dpcpp version, need validate dpcpp available first
        import json
        import pathlib

        parent_folder = pathlib.Path(__file__).parent.absolute()
        version_file = parent_folder.joinpath(
            "..", "configs", "framework_info", "dpcpp_version.json"
        )
        with open(version_file) as json_file:
            version = json.load(json_file)["dpcpp_version"]
        return version
