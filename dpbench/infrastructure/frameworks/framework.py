# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: BSD-3-Clause

import logging
from importlib.util import find_spec
from typing import Any, Callable, Dict, final

import pkg_resources

import dpbench.config as cfg


class Framework(object):
    """A class for reading and processing framework information."""

    def __init__(
        self,
        fname: str = None,
        config: cfg.Framework = None,
    ):
        """Reads framework information.
        :param fname: The framework name. It must be provided if no config was
        provided.
        :param config: The framework configuration. It must be provided if no
        fname was provided.
        """

        if fname is None and config is None:
            raise ValueError("At least one of fname or config must be provided")

        if config is None:
            config = [
                f for f in cfg.GLOBAL.frameworks if fname in f.simple_name
            ]
            if len(config) < 1:
                raise ValueError(f"Configuration with name {fname} not found")
            config = config[0]

        self.info = config
        self.fname = self.info.simple_name

        import cpuinfo

        self.device_info = cpuinfo.get_cpu_info().get("brand_raw")

    @staticmethod
    def required_packages() -> list[str]:
        return []

    @classmethod
    def get_missing_required_packages(cls) -> None:
        unavailable_packages = []
        for pkg in cls.required_packages():
            spec = find_spec(pkg)
            if spec is None:
                unavailable_packages.append(pkg)

        return unavailable_packages

    def device_filter_string(self) -> str:
        """Returns the sycl device's filter string if the framework has an
        associated sycl device."""

        logging.exception("No device string exists for device")
        return "unknown"

    def version(self) -> str:
        """Returns the framework version."""
        if self.fname == "python":
            import platform

            return platform.python_version()

        pkg_name = self.fname

        if self.fname == "numba_cuda":
            pkg_name = "numba"

        try:
            return pkg_resources.get_distribution(pkg_name).version
        except pkg_resources.DistributionNotFound:
            logging.exception("No version information exists for framework")
            return "unknown"

    def copy_to_func(self) -> Callable:
        """Returns the copy-method that should be used
        for copying the benchmark arguments from host
        to device."""
        import numpy

        return numpy.copy

    def copy_from_func(self) -> Callable:
        """Returns the copy-method that should be used
        for copying the benchmark arguments from device
        to host."""
        import numpy

        return numpy.copy

    def execute(self, impl_fn: Callable, input_args: Dict):
        """A wrapper for a framework to customize how a benchmark
        implementation should be executed.

        :param impl: A benchmark implementation.
        """
        return impl_fn(**input_args)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.fname == other.fname

    def __hash__(self):
        return hash((self.fname))
