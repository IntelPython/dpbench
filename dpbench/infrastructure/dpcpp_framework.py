# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: BSD-3-Clause

import pathlib
import os
from typing import Any, Callable, Dict, Sequence, Tuple

from dpbench.infrastructure import Benchmark, Framework


class DpcppFramework(Framework):
    """A class for reading and processing framework information."""

    def __init__(self, fname: str, device: str = None):
        """Reads framework information.
        :param fname: The framework name.
        """

        self.device = "default" if device is None else device
        super().__init__(fname)

    def imports(self) -> Dict[str, Any]:
        """Returns a dictionary any modules and methods needed for running
        a benchmark."""
        import dpctl
        print("******import dpcpp lib**********")
        #possible import dpcpp lib
        return {"dpctl": dpctl}

    def version(self) -> str:
        """Returns the framework version."""
        #hack the version 
        return os.system("dpcpp --version | grep -Po '\(.*?\)' | grep '\.'");