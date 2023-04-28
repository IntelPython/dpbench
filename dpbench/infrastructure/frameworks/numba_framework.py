# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: BSD-3-Clause

import dpbench.config as cfg

from .framework import Framework

_impl = {
    "object-mode": "o",
    "object-mode-parallel": "op",
    "object-mode-parallel-range": "opr",
    "nopython-mode": "n",
    "nopython-mode-parallel": "np",
    "nopython-mode-parallel-range": "npr",
}


class NumbaFramework(Framework):
    """A class for reading and processing framework information."""

    def __init__(self, fname: str = None, config: cfg.Framework = None):
        """Reads framework information.
        :param fname: The framework name.
        """

        super().__init__(fname, config)
