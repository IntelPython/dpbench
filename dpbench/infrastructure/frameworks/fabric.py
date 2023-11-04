# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import logging

import dpbench.config as cfg

from .cupy_framework import CupyFramework
from .dpcpp_framework import DpcppFramework
from .dpnp_framework import DpnpFramework
from .framework import Framework
from .numba_dpex_framework import NumbaDpexFramework
from .numba_framework import NumbaFramework
from .numba_mlir_framework import NumbaMlirFramework


# TODO: do initialization only once for all benchmarks
def build_framework_map() -> dict[str, Framework]:
    """Create a dictionary mapping each implementation postfix to a
    corresponding Framework object.

    Args:
        impl_fnlist : list of implementation functions

    Returns:
        Dict: Dictionary mapping implementation function to a Framework
    """

    result = dict()

    for framework_config in cfg.GLOBAL.frameworks:
        framework = build_framework(framework_config)

        for postfix in framework_config.postfixes:
            result[postfix.postfix] = framework

    return result


def build_framework(framework_config: cfg.Framework) -> Framework:
    available_classes = [
        Framework,
        DpcppFramework,
        DpnpFramework,
        CupyFramework,
        NumbaFramework,
        NumbaDpexFramework,
        NumbaMlirFramework,
    ]

    available_classes = {_cls.__name__: _cls for _cls in available_classes}

    constructor = available_classes.get(framework_config.class_, None)

    if constructor is None:
        logging.warn(
            f"Could not find class for {framework_config.simple_name}, using default one."
        )
        constructor = Framework

    return constructor(config=framework_config)
