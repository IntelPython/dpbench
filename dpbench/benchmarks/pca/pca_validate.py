# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
from dpbench.infrastructure.benchmark_validation import (
    validate as default_validate,
)


def validate(expected: dict[str, any], actual: dict[str, any], rel_error=1e-05):
    # TODO implement actual validation suitable for pca workload
    return default_validate(expected, actual, rel_error)
