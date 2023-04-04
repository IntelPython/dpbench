# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class ErrorCodes(Enum):
    """An enumeration of the various error codes returned by dpbench"""

    SUCCESS = 0
    UNIMPLEMENTED = -1
    NO_FRAMEWORK = -2
    FAILED_EXECUTION = -3
    FAILED_VALIDATION = -4
    EXECUTION_TIMEOUT = -5


class ValidationStatusCodes(Enum):
    """An enumeration of the error codes indicating success/failure of the
    benchmark validation step
    """

    SUCCESS = 0
    FAILURE = -1
