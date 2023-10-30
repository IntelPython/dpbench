# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Namespace class for parsed arguments."""

import argparse
from typing import Union


class Namespace(argparse.Namespace):
    """Namespace class for parsed arguments."""

    benchmarks: set[str]
    implementations: list[str]
    all_implementations: bool
    preset: str
    sycl_device: str
    dpbench: bool
    npbench: bool
    polybench: bool
    rodinia: bool
    print_results: bool
    validate: bool
    run_id: Union[int, None]
    last_run: bool
    results_db: str
    save: bool
    repeat: int
    timeout: float
    precision: Union[str, None]
    program: str
    color: str
    comparisons: list[str]
    log_level: str
    skip_expected_failures: bool


class CommaSeparateStringAction(argparse.Action):
    """Action that reads comma separated string into set of strings.

    This action supposed to be used in argparse argument.
    """

    def __call__(self, _, namespace, values, __):
        """Split values into set of strings."""
        setattr(namespace, self.dest, set(values.split(",")))


class CommaSeparateStringListAction(argparse.Action):
    """Action that reads comma separated string into set of strings.

    This action supposed to be used in argparse argument.
    """

    def __call__(self, _, namespace, values, __):
        """Split values into list of strings."""
        setattr(namespace, self.dest, values.split(","))
