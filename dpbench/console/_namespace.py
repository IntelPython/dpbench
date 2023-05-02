# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Namespace class for parsed arguments."""

import argparse
from typing import Union


class Namespace(argparse.Namespace):
    """Namespace class for parsed arguments."""

    benchmarks: set[str]
    implementations: set[str]
    all_implementations: bool
    preset: str
    dpbench: bool
    npbench: bool
    polybench: bool
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
