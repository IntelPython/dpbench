# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Set expected failure to benchmark configuration based on results.

This script requires tomlkit library installed.
"""

import argparse
import os

import tomlkit

from dpbench.infrastructure.reporter import get_failures_from_results

dirname: str = os.path.dirname(__file__)

parser = argparse.ArgumentParser(
    description="Set expected failure for benchmarks based on the run.",
)

parser.add_argument(
    "dirname", default=os.path.join(dirname, "../dpbench/configs/bench_info")
)

parser.add_argument(
    "-r",
    "--run_id",
    default=1,
)

parser.add_argument(
    "-n",
    "--no-remove",
    action=argparse.BooleanOptionalAction,
    default=False,
)

args = parser.parse_args()

failures_list = get_failures_from_results(run_id=args.run_id)
failures = dict()
for benchmark, implementation in failures_list:
    if not failures.get(benchmark):
        failures[benchmark] = []
    failures[benchmark] += [implementation]

for benchmark, implementation in failures_list:
    failures[benchmark].sort()

for root, _, files in os.walk(args.dirname):
    for file in files:
        bench_info_file = os.path.join(root, file)
        if bench_info_file[-5:] != ".toml":
            continue

        with open(bench_info_file) as file:
            file_contents = file.read()

        bench_info = tomlkit.loads(file_contents)
        bench = bench_info["benchmark"]["module_name"]

        if (
            not args.no_remove
            and failures.get(bench, None) is None
            and bench_info["benchmark"].get(
                "expected_failure_implementations", None
            )
            is not None
        ):
            bench_info["benchmark"].pop("expected_failure_implementations")

        if failures.get(bench, None) is not None:
            bench_info["benchmark"][
                "expected_failure_implementations"
            ] = failures[bench]

        file_contents = tomlkit.dumps(bench_info)

        with open(bench_info_file, "w") as file:
            file.write(file_contents)
