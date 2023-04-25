# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Butch convert json to toml.

This is the script file that intends to be used as bulk converter of json files
into toml files.
"""

import argparse
import json
import os

import tomli_w

dirname: str = os.path.dirname(__file__)

parser = argparse.ArgumentParser(description="Covert json to toml.")

parser.add_argument(
    "dirname", default=os.path.join(dirname, "../dpbench/configs")
)
parser.add_argument("-r", "--remove", default=False)

args = parser.parse_args()

for bench_info_file in os.listdir(args.dirname):
    bench_info_file = os.path.join(args.dirname, bench_info_file)
    if bench_info_file[-5:] != ".json":
        continue

    with open(bench_info_file) as file:
        file_contents = file.read()

    bench_info = json.loads(file_contents)
    bench_info_toml = tomli_w.dumps(bench_info)
    bench_info_file_toml = bench_info_file[:-5] + ".toml"

    with open(bench_info_file_toml, "w") as file:
        file.write(bench_info_toml)

    if args.remove:
        os.remove(bench_info_file)
        if os.path.exists(bench_info_file + ".license"):
            os.remove(bench_info_file + ".license")
