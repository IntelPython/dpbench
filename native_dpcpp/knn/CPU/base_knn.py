# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os, utils
import numpy as np
import argparse
from dpbench_datagen.knn import gen_data_to_file

# make xrange available in python 3
try:
    xrange
except NameError:
    xrange = range


CLASSES_NUM = 3
TRAIN_DATA_SIZE = 2 ** 10


# create input data, call blackscholes computation function (alg)
def run(name, sizes=1, step=2, nopt=2 ** 10):
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=sizes, help="Number of steps")
    parser.add_argument("--step", type=int, default=step, help="Factor for each step")
    parser.add_argument("--size", type=int, default=nopt, help="Initial data size")
    parser.add_argument(
        "--repeat", type=int, default=1, help="Iterations inside measured region"
    )
    # parser.add_argument('--text', default='', help='Print with each result')
    parser.add_argument(
        "--test",
        required=False,
        action="store_true",
        help="Check for correctness by comparing output with naieve Python version",
    )
    parser.add_argument(
        "--json",
        required=False,
        default=__file__.replace("py", "json"),
        help="output json data filename",
    )

    args = parser.parse_args()
    nopt = args.size
    repeat = args.repeat

    clean_string = ["make", "clean"]
    utils.run_command(clean_string, verbose=True)

    build_string = ["make"]
    utils.run_command(build_string, verbose=True)

    if os.path.isfile("runtimes.csv"):
        os.remove("runtimes.csv")

    for _ in xrange(sizes):
        # generate input data
        gen_data_to_file(nopt)

        # run the C program
        run_cmd = ["./knn", str(nopt), str(repeat)]
        utils.run_command(run_cmd, verbose=True)
        nopt *= step
        repeat -= step
        if repeat < 1:
            repeat = 1


if __name__ == "__main__":
    run("Knn dpcpp")
