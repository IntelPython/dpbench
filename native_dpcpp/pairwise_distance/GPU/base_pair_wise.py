# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import sys

import dpctl
import dpctl.tensor as dpt
import numpy as np
from dpbench_datagen.pairwise_distance import gen_data_to_file, gen_rand_data
from dpbench_python.pairwise_distance.pairwise_distance_python import (
    pairwise_distance_python,
)

import utils

######################################################
# GLOBAL DECLARATIONS THAT WILL BE USED IN ALL FILES #
######################################################

# make xrange available in python 3
try:
    xrange
except NameError:
    xrange = range


def gen_data(nopt, dims):
    X, Y = gen_rand_data(nopt, dims)
    return (X, Y, np.empty((nopt, nopt)))


def run(name, sizes=6, step=2, nopt=2**10):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps", required=False, default=sizes, help="Number of steps"
    )
    parser.add_argument(
        "--step", required=False, default=step, help="Factor for each step"
    )
    parser.add_argument(
        "--size", required=False, default=nopt, help="Initial data size"
    )
    parser.add_argument(
        "--repeat",
        required=False,
        default=1,
        help="Iterations inside measured region",
    )
    parser.add_argument(
        "--text", required=False, default="", help="Print with each result"
    )
    parser.add_argument("-d", type=int, default=3, help="Dimensions")
    parser.add_argument(
        "--usm",
        required=False,
        action="store_true",
        help="Use USM Shared or pure numpy",
    )
    parser.add_argument(
        "--test",
        required=False,
        action="store_true",
        help="Check for correctness by comparing output with naieve Python version",
    )

    args = parser.parse_args()
    sizes = int(args.steps)
    step = int(args.step)
    nopt = int(args.size)
    repeat = int(args.repeat)
    dims = int(args.d)

    clean_string = ["make", "clean"]
    utils.run_command(clean_string, verbose=True)

    if args.usm:
        build_string = ["make", "comp"]
        utils.run_command(build_string, verbose=True)
        exec_name = "./pairwise_distance_comp"
    else:
        build_string = ["make"]
        utils.run_command(build_string, verbose=True)
        exec_name = "./pairwise_distance"

    if args.test:
        X, Y, p_D = gen_data(nopt, dims)
        pairwise_distance_python(X, Y, p_D)

        # run dpcpp
        gen_data_to_file(nopt, dims)
        # run the C program
        run_cmd = [exec_name, str(nopt), str(1), "-t"]
        utils.run_command(run_cmd, verbose=True)

        # read output of dpcpp
        n_D = np.fromfile("D.bin").reshape(nopt, nopt)

        if np.allclose(n_D, p_D):
            print("Test succeeded\n")
        else:
            print("Test failed\n")
        return

    if os.path.isfile("runtimes.csv"):
        os.remove("runtimes.csv")

    for i in xrange(sizes):
        gen_data_to_file(nopt, dims)

        # run the C program
        run_cmd = [exec_name, str(nopt), str(repeat)]
        utils.run_command(run_cmd, verbose=True)
        nopt *= step
        repeat -= step
        if repeat < 1:
            repeat = 1


if __name__ == "__main__":
    run("Pairwise distance dpcpp")
