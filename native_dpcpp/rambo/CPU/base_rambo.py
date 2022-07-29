# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os

import numpy as np
from dpbench_python.rambo.rambo_python import rambo_python

import utils

# make xrange available in python 3
try:
    xrange
except NameError:
    xrange = range

# create input data, call blackscholes computation function (alg)
def run(name, sizes=5, step=2, nopt=2**20):
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

    clean_string = ["make", "clean"]
    utils.run_command(clean_string, verbose=True)

    if args.usm:
        build_string = ["make", "comp"]
        utils.run_command(build_string, verbose=True)
        exec_name = "./rambo_comp"
    else:
        build_string = ["make"]
        utils.run_command(build_string, verbose=True)
        exec_name = "./rambo"

    if args.test:
        e_p = rambo_python(nopt)

        # run dpcpp
        run_cmd = [exec_name, str(nopt), str(1), "-t"]
        utils.run_command(run_cmd, verbose=True)

        e_n = np.fromfile("output.bin", np.float64).reshape(nopt, 4, 4)

        if np.allclose(e_p, e_n):
            print("Test succeeded\n")
        else:
            print("Test failed\n", "Python: ", e_p, "\n Numba: ", e_n)
        return

    if os.path.isfile("runtimes.csv"):
        os.remove("runtimes.csv")

    for i in xrange(sizes):
        # run the C program
        run_cmd = [exec_name, str(nopt), str(repeat)]
        utils.run_command(run_cmd, verbose=True)
        nopt *= step
        repeat -= step
        if repeat < 1:
            repeat = 1


if __name__ == "__main__":
    run("Rambo dpcpp")
