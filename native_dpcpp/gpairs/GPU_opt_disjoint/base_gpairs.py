# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import numpy as np
import numpy.random as rnd
import dpctl, dpctl.tensor as dpt
import run_utils as utils
from dpbench_python.gpairs.gpairs_python import gpairs_python
from dpbench_datagen.gpairs import gen_rand_data, gen_data_to_file

try:
    import itimer as it

    now = it.itime
    get_mops = it.itime_mops_now
except:
    from timeit import default_timer

    now = default_timer
    get_mops = lambda t0, t1, n: (n / (t1 - t0), t1 - t0)

######################################################
# GLOBAL DECLARATIONS THAT WILL BE USED IN ALL FILES #
######################################################

# make xrange available in python 3
try:
    xrange
except NameError:
    xrange = range

###############################################


def gen_data_np(npoints, dtype=np.float32):
    x1, y1, z1, w1, x2, y2, z2, w2, DEFAULT_RBINS_SQUARED = gen_rand_data(
        npoints, dtype
    )
    result = np.zeros_like(DEFAULT_RBINS_SQUARED)[:-1].astype(dtype)
    return (x1, y1, z1, w1, x2, y2, z2, w2, DEFAULT_RBINS_SQUARED, result)


##############################################


def run(name, sizes=5, step=2, nopt=2**16):
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
        exec_name = "./gpairs_comp"
    else:
        build_string = ["make"]
        utils.run_command(build_string, verbose=True)
        exec_name = "./gpairs"

    if args.test:
        (
            x1,
            y1,
            z1,
            w1,
            x2,
            y2,
            z2,
            w2,
            DEFAULT_RBINS_SQUARED,
            result_p,
        ) = gen_data_np(nopt)
        gpairs_python(
            x1, y1, z1, w1, x2, y2, z2, w2, DEFAULT_RBINS_SQUARED, result_p
        )

        # run dpcpp
        gen_data_to_file(nopt, np.float32)
        run_cmd = [exec_name, str(nopt), str(1), "-t"]
        utils.run_command(run_cmd, verbose=True)

        # read output of dpcpp into result_p
        result_n = np.fromfile("result.bin", np.float32)

        # compare outputs
        if np.allclose(result_p, result_n):
            print(
                "Test succeeded. Python result: ",
                result_p,
                "\nDPC++ result: ",
                result_n,
                "\n",
            )
        else:
            print(
                "Test failed. Python result: ",
                result_p,
                "\nDPC++ result: ",
                result_n,
                "\n",
            )
        return

    if os.path.isfile("runtimes.csv"):
        os.remove("runtimes.csv")

    for i in xrange(sizes):
        # generate input data
        gen_data_to_file(nopt, np.float32)

        # run the C program
        run_cmd = [exec_name, str(nopt), str(repeat)]
        utils.run_command(run_cmd, verbose=True)
        nopt *= step
        repeat -= step
        if repeat < 1:
            repeat = 1


if __name__ == "__main__":
    run("Gpairs dpcpp")
