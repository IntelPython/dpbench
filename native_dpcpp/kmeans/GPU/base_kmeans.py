# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import os

import numpy as np
import utils as utils
from dpbench_datagen.kmeans import gen_data_to_file, gen_rand_data
from dpbench_python.kmeans.kmeans_python import kmeans_python

# make xrange available in python 3
try:
    xrange
except NameError:
    xrange = range

NUMBER_OF_CENTROIDS = 10

# create input data, call blackscholes computation function (alg)
def run(name, sizes=5, step=2, nopt=2**17):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps", type=int, default=sizes, help="Number of steps"
    )
    parser.add_argument(
        "--step", type=int, default=step, help="Factor for each step"
    )
    parser.add_argument(
        "--size", type=int, default=nopt, help="Initial data size"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Iterations inside measured region",
    )
    parser.add_argument(
        "--test",
        required=False,
        action="store_true",
        help="Check for correctness by comparing output with naieve Python version",
    )
    parser.add_argument(
        "--usm",
        required=False,
        action="store_true",
        help="Use USM Shared or data transfer",
    )

    args = parser.parse_args()
    nopt = args.size
    repeat = args.repeat
    sizes = args.steps

    clean_string = ["make", "clean"]
    utils.run_command(clean_string, verbose=True)

    if args.usm:
        build_string = ["make", "comp"]
        utils.run_command(build_string, verbose=True)
        exec_name = "./kmeans_comp"
    else:
        build_string = ["make"]
        utils.run_command(build_string, verbose=True)
        exec_name = "./kmeans"

    if args.test:
        (
            X,
            arrayPclusters_p,
            arrayC_p,
            arrayCsum_p,
            arrayCnumpoint_p,
        ) = gen_rand_data(nopt, dtype=np.float32)
        kmeans_python(
            X,
            arrayPclusters_p,
            arrayC_p,
            arrayCsum_p,
            arrayCnumpoint_p,
            nopt,
            NUMBER_OF_CENTROIDS,
        )

        # run dpcpp
        gen_data_to_file(nopt, dtype=np.float32)
        # run the C program
        run_cmd = [exec_name, str(nopt), str(1), "-t"]
        utils.run_command(run_cmd, verbose=True)

        # read output of dpcpp
        arrayC_n = np.fromfile("arrayC.bin", np.float32).reshape(
            NUMBER_OF_CENTROIDS, 2
        )
        arrayCsum_n = np.fromfile("arrayCsum.bin", np.float32).reshape(
            NUMBER_OF_CENTROIDS, 2
        )
        arrayCnumpoint_n = np.fromfile("arrayCnumpoint.bin", np.int32)

        if (
            np.allclose(arrayC_n, arrayC_p)
            and np.allclose(arrayCsum_n, arrayCsum_p)
            and np.allclose(arrayCnumpoint_n, arrayCnumpoint_p)
        ):
            print("Test succeeded\n")
        else:
            print(
                "Test failed\n",
                "arrayC_Python:",
                arrayC_p,
                "\n arrayC_numba:",
                arrayC_n,
                "arrayCsum_python:",
                arrayCsum_p,
                "\n arracyCsum_numba:",
                arrayCsum_n,
                "arrayCnumpoint_python:",
                arrayCnumpoint_p,
                "\n arrayCnumpoint_numba:",
                arrayCnumpoint_n,
            )
        return

    if os.path.isfile("runtimes.csv"):
        os.remove("runtimes.csv")

    for _ in xrange(sizes):
        # generate input data
        gen_data_to_file(nopt, dtype=np.float32)

        # run the C program
        run_cmd = [exec_name, str(nopt), str(repeat)]
        utils.run_command(run_cmd, verbose=True)
        nopt *= step
        repeat -= step
        if repeat < 1:
            repeat = 1


if __name__ == "__main__":
    run("Kmeans dpcpp")
