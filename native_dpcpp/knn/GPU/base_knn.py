# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import run_utils as utils
import numpy as np
import argparse
from dpbench_datagen.knn import (
    gen_data_to_file,
    gen_train_data,
    gen_test_data,
    N_NEIGHBORS,
    CLASSES_NUM,
    TRAIN_DATA_SIZE,
    DATA_DIM,
)
from dpbench_python.knn.knn_python import knn_python

# make xrange available in python 3
try:
    xrange
except NameError:
    xrange = range

# create input data, call blackscholes computation function (alg)
def run(name, sizes=5, step=2, nopt=2 ** 20):
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=sizes, help="Number of steps")
    parser.add_argument("--step", type=int, default=step, help="Factor for each step")
    parser.add_argument("--size", type=int, default=nopt, help="Initial data size")
    parser.add_argument(
        "--repeat", type=int, default=1, help="Iterations inside measured region"
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
    parser.add_argument(
        "--json",
        required=False,
        default=__file__.replace("py", "json"),
        help="output json data filename",
    )

    args = parser.parse_args()
    nopt = args.size
    sizes = args.steps
    repeat = args.repeat

    clean_string = ["make", "clean"]
    utils.run_command(clean_string, verbose=True)

    if args.usm:
        build_string = ["make", "comp"]
        utils.run_command(build_string, verbose=True)
        exec_name = "./knn_comp"
    else:
        build_string = ["make"]
        utils.run_command(build_string, verbose=True)
        exec_name = "./knn"

    if args.test:
        x_train, y_train = gen_train_data()
        x_test = gen_test_data(nopt)
        p_predictions = np.empty(nopt)
        # p_queue_neighbors_lst = np.empty((nopt, N_NEIGHBORS, 2))
        p_votes_to_classes_lst = np.zeros((nopt, CLASSES_NUM))

        knn_python(
            x_train,
            y_train,
            x_test,
            N_NEIGHBORS,
            CLASSES_NUM,
            TRAIN_DATA_SIZE,
            nopt,
            p_predictions,
            # p_queue_neighbors_lst,
            p_votes_to_classes_lst,
            DATA_DIM,
        )

        # run dpcpp
        gen_data_to_file(nopt)
        # run the C program
        run_cmd = [exec_name, str(nopt), str(1), "-t"]
        utils.run_command(run_cmd, verbose=True)

        # read output of dpcpp
        n_predictions = np.fromfile("predictions.bin", np.int64)

        if np.allclose(n_predictions, p_predictions):
            print(
                "Test succeeded. Python predictions: ",
                p_predictions,
                " DPC++ predictions: ",
                n_predictions,
                "\n",
            )
        else:
            print(
                "Test failed. Python predictions: ",
                p_predictions,
                " DPC++ predictions: ",
                n_predictions,
                "\n",
            )
        return

    if os.path.isfile("runtimes.csv"):
        os.remove("runtimes.csv")

    for _ in xrange(sizes):
        # generate input data
        gen_data_to_file(nopt)

        # run the C program
        run_cmd = [exec_name, str(nopt), str(repeat)]
        utils.run_command(run_cmd, verbose=True)
        nopt *= step
        repeat -= step
        if repeat < 1:
            repeat = 1


if __name__ == "__main__":
    run("Knn dpcpp")
