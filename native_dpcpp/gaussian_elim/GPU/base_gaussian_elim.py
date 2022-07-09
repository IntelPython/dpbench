# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import os

import numpy as np
from dpbench_datagen.gaussian_elim import gen_data_to_file, gen_matrix, gen_vec

import utils


def run(name, sizes=1, step=2, nopt=2**2):
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

    if args.usm:
        build_string = ["make", "comp"]
        utils.run_command(build_string, verbose=True)
        exec_name = "./gaussian_comp"
    else:
        build_string = ["make"]
        utils.run_command(build_string, verbose=True)
        exec_name = "./gaussian"

    if args.test:
        reference_result = [5.02e-02, 5.00e-04, 5.00e-04, 5.02e-02]
        ref_size = 4

        # run dpcpp
        gen_data_to_file(ref_size, 1.0)
        # run the C program
        run_cmd = [exec_name, str(ref_size), str(1), "-t"]
        utils.run_command(run_cmd, verbose=True)

        # read output of dpcpp
        result = np.fromfile("result.bin", np.float32)

        if np.allclose(result, reference_result):
            print(
                "Test succeeded. Python result: ",
                reference_result,
                " DPC++ result: ",
                result,
                "\n",
            )
        else:
            print(
                "Test failed. Python result: ",
                reference_result,
                " DPC++ result: ",
                result,
                "\n",
            )
        return

    for _ in range(sizes):
        # generate input data
        # value = 1.0 for the vector of coefficients (b)
        gen_data_to_file(nopt, 1.0)

        # run the C program
        run_cmd = [exec_name, str(nopt), str(repeat)]
        utils.run_command(run_cmd, verbose=True)
        nopt *= step
        repeat -= step
        if repeat < 1:
            repeat = 1

    if os.path.isfile("./gaussian"):
        os.remove("./gaussian")

    if os.path.isfile("./gaussian_comp"):
        os.remove("./gaussian_comp")

    if os.path.isfile("m_data.bin"):
        os.remove("m_data.bin")

    if os.path.isfile("v_data.bin"):
        os.remove("m_data.bin")


if __name__ == "__main__":
    run("Gaussian elimination dpcpp")
