# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np
import sys, json, os
import run_utils as utils

# import numpy.random_intel as rnd

from dpbench_python.l2_distance.l2_distance_python import l2_distance_python
from dpbench_datagen.l2_distance import gen_data, gen_data_to_file

try:
    xrange
except NameError:
    xrange = range


def run(name, alg, sizes=10, step=2, nopt=2 ** 16):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', required=False, default=sizes, help="Number of steps")
    parser.add_argument('--step', required=False, default=step, help="Factor for each step")
    parser.add_argument('--size', required=False, default=nopt, help="Initial data size")
    parser.add_argument('--repeat', required=False, default=1, help="Iterations inside measured region")
    parser.add_argument('-d', type=int, default=1, help='Dimensions')
    parser.add_argument('--test', required=False, action='store_true',
                        help="Check for correctness by comparing output with naieve Python version")
    parser.add_argument('--usm', required=False, action='store_true', help="Use USM Shared or pure numpy")

    args = parser.parse_args()
    sizes = int(args.steps)
    step = int(args.step)
    nopt = int(args.size)
    repeat = int(args.repeat)
    dims = int(args.d)

    clean_string = ['make', 'clean']
    utils.run_command(clean_string, verbose=True)

    if args.usm:
        build_string = ['make', 'comp']
        utils.run_command(build_string, verbose=True)
        exec_name = "./l2_distance_comp"
    else:
        build_string = ['make']
        utils.run_command(build_string, verbose=True)
        exec_name = "./l2_distance"

    if args.test:
        X, Y = gen_data(nopt, dims)
        p_dis = l2_distance_python(X, Y)

        # run dpcpp
        gen_data_to_file(nopt, dims)
        # run the C program

        run_cmd = [exec_name, str(nopt), str(1), "-t"]
        utils.run_command(run_cmd, verbose=True)

        # read output of dpcpp
        n_dis = np.fromfile("distance.bin", np.float32)

        if np.allclose(n_dis, p_dis):
            print("Test succeeded. Python dis: ", p_dis, " Numba dis: ", n_dis, "\n")
        else:
            print("Test failed. Python dis: ", p_dis, " Numba dis: ", n_dis, "\n")
        return

    if os.path.isfile('runtimes.csv'):
        os.remove('runtimes.csv')

    for _ in xrange(sizes):
        # generate input data
        gen_data_to_file(nopt, dims)

        # run the C program
        run_cmd = [exec_name, str(nopt), str(repeat)]
        utils.run_command(run_cmd, verbose=True)
        nopt *= step
        repeat -= step
        if repeat < 1:
            repeat = 1


if __name__ == '__main__':
    run('l2_distance dpcpp')
