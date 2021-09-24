# *****************************************************************************
# Copyright (c) 2020, Intel Corporation All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#     Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************

import os
import numpy as np
import run_utils as utils
from dpbench_python.dbscan.dbscan_python import dbscan_python
from dpbench_datagen.dbscan import gen_rand_data, gen_data_to_file

######################################################
# GLOBAL DECLARATIONS THAT WILL BE USED IN ALL FILES #
######################################################

# make xrange available in python 3
try:
    xrange
except NameError:
    xrange = range

#################################################
def gen_data_np(nopt, dims, a_minpts, a_eps):
    data, p_eps, p_minpts = gen_rand_data(nopt, dims)
    assignments = np.empty(nopt, dtype=np.int64)

    minpts = p_minpts or a_minpts
    eps = p_eps or a_eps

    return (data, assignments, eps, minpts)

#################################################

def run(name, sizes=5, step=2, nopt=2**10):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=sizes,
                        help='Number of steps')
    parser.add_argument('--step', type=int, default=step,
                        help='Factor for each step')
    parser.add_argument('--size', type=int, default=nopt,
                        help='Initial data size')
    parser.add_argument('--repeat', type=int, default=1,
                        help='Iterations inside measured region')
    parser.add_argument('--dims', type=int, default=10, help='Dimensions')
    parser.add_argument('--eps', type=float, default=0.6, help='Neighborhood value')
    parser.add_argument('--minpts', type=int, default=20, help='minPts')
    parser.add_argument('--usm',   required=False, action='store_true',  help="Use USM Shared or pure numpy")
    parser.add_argument('--test',  required=False, action='store_true', help="Check for correctness by comparing output with naieve Python version")

    args = parser.parse_args()
    nopt = args.size
    repeat = args.repeat

    if args.usm:
        print("Warn: Compute only measurement not available for DBSCAN since it executes both on host and device\n")

    build_string = ['make']
    utils.run_command(build_string, verbose=True)
    exec_name = "./dbscan"

    if args.test:
        data, p_assignments, eps, minpts = gen_data_np(nopt, args.dims, args.minpts, args.eps)
        p_nclusters = dbscan_python(nopt, args.dims, data, eps, minpts, p_assignments)

        # if args.usm is True:
        #     data, assignments, eps, minpts = gen_data_usm(nopt, args.dims, args.minpts, args.eps)
        #     n_nclusters = alg(nopt, args.dims, data, eps, minpts, assignments)
        # else:
        eps, minpts = gen_data_to_file(nopt, args.dims)
        run_cmd = [exec_name, str(nopt), str(args.dims), str(minpts), str(eps), str(1), "-t"]
        utils.run_command(run_cmd, verbose=True)

        n_assignments = np.fromfile("assignments.bin", np.int64)

        if np.allclose(n_assignments, p_assignments):
            print("Test succeeded.\n")
            print("n_assignments = ", n_assignments, "\n p_assignments = ", p_assignments)
        else:
            print("Test failed.\n")
            print("n_assignments = ", n_assignments, "\n p_assignments = ", p_assignments)
        return

    # delete perf_output csv and runtimes csv
    if os.path.isfile('runtimes.csv'):
        os.remove('runtimes.csv')

    if os.path.isfile('perf_output.csv'):
        os.remove('perf_output.csv')

    for _ in xrange(args.steps):
        eps, minpts = gen_data_to_file(nopt, args.dims)

        # run the C program
        run_cmd = [exec_name, str(nopt), str(args.dims), str(minpts), str(eps), str(repeat)]
        utils.run_command(run_cmd, verbose=True)

        nopt *= args.step
        repeat = max(repeat - args.step, 1)


if __name__ == '__main__':
    run('DBSCAN native')
