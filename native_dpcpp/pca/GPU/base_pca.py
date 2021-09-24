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
from sklearn.datasets import make_classification, make_regression
import pandas as pd
import utils

######################################################
# GLOBAL DECLARATIONS THAT WILL BE USED IN ALL FILES #
######################################################

# make xrange available in python 3
try:
    xrange
except NameError:
    xrange = range

#################################################


def gen_c_data(nopt, dims):
    return make_classification(n_samples=nopt, n_features=dims, random_state=0)


def gen_r_data(nopt, dims):
    return make_regression(n_samples=nopt, n_features=dims, random_state=0)


#################################################


def run(name, sizes=10, step=2, nopt=2**10):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=sizes, help='Number of steps')
    parser.add_argument('--step', type=int, default=step, help='Factor for each step')
    parser.add_argument('--size', type=int, default=nopt, help='Initial data size')
    parser.add_argument('--repeat', type=int, default=1, help='Iterations inside measured region')
    parser.add_argument('--text', default='', help='Print with each result')
    parser.add_argument('--dims', type=int, default=2**7, help='Dimensions')

    args = parser.parse_args()
    repeat = args.repeat

    # delete perf_output csv and runtimes csv
    if os.path.isfile('runtimes.csv'):
        os.remove('runtimes.csv')

    if os.path.isfile('perf_output.csv'):
        os.remove('perf_output.csv')

    clean_string = ['make', 'clean']
    utils.run_command(clean_string, verbose=True)

    build_string = ['make']
    utils.run_command(build_string, verbose=True)

    nopt = int(args.size)
    for i in xrange(args.steps):
        r_data, _ = gen_r_data(nopt, args.dims)

        # write data to csv file
        pd.DataFrame(r_data).to_csv('pca_normalized.csv', header=None, index=None)

        # run the C program
        run_cmd = ['./pca', str(nopt), str(args.dims), str(repeat)]
        utils.run_command(run_cmd, verbose=True)

        nopt *= args.step
        repeat -= args.step
        if repeat < 1:
            repeat = 1


if __name__ == '__main__':
    run('PCA native')
