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
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

import run_utils as utils

######################################################
# GLOBAL DECLARATIONS THAT WILL BE USED IN ALL FILES #
######################################################

# make xrange available in python 3
try:
    xrange
except NameError:
    xrange = range

#################################################


class DataSize(NamedTuple):
    n_samples: int
    n_features: int


class Params(NamedTuple):
    eps: float
    minpts: int


SEED = 7777777
OPTIMAL_PARAMS = {
    DataSize(n_samples=2 ** 8, n_features=2): Params(eps=0.173, minpts=4),
    DataSize(n_samples=2 ** 8, n_features=3): Params(eps=0.35, minpts=6),
    DataSize(n_samples=2 ** 8, n_features=10): Params(eps=0.8, minpts=20),
    DataSize(n_samples=2 ** 9, n_features=2): Params(eps=0.15, minpts=4),
    DataSize(n_samples=2 ** 9, n_features=3): Params(eps=0.1545, minpts=6),
    DataSize(n_samples=2 ** 9, n_features=10): Params(eps=0.7, minpts=20),
    DataSize(n_samples=2 ** 10, n_features=2): Params(eps=0.1066, minpts=4),
    DataSize(n_samples=2 ** 10, n_features=3): Params(eps=0.26, minpts=6),
    DataSize(n_samples=2 ** 10, n_features=10): Params(eps=0.6, minpts=20),
    DataSize(n_samples=2 ** 11, n_features=2): Params(eps=0.095, minpts=4),
    DataSize(n_samples=2 ** 11, n_features=3): Params(eps=0.18, minpts=6),
    DataSize(n_samples=2 ** 11, n_features=10): Params(eps=0.6, minpts=20),
    DataSize(n_samples=2 ** 12, n_features=2): Params(eps=0.0715, minpts=4),
    DataSize(n_samples=2 ** 12, n_features=3): Params(eps=0.17, minpts=6),
    DataSize(n_samples=2 ** 12, n_features=10): Params(eps=0.6, minpts=20),
    DataSize(n_samples=2 ** 13, n_features=2): Params(eps=0.073, minpts=4),
    DataSize(n_samples=2 ** 13, n_features=3): Params(eps=0.149, minpts=6),
    DataSize(n_samples=2 ** 13, n_features=10): Params(eps=0.6, minpts=20),
    DataSize(n_samples=2 ** 14, n_features=2): Params(eps=0.0695, minpts=4),
    DataSize(n_samples=2 ** 14, n_features=3): Params(eps=0.108, minpts=6),
    DataSize(n_samples=2 ** 14, n_features=10): Params(eps=0.6, minpts=20),
    DataSize(n_samples=2 ** 15, n_features=2): Params(eps=0.0695, minpts=4),
    DataSize(n_samples=2 ** 15, n_features=3): Params(eps=0.108, minpts=6),
    DataSize(n_samples=2 ** 15, n_features=10): Params(eps=0.6, minpts=20),
    DataSize(n_samples=2 ** 16, n_features=2): Params(eps=0.0695, minpts=4),
    DataSize(n_samples=2 ** 16, n_features=3): Params(eps=0.108, minpts=6),
    DataSize(n_samples=2 ** 16, n_features=10): Params(eps=0.6, minpts=20),
}


def gen_data(n_samples, n_features, centers=10, random_state=SEED):
    X, *_ = make_blobs(
        n_samples=n_samples, n_features=n_features, centers=centers, random_state=SEED
    )
    X = StandardScaler().fit_transform(X)

    return X.flatten()


#################################################


def run(name, sizes=5, step=2, nopt=2 ** 10):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=sizes, help="Number of steps")
    parser.add_argument("--step", type=int, default=step, help="Factor for each step")
    parser.add_argument("--size", type=int, default=nopt, help="Initial data size")
    parser.add_argument(
        "--repeat", type=int, default=1, help="Iterations inside measured region"
    )
    parser.add_argument("--dims", type=int, default=10, help="Dimensions")
    parser.add_argument("--eps", type=float, default=0.6, help="Neighborhood value")
    parser.add_argument("--minpts", type=int, default=20, help="minPts")
    parser.add_argument("--skip-compile", action="store_true", help="Skip compilation")
    parser.add_argument(
        "--dry-run", action="store_true", help="Generate data and compile C code"
    )

    args = parser.parse_args()
    nopt = args.size
    repeat = args.repeat

    # delete perf_output csv and runtimes csv
    if os.path.isfile("runtimes.csv"):
        os.remove("runtimes.csv")

    if os.path.isfile("perf_output.csv"):
        os.remove("perf_output.csv")

    if not args.skip_compile:
        clean_string = ["make", "clean"]
        utils.run_command(clean_string, verbose=True)

        build_string = ["make"]
        utils.run_command(build_string, verbose=True)

    for _ in xrange(args.steps):
        data = gen_data(nopt, args.dims)

        # write data to csv file
        pd.DataFrame(data).to_csv("data.csv", header=None, index=None)

        data_size = DataSize(n_samples=nopt, n_features=args.dims)
        params = OPTIMAL_PARAMS.get(data_size, Params(eps=args.eps, minpts=args.minpts))
        # if params.eps is None or params.minpts is None:
        #     err_msg_tmpl = 'ERF: {}: Size: {} Dim: {} Eps: {} minPts: {}'
        #     raise ValueError(err_msg_tmpl.format(name, nopt, args.dims, params.eps, params.minpts))

        minpts = params.minpts or args.minpts
        eps = params.eps or args.eps

        # run the C program
        run_cmd = [
            "./dbscan",
            str(args.steps),
            str(nopt),
            str(args.dims),
            str(minpts),
            str(eps),
            str(repeat),
        ]
        utils.run_command(run_cmd, verbose=True, dry_run=args.dry_run)

        nopt *= args.step
        repeat = max(repeat - args.step, 1)


if __name__ == "__main__":
    run("DBSCAN native")
