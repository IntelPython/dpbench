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

import argparse
import sys
import numpy as np
import numpy.random as rnd
import sys,json
from typing import NamedTuple
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

try:
    import itimer as it

    now = it.itime
    get_mops = it.itime_mops_now
except:
    from timeit import default_timer

    now = default_timer
    get_mops = lambda t0, t1, n: (n / (t1 - t0), t1-t0)

######################################################
# GLOBAL DECLARATIONS THAT WILL BE USED IN ALL FILES #
######################################################

# make xrange available in python 3
try:
    xrange
except NameError:
    xrange = range


###############################################


class DataSize(NamedTuple):
    n_samples: int
    n_features: int


class Params(NamedTuple):
    eps: float
    minpts: int


SEED = 7777777
OPTIMAL_PARAMS = {
    DataSize(n_samples=2**8, n_features=2): Params(eps=0.173, minpts=4),
    DataSize(n_samples=2**8, n_features=3): Params(eps=0.35, minpts=6),
    DataSize(n_samples=2**8, n_features=10): Params(eps=0.8, minpts=20),
    DataSize(n_samples=2**9, n_features=2): Params(eps=0.15, minpts=4),
    DataSize(n_samples=2**9, n_features=3): Params(eps=0.1545, minpts=6),
    DataSize(n_samples=2**9, n_features=10): Params(eps=0.7, minpts=20),
    DataSize(n_samples=2**10, n_features=2): Params(eps=0.1066, minpts=4),
    DataSize(n_samples=2**10, n_features=3): Params(eps=0.26, minpts=6),
    DataSize(n_samples=2**10, n_features=10): Params(eps=0.6, minpts=20),
    DataSize(n_samples=2**11, n_features=2): Params(eps=0.095, minpts=4),
    DataSize(n_samples=2**11, n_features=3): Params(eps=0.18, minpts=6),
    DataSize(n_samples=2**11, n_features=10): Params(eps=0.6, minpts=20),
    DataSize(n_samples=2**12, n_features=2): Params(eps=0.0715, minpts=4),
    DataSize(n_samples=2**12, n_features=3): Params(eps=0.17, minpts=6),
    DataSize(n_samples=2**12, n_features=10): Params(eps=0.6, minpts=20),
    DataSize(n_samples=2**13, n_features=2): Params(eps=0.073, minpts=4),
    DataSize(n_samples=2**13, n_features=3): Params(eps=0.149, minpts=6),
    DataSize(n_samples=2**13, n_features=10): Params(eps=0.6, minpts=20),
    DataSize(n_samples=2**14, n_features=2): Params(eps=0.0695, minpts=4),
    DataSize(n_samples=2**14, n_features=3): Params(eps=0.108, minpts=6),
    DataSize(n_samples=2**14, n_features=10): Params(eps=0.6, minpts=20),
    DataSize(n_samples=2**15, n_features=2): Params(eps=0.0695, minpts=4),
    DataSize(n_samples=2**15, n_features=3): Params(eps=0.108, minpts=6),
    DataSize(n_samples=2**15, n_features=10): Params(eps=0.6, minpts=20),    
    DataSize(n_samples=2**16, n_features=2): Params(eps=0.0695, minpts=4),
    DataSize(n_samples=2**16, n_features=3): Params(eps=0.108, minpts=6),
    DataSize(n_samples=2**16, n_features=10): Params(eps=0.6, minpts=20),
}


def gen_data(n_samples, n_features, centers=10, random_state=SEED):
    X, *_ = make_blobs(n_samples=n_samples, n_features=n_features,
                       centers=centers, random_state=SEED)
    X = StandardScaler().fit_transform(X)

    return X.flatten()


##############################################

def run(name, alg, sizes=5, step=2, nopt=2**10):
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
    parser.add_argument('--json',  required=False, default=__file__.replace('py','json'), help="output json data filename")

    args = parser.parse_args()
    nopt = args.size
    repeat = args.repeat
 
    output = {}
    output['name']      = name
    output['sizes']     = sizes
    output['step']      = step
    output['repeat']    = repeat
    output['randseed']  = SEED
    output['metrics']   = []

    rnd.seed(SEED)

    with open('perf_output.csv', 'w', 1) as mops_fd, open('runtimes.csv', 'w', 1) as runtimes_fd:
        for _ in xrange(args.steps):
            data = gen_data(nopt, args.dims)
            assignments = np.empty(nopt, dtype=np.int64)

            data_size = DataSize(n_samples=nopt, n_features=args.dims)
            params = OPTIMAL_PARAMS.get(data_size, Params(eps=args.eps, minpts=args.minpts))
            # if params.eps is None or params.minpts is None:
            #     err_msg_tmpl = 'ERF: {}: Size: {} Dim: {} Eps: {} minPts: {}'
            #     raise ValueError(err_msg_tmpl.format(name, nopt, args.dims, params.eps, params.minpts))

            minpts = params.minpts or args.minpts
            eps = params.eps or args.eps

            nclusters = alg(nopt, args.dims, data, eps, minpts, assignments)  # warmup

            t0 = now()
            for _ in xrange(repeat):
                nclusters = alg(nopt, args.dims, data, eps, minpts, assignments)
            mops, time = get_mops(t0, now(), nopt)
            result_mops = mops * repeat / 1e6

            print("ERF: {:15s} | Size: {:10d} | MOPS: {:15.2f} | TIME: {:10.6f}".format(name, nopt, result_mops,time),flush=True)
            output['metrics'].append((nopt,mops,time))

            mops_fd.write('{},{},{},{},{},{}\n'.format(nopt, args.dims, eps, minpts, nclusters, result_mops))
            runtimes_fd.write('{},{},{},{},{},{}\n'.format(nopt, args.dims, eps, minpts, nclusters, time))

            nopt *= args.step
            repeat = max(repeat - args.step, 1)
    json.dump(output,open(args.json,'w'),indent=2, sort_keys=True)
