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
import sys,os,json
import numpy as np
import numpy.random as rnd

DATA_DIM = 16
SEED = 7777777


try:
    import itimer as it

    now = it.itime
    get_mops = it.itime_mops_now
except:
    from timeit import default_timer

    now = default_timer
    get_mops = lambda t0, t1, n: (n / (t1 - t0),t1-t0)

######################################################
# GLOBAL DECLARATIONS THAT WILL BE USED IN ALL FILES #
######################################################

# make xrange available in python 3
try:
    xrange
except NameError:
    xrange = range


###############################################
def gen_data_x(nopt, data_dim=DATA_DIM):
    data = rnd.rand(nopt, data_dim)
    return data


def gen_data_y(nopt, classes_num=3):
    data = rnd.randint(classes_num, size=nopt)
    return data


##############################################

def run(name, alg, sizes=10, step=2, nopt=2**10):
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=sizes,
                        help='Number of steps')
    parser.add_argument('--step', type=int, default=step,
                        help='Factor for each step')
    parser.add_argument('--size', type=int, default=nopt,
                        help='Initial data size')
    parser.add_argument('--repeat', type=int, default=1,
                        help='Iterations inside measured region')
    parser.add_argument('--text', default='', help='Print with each result')
    parser.add_argument('--json',  required=False, default=__file__.replace('py','json'), help="output json data filename")

    args = parser.parse_args()
    nopt = args.size
    repeat = args.repeat
    train_data_size = 2**10
 
    output = {}
    output['name']      = name
    output['sizes']     = sizes
    output['step']      = step
    output['repeat']    = repeat
    output['randseed']  = SEED
    output['metrics']   = []

    rnd.seed(SEED)
    
    with open('perf_output.csv', 'w', 1) as fd,  open("runtimes.csv", 'w', 1) as fd2:
        for _ in xrange(args.steps):

            x_train, y_train = gen_data_x(train_data_size), gen_data_y(train_data_size)
            x_test = gen_data_x(nopt)

            n_neighbors = 5

            sys.stdout.flush()

            predictions = alg(x_train, y_train, x_test, k=n_neighbors)  # warmup

            t0 = now()
            for _ in xrange(repeat):
                predictions = alg(x_train, y_train, x_test, k=n_neighbors)
            mops, time = get_mops(t0, now(), nopt)

            result_mops = mops * repeat
            fd.write('{},{}\n'.format(nopt, result_mops))
            fd2.write('{},{}\n'.format(nopt, time))
            
            print("ERF: {:15s} | Size: {:10d} | MOPS: {:15.2f} | TIME: {:10.6f}".format(name, nopt, mops*repeat,time),flush=True)
            output['metrics'].append((nopt,mops,time))

            nopt *= args.step
            repeat = max(repeat - args.step, 1)
    json.dump(output,open(args.json,'w'),indent=2, sort_keys=True)
