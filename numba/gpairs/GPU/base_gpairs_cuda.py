# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np

try:
    import itimer as it
    now = it.itime
    get_mops = it.itime_mops_now
except:
    from timeit import default_timer
    now = default_timer
    get_mops = lambda t0, t1, n: (n / (t1 - t0),t1-t0)

from gpairs.pair_counter.tests.generate_test_data import (
    DEFAULT_RBINS_SQUARED)
from numba import cuda

######################################################
# GLOBAL DECLARATIONS THAT WILL BE USED IN ALL FILES #
######################################################

# make xrange available in python 3
try:
    xrange
except NameError:
    xrange = range

###############################################

def gen_data(npoints):
    Lbox = 500.
    from gpairs.pair_counter.tests import random_weighted_points
    n1 = npoints
    n2 = npoints
    x1, y1, z1, w1 = random_weighted_points(n1, Lbox, 0)
    x2, y2, z2, w2 = random_weighted_points(n2, Lbox, 1)

    return (
        x1, y1, z1, w1, x2, y2, z2, w2
    )

def copy_h2d(x1, y1, z1, w1, x2, y2, z2, w2):
    d_x1 = cuda.to_device(x1.astype(np.float32))
    d_y1 = cuda.to_device(y1.astype(np.float32))
    d_z1 = cuda.to_device(z1.astype(np.float32))
    d_w1 = cuda.to_device(w1.astype(np.float32))

    d_x2 = cuda.to_device(x2.astype(np.float32))
    d_y2 = cuda.to_device(y2.astype(np.float32))
    d_z2 = cuda.to_device(z2.astype(np.float32))
    d_w2 = cuda.to_device(w2.astype(np.float32))

    d_rbins_squared = cuda.to_device(
        DEFAULT_RBINS_SQUARED.astype(np.float32))

    return (
        d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2, d_rbins_squared
    )

def copy_d2h(d_result):
    return d_result.copy_to_host()

##############################################	

def run(name, alg, sizes=10, step=2, nopt=2**10):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', required=False, default=sizes,  help="Number of steps")
    parser.add_argument('--step',  required=False, default=step,   help="Factor for each step")
    parser.add_argument('--size',  required=False, default=nopt,   help="Initial data size")
    parser.add_argument('--repeat',required=False, default=100,    help="Iterations inside measured region")
    parser.add_argument('--text',  required=False, default="",     help="Print with each result")
    
    args = parser.parse_args()
    sizes= int(args.steps)
    step = int(args.step)
    nopt = int(args.size)
    repeat=int(args.repeat)

    f=open("perf_output.csv",'w')
    f2 = open("runtimes.csv",'w',1)
    
    for i in xrange(sizes):
        x1, y1, z1, w1, x2, y2, z2, w2 = gen_data(nopt)
        d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2, d_rbins_squared = copy_h2d(x1, y1, z1, w1, x2, y2, z2, w2)
        iterations = xrange(repeat)
        #print("ERF: {}: Size: {}".format(name, nopt), end=' ', flush=True)
        #sys.stdout.flush()

        alg(d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2, d_rbins_squared) #warmup
        t0 = now()
        for _ in iterations:
            #t1 = now()
            alg(d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2, d_rbins_squared)
            #print("Time:", now()-t1)

        mops,time = get_mops(t0, nopt)
        f.write(str(nopt) + "," + str(mops*2*repeat) + "\n")
        f2.write(str(nopt) + "," + str(time) + "\n")
        #print("MOPS:", mops*2*repeat, args.text)
        nopt *= step
        repeat -= step
        if repeat < 1:
            repeat = 1
    f.close()
    f2.close()
