# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT


import cupy as cp
import cupy.random as rnd

try:
    import itimer as it
    now = it.itime
    get_mops = it.itime_mops_now
except:
    from timeit import default_timer
    now = default_timer
    get_mops = lambda t0, n: n / (now() - t0)

######################################################
# GLOBAL DECLARATIONS THAT WILL BE USED IN ALL FILES #
######################################################

# make xrange available in python 3
try:
    xrange
except NameError:
    xrange = range

###############################################

def gen_data(nopt,dims):
    return (
        rnd.random((nopt, dims)),
        rnd.random((nopt, dims))
    )

##############################################	

def run(name, alg, sizes=15, step=2, nopt=c**16):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', required=False, default=sizes,  help="Number of steps")
    parser.add_argument('--step',  required=False, default=step,   help="Factor for each step")
    parser.add_argument('--size',  required=False, default=nopt,   help="Initial data size")
    parser.add_argument('--repeat',required=False, default=100,    help="Iterations inside measured region")
    parser.add_argument('--text',  required=False, default="",     help="Print with each result")
    parser.add_argument('-d', type=int, default=1, help='Dimensions')
    
    args = parser.parse_args()
    sizes= int(args.steps)
    step = int(args.step)
    nopt = int(args.size)
    repeat=int(args.repeat)
    dims = int(args.d)

    rnd.seed(SEED)

    f=open("perf_output.csv",'w')
    for i in xrange(sizes):
        X,Y = gen_data(nopt,dims)
        iterations = xrange(repeat)
        #print("ERF: {}: Size: {}".format(name, nopt), end=' ', flush=True)
        #sys.stdout.flush()

        alg(X,Y) #warmup
        cp.cuda.runtime.deviceSynchronize() #cuda.Stream.null.synchronize()
        t0 = now()
        for _ in iterations:
            alg(X,Y)
            cp.cuda.runtime.deviceSynchronize() #cuda.Stream.null.synchronize()

        mops = get_mops(t0, nopt)
        #print("MOPS:", mops*2*repeat, args.text)
        f.write(str(nopt) + "," + str(mops*2*repeat) + "\n")
        nopt *= step
        repeat -= step
        if repeat < 1:
            repeat = 1
    f.close()
