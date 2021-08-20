# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT


from __future__ import print_function
import numpy as np
import sys,json
from dpbench_python.blackscholes.bs_python import black_scholes_python

try:
    from numpy import erf
    numpy_ver += "-erf"
except:
    from scipy.special import erf

try:
    from numpy import invsqrt
    numpy_ver += "-invsqrt"
except:
    #from numba import jit
    invsqrt = lambda x: 1.0/np.sqrt(x)
    #invsqrt = jit(['f8(f8)','f8[:](f8[:])'])(invsqrt)

try:
    import itimer as it
    now = it.itime
    get_mops = it.itime_mops_now
except:
    from timeit import default_timer
    now = default_timer
    get_mops = lambda t0, t1, n: (n / (t1 - t0),t1-t0)

from dpbench_datagen.blackscholes import gen_rand_data

######################################################
# GLOBAL DECLARATIONS THAT WILL BE USED IN ALL FILES #
######################################################

# make xrange available in python 3
try:
    xrange
except NameError:
    xrange = range

SEED = 7777777
S0L = 10.0
S0H = 50.0
XL = 10.0
XH = 50.0
TL = 1.0
TH = 2.0
RISK_FREE = 0.1
VOLATILITY = 0.2
TEST_ARRAY_LENGTH = 1024

###############################################

def gen_data(nopt):
    return (gen_rand_data(nopt))

##############################################	

def run(name, alg, sizes=14, step=2, nopt=2**15):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', required=False, default=sizes,  help="Number of steps")
    parser.add_argument('--step',  required=False, default=step,   help="Factor for each step")
    parser.add_argument('--size',  required=False, default=nopt,   help="Initial data size")
    parser.add_argument('--repeat',required=False, default=1,    help="Iterations inside measured region")
    parser.add_argument('--text',  required=False, default="",     help="Print with each result")
    parser.add_argument('--test',  required=False, action='store_true', help="Check for correctness by comparing output with naieve Python version")
    parser.add_argument('--json',  required=False, default=__file__.replace('py','json'), help="output json data filename")
	
    args = parser.parse_args()
    sizes= int(args.steps)
    step = int(args.step)
    nopt = int(args.size)
    repeat=int(args.repeat)
 
    output = {}
    output['name']      = name
    output['sizes']     = sizes
    output['step']      = step
    output['repeat']    = repeat
    output['randseed']  = SEED
    output['metrics']   = []
    kwargs={}

    if args.test:
        price, strike, t = gen_data(nopt)
        p_call = np.zeros(nopt, dtype=np.float64)
        p_put  = -np.ones(nopt, dtype=np.float64)
        black_scholes_python(nopt, price, strike, t, RISK_FREE, VOLATILITY, p_call, p_put)

        n_call = np.zeros(nopt, dtype=np.float64)
        n_put  = -np.ones(nopt, dtype=np.float64)
        alg(nopt, price, strike, t, RISK_FREE, VOLATILITY, n_call, n_put)

        if np.allclose(n_call, p_call) and np.allclose(n_put, p_put):
            print("Test succeeded\n")
        else:
            print("Test failed\n")
        return
            
    f1 = open("perf_output.csv",'w',1)
    f2 = open("runtimes.csv",'w',1)
    
    for i in xrange(sizes):
        price, strike, t = gen_data(nopt)
        call = np.zeros(nopt, dtype=np.float64)
        put  = -np.ones(nopt, dtype=np.float64)
        iterations = xrange(repeat)
        sys.stdout.flush()

        alg(nopt, price, strike, t, RISK_FREE, VOLATILITY, call, put) #warmup
        t0 = now()
        for _ in iterations:
            alg(nopt, price, strike, t, RISK_FREE, VOLATILITY, call, put)
            
        mops,time = get_mops(t0, now(), nopt)
        print("ERF: {:15s} | Size: {:10d} | MOPS: {:15.2f} | TIME: {:10.6f}".format(name, nopt, mops*2*repeat,time),flush=True)
        output['metrics'].append((nopt,mops,time))
        f1.write(str(nopt) + "," + str(mops*2*repeat) + "\n")
        f2.write(str(nopt) + "," + str(time) + "\n")
        nopt *= step
        repeat -= step
        if repeat < 1:
            repeat = 1
    json.dump(output,open(args.json,'w'),indent=2, sort_keys=True)
    f1.close()
    f2.close()
