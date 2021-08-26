# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import run_utils as utils
import numpy as np
from dpbench_datagen.blackscholes import gen_data_to_file, gen_rand_data
from dpbench_python.blackscholes.bs_python import black_scholes_python

# make xrange available in python 3
try:
    xrange
except NameError:
    xrange = range

def ip_data_to_file(nopt):
    gen_data_to_file(nopt)

def gen_data_np(nopt):
    price, strike, t =  gen_rand_data(nopt)
    return (price, strike, t,
            np.zeros(nopt, dtype=np.float64),
            -np.ones(nopt, dtype=np.float64))

RISK_FREE = 0.1
VOLATILITY = 0.2
    
# create input data, call blackscholes computation function (alg)
def run(name, sizes=14, step=2, nopt=2**15):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', required=False, default=sizes,  help="Number of steps")
    parser.add_argument('--step',  required=False, default=step,   help="Factor for each step")
    parser.add_argument('--size',  required=False, default=nopt,   help="Initial data size")
    parser.add_argument('--repeat',required=False, default=1,    help="Iterations inside measured region")
    parser.add_argument('--usm',   required=False, action='store_true',  help="Use USM Shared or pure numpy")
    parser.add_argument('--test',  required=False, action='store_true', help="Check for correctness by comparing output with naieve Python version")
	
    args = parser.parse_args()
    sizes= int(args.steps)
    step = int(args.step)
    nopt = int(args.size)
    repeat=int(args.repeat)

    clean_string = ['make', 'clean']
    utils.run_command(clean_string, verbose=True)

    if args.usm:
        build_string = ['make' ,'comp']
        utils.run_command(build_string, verbose=True)
        exec_name = "./black_scholes_comp"
    else:
        build_string = ['make']
        utils.run_command(build_string, verbose=True)
        exec_name = "./black_scholes"
        
    if args.test:
        #run sequential python
        price, strike, t, p_call, p_put = gen_data_np(nopt)
        black_scholes_python(nopt, price, strike, t, RISK_FREE, VOLATILITY, p_call, p_put)

        #run dpcpp
        ip_data_to_file(nopt)
        run_cmd = [exec_name, str(nopt), str(1), "-t"]
        utils.run_command(run_cmd, verbose=True)

        #read output of dpcpp into n_call, n_put
        n_call = np.fromfile("call.bin", np.float64)

        #read output of dpcpp into n_call, n_put
        n_put = np.fromfile("put.bin", np.float64)
        
        #compare outputs
        if np.allclose(n_call, p_call) and np.allclose(n_put, p_put):
            print("Test succeeded\n")
        else:
            print("Test failed\n")        
        return

    if os.path.isfile('runtimes.csv'):
        os.remove('runtimes.csv')
        
    for i in xrange(sizes):
        # generate input data
        ip_data_to_file(nopt)

        # run the C program
        run_cmd = [exec_name, str(nopt), str(repeat)]
        utils.run_command(run_cmd, verbose=True)
        nopt *= step
        repeat -= step
        if repeat < 1:
            repeat = 1

if __name__ == '__main__':
    run('Blackscholes dpcpp')
