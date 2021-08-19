# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os, utils
from dpbench_datagen.blackscholes import gen_data_to_file

# make xrange available in python 3
try:
    xrange
except NameError:
    xrange = range

def gen_data_np(nopt):
    gen_data_to_file(nopt)
    
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

    #TODO
    if args.test:
        return

    if os.path.isfile('runtimes.csv'):
        os.remove('runtimes.csv')

    clean_string = ['make', 'clean']
    utils.run_command(clean_string, verbose=True)

    build_string = ['make']
    utils.run_command(build_string, verbose=True)        
        
    for i in xrange(sizes):
        # generate input data
        gen_data_np(nopt)            
        iterations = xrange(repeat)

        # run the C program
        run_cmd = ['./black_scholes', str(nopt), str(repeat)]
        utils.run_command(run_cmd, verbose=True)
        nopt *= step
        repeat -= step
        if repeat < 1:
            repeat = 1

if __name__ == '__main__':
    run('Blackscholes dpcpp')
