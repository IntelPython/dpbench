# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import json,os
import numpy as np
from dpbench_python.rambo.rambo_python import rambo_python

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

def get_device_selector (is_gpu = True):
    if is_gpu is True:
        device_selector = "gpu"
    else:
        device_selector = "cpu"

    if os.environ.get('SYCL_DEVICE_FILTER') is None or os.environ.get('SYCL_DEVICE_FILTER') == "opencl":
        return "opencl:" + device_selector

    if os.environ.get('SYCL_DEVICE_FILTER') == "level_zero":
        return "level_zero:" + device_selector

    return os.environ.get('SYCL_DEVICE_FILTER')

###############################################


def run(name, alg, sizes=5, step=2, nopt=2**20):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps',  required=False, default=sizes,  help="Number of steps")
    parser.add_argument('--step',   required=False, default=step,   help="Factor for each step")
    parser.add_argument('--size',   required=False, default=nopt,   help="Initial data size")
    parser.add_argument('--repeat', required=False, default=1,    help="Iterations inside measured region")
    parser.add_argument('--text',   required=False, default="",     help="Print with each result")
    parser.add_argument('--json',  required=False, default=__file__.replace('py','json'), help="output json data filename")
    parser.add_argument('--usm',   required=False, action='store_true',  help="Use USM Shared or pure numpy")
    parser.add_argument('--test',  required=False, action='store_true', help="Check for correctness by comparing output with naieve Python version")

    args = parser.parse_args()
    sizes = int(args.steps)
    step = int(args.step)
    nopt = int(args.size)
    repeat = int(args.repeat)

    output = {}
    output['name']      = name
    output['sizes']     = sizes
    output['step']      = step
    output['repeat']    = repeat
    output['metrics']   = []

    if args.test:
        e_p = rambo_python(nopt)
        e_n = alg(nopt)

        if np.allclose(e_p, e_n):
            print("Test succeeded\n")
        else:
            print("Test failed\n")
        return

    f = open("perf_output.csv", 'w', 1)
    f2 = open("runtimes.csv", 'w', 1)

    for i in xrange(sizes):
        iterations = xrange(repeat)

        alg(nopt) #warmup
        t0 = now()
        for _ in iterations:
            alg(nopt)

        mops, time = get_mops(t0, now(), nopt)
        f.write(str(nopt) + "," + str(mops*repeat/1e6) + "\n")
        f2.write(str(nopt) + "," + str(time) + "\n")
        print("ERF: {:15s} | Size: {:10d} | MOPS: {:15.2f} | TIME: {:10.6f}".format(name, nopt, mops*repeat,time),flush=True)
        output['metrics'].append((nopt,mops,time))

        nopt *= step
        repeat -= step
        if repeat < 1:
            repeat = 1

    json.dump(output,open(args.json,'w'),indent=2, sort_keys=True)
    f.close()
    f2.close()
