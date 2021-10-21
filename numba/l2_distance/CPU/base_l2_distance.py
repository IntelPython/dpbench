# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np
import json
import datetime

# import numpy.random_intel as rnd
import numpy.random as rnd

from dpbench_python.l2_distance.l2_distance_python import l2_distance_python
from dpbench_datagen.l2_distance import gen_data

try:
    import itimer as it

    now = it.itime
    get_mops = it.itime_mops_now
except:
    from timeit import default_timer

    now = default_timer
    get_mops = lambda t0, t1, n: (n / (t1 - t0), t1 - t0)

######################################################
# GLOBAL DECLARATIONS THAT WILL BE USED IN ALL FILES #
######################################################
SEED = 7777777
# make xrange available in python 3
try:
    xrange
except NameError:
    xrange = range


def run(name, alg, sizes=10, step=2, nopt=2 ** 16):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps", required=False, default=sizes, help="Number of steps"
    )
    parser.add_argument(
        "--step", required=False, default=step, help="Factor for each step"
    )
    parser.add_argument(
        "--size", required=False, default=nopt, help="Initial data size"
    )
    parser.add_argument(
        "--repeat", required=False, default=1, help="Iterations inside measured region"
    )
    parser.add_argument(
        "--text", required=False, default="", help="Print with each result"
    )
    parser.add_argument(
        "--json",
        required=False,
        default=__file__.replace("py", "json"),
        help="output json data filename",
    )
    parser.add_argument("-d", type=int, default=1, help="Dimensions")
    parser.add_argument(
        "--test",
        required=False,
        action="store_true",
        help="Check for correctness by comparing output with naieve Python version",
    )

    args = parser.parse_args()
    sizes = int(args.steps)
    step = int(args.step)
    nopt = int(args.size)
    repeat = int(args.repeat)
    dims = int(args.d)

    f = open("perf_output.csv", "w", 1)
    f2 = open("runtimes.csv", "w", 1)

    output = {}
    output['name']      = name
    output['datetime']  = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S')
    output['sizes']     = sizes
    output['step']      = step
    output['repeat']    = repeat
    output['dims']      = dims
    output['randseed']  = SEED
    output['metrics']   = []

    if args.test:
        X, Y = gen_data(nopt, dims)
        p_dis = l2_distance_python(X, Y)
        n_dis = alg(X, Y)

        # RMS error grows proportional to sqrt(n)
        # absolute(a - b) <= (atol + rtol * absolute(b))
        if np.allclose(n_dis, p_dis, rtol=1e-05 * np.sqrt(nopt)):
            print("Test succeeded. Python dis: ", p_dis, " Numba dis: ", n_dis, "\n")
        else:
            print("Test failed. Python dis: ", p_dis, " Numba dis: ", n_dis, "\n")
        return

    for i in xrange(sizes):
        X, Y = gen_data(nopt, dims)
        iterations = xrange(repeat)

        alg(X, Y)  # warmup
        t0 = now()
        for _ in iterations:
            alg(X, Y)

        mops, time = get_mops(t0, now(), nopt)
        out_msg_tmpl = "ERF: {}: Size: {} Dim: {} MOPS: {} Time: {}"
        print(
            "ERF: {:15s} | Size: {:10d} | MOPS: {:15.2f} | TIME: {:10.6f}".format(
                name, nopt, mops, time
            ),
            flush=True,
        )
        output["metrics"].append((nopt, mops, time))
        f.write(str(nopt) + "," + str(mops * 2 * repeat) + "\n")
        f2.write(str(nopt) + "," + str(time) + "\n")

        nopt *= step
        repeat -= step
        if repeat < 1:
            repeat = 1
    json.dump(output, open(args.json, "w"), indent=2, sort_keys=True)
    f.close()
    f2.close()
