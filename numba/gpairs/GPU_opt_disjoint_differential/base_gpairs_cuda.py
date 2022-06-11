# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import json
import os

import numpy as np
from dpbench_datagen.gpairs import gen_rand_data
from dpbench_python.gpairs.gpairs_python import gpairs_python

from numba import cuda

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

# make xrange available in python 3
try:
    xrange
except NameError:
    xrange = range

###############################################


def gen_data_np(npoints, dtype=np.float64):
    x1, y1, z1, w1, x2, y2, z2, w2, DEFAULT_RBINS_SQUARED = gen_rand_data(
        npoints, dtype
    )
    result = np.zeros_like(DEFAULT_RBINS_SQUARED)[:-1].astype(dtype)
    return (x1, y1, z1, w1, x2, y2, z2, w2, DEFAULT_RBINS_SQUARED, result)


def gen_data_device(npoints, dtype=np.float64):
    # init numpy obj
    x1, y1, z1, w1, x2, y2, z2, w2, DEFAULT_RBINS_SQUARED, result = gen_data_np(
        npoints, dtype
    )

    d_x1 = cuda.to_device(x1.astype(dtype))
    d_y1 = cuda.to_device(y1.astype(dtype))
    d_z1 = cuda.to_device(z1.astype(dtype))
    d_w1 = cuda.to_device(w1.astype(dtype))

    d_x2 = cuda.to_device(x2.astype(dtype))
    d_y2 = cuda.to_device(y2.astype(dtype))
    d_z2 = cuda.to_device(z2.astype(dtype))
    d_w2 = cuda.to_device(w2.astype(dtype))

    d_rbins_squared = cuda.to_device(DEFAULT_RBINS_SQUARED.astype(dtype))
    d_result = cuda.to_device(result.astype(dtype))

    return (
        d_x1,
        d_y1,
        d_z1,
        d_w1,
        d_x2,
        d_y2,
        d_z2,
        d_w2,
        d_rbins_squared,
        d_result,
    )


def copy_d2h(d_result):
    return d_result.copy_to_host()


##############################################


def run(name, alg, sizes=10, step=2, nopt=2**10):
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
        "--repeat",
        required=False,
        default=100,
        help="Iterations inside measured region",
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
    parser.add_argument(
        "--device",
        required=False,
        action="store_true",
        help="Copy data to device and exlude it from timing.",
    )
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

    output = {}
    output["name"] = name
    output["sizes"] = sizes
    output["step"] = step
    output["repeat"] = repeat
    output["metrics"] = []

    if args.test:
        (
            x1,
            y1,
            z1,
            w1,
            x2,
            y2,
            z2,
            w2,
            DEFAULT_RBINS_SQUARED,
            result_p,
        ) = gen_data_np(nopt)
        gpairs_python(
            x1, y1, z1, w1, x2, y2, z2, w2, DEFAULT_RBINS_SQUARED, result_p
        )

        if args.device is True:  # test usm feature
            (
                x1,
                y1,
                z1,
                w1,
                x2,
                y2,
                z2,
                w2,
                DEFAULT_RBINS_SQUARED,
                result_device,
            ) = gen_data_device(nopt)
            alg(
                x1,
                y1,
                z1,
                w1,
                x2,
                y2,
                z2,
                w2,
                DEFAULT_RBINS_SQUARED,
                result_device,
            )
            result_n = copy_d2h(result_device)
        else:
            (
                x1_n,
                y1_n,
                z1_n,
                w1_n,
                x2_n,
                y2_n,
                z2_n,
                w2_n,
                DEFAULT_RBINS_SQUARED,
                result_n,
            ) = gen_data_np(nopt)

            # pass numpy generated data to kernel
            alg(x1, y1, z1, w1, x2, y2, z2, w2, DEFAULT_RBINS_SQUARED, result_n)

        if np.allclose(result_p, result_n):
            print("Test succeeded\n")
        else:
            print("Test failed\n")
        return

    f = open("perf_output.csv", "w")
    f2 = open("runtimes.csv", "w", 1)

    for i in xrange(sizes):
        if args.device is True:
            (
                x1,
                y1,
                z1,
                w1,
                x2,
                y2,
                z2,
                w2,
                DEFAULT_RBINS_SQUARED,
                result,
            ) = gen_data_device(nopt)
        else:
            (
                x1,
                y1,
                z1,
                w1,
                x2,
                y2,
                z2,
                w2,
                DEFAULT_RBINS_SQUARED,
                result,
            ) = gen_data_np(nopt)
        iterations = xrange(repeat)

        alg(
            x1, y1, z1, w1, x2, y2, z2, w2, DEFAULT_RBINS_SQUARED, result
        )  # warmup
        t0 = now()
        for _ in iterations:
            alg(x1, y1, z1, w1, x2, y2, z2, w2, DEFAULT_RBINS_SQUARED, result)

        mops, time = get_mops(t0, now(), nopt)
        f.write(str(nopt) + "," + str(mops * 2 * repeat) + "\n")
        f2.write(str(nopt) + "," + str(time) + "\n")
        print(
            "ERF: {:15s} | Size: {:10d} | MOPS: {:15.2f} | TIME: {:10.6f}".format(
                name, nopt, mops * 2 * repeat, time
            ),
            flush=True,
        )
        output["metrics"].append((nopt, mops, time))
        nopt *= step
        repeat -= step
        if repeat < 1:
            repeat = 1
    json.dump(output, open(args.json, "w"), indent=2, sort_keys=True)
    f.close()
    f2.close()
