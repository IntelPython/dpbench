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
import sys, os, json
import numpy as np
from dpbench_python.dbscan.dbscan_python import dbscan_python
from dpbench_datagen.dbscan import gen_rand_data

try:
    import dpctl, dpctl.tensor as dpt
except ImportError:
    pass

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


def get_device_selector(is_gpu=False):
    if is_gpu is True:
        device_selector = "gpu"
    else:
        device_selector = "cpu"

    if (
        os.environ.get("SYCL_DEVICE_FILTER") is None
        or os.environ.get("SYCL_DEVICE_FILTER") == "opencl"
    ):
        return "opencl:" + device_selector

    if os.environ.get("SYCL_DEVICE_FILTER") == "level_zero":
        return "level_zero:" + device_selector

    return os.environ.get("SYCL_DEVICE_FILTER")


##############################################


def gen_data_usm(nopt, dims, a_minpts, a_eps):
    data, p_eps, p_minpts = gen_rand_data(nopt, dims)
    assignments = np.empty(nopt, dtype=np.int64)

    with dpctl.device_context(get_device_selector()) as gpu_queue:
        data_usm = dpt.usm_ndarray(
            data.shape,
            dtype=data.dtype,
            buffer="device",
            buffer_ctor_kwargs={"queue": gpu_queue},
        )
        assignments_usm = dpt.usm_ndarray(
            assignments.shape,
            dtype=assignments.dtype,
            buffer="device",
            buffer_ctor_kwargs={"queue": gpu_queue},
        )

    data_usm.usm_data.copy_from_host(data.view("u1"))
    assignments_usm.usm_data.copy_from_host(assignments.view("u1"))

    minpts = p_minpts or a_minpts
    eps = p_eps or a_eps

    return (data_usm, assignments_usm, eps, minpts)


def gen_data_np(nopt, dims, a_minpts, a_eps):
    data, p_eps, p_minpts = gen_rand_data(nopt, dims)
    assignments = np.empty(nopt, dtype=np.int64)

    minpts = p_minpts or a_minpts
    eps = p_eps or a_eps

    return (data, assignments, eps, minpts)


def run(name, alg, sizes=5, step=2, nopt=2 ** 10):
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=sizes, help="Number of steps")
    parser.add_argument("--step", type=int, default=step, help="Factor for each step")
    parser.add_argument("--size", type=int, default=nopt, help="Initial data size")
    parser.add_argument(
        "--repeat", type=int, default=1, help="Iterations inside measured region"
    )
    parser.add_argument("--dims", type=int, default=10, help="Dimensions")
    parser.add_argument("--eps", type=float, default=0.6, help="Neighborhood value")
    parser.add_argument("--minpts", type=int, default=20, help="minPts")
    parser.add_argument(
        "--json",
        required=False,
        default=__file__.replace("py", "json"),
        help="output json data filename",
    )
    parser.add_argument(
        "--test",
        required=False,
        action="store_true",
        help="Check for correctness by comparing output with naieve Python version",
    )
    parser.add_argument(
        "--usm",
        required=False,
        action="store_true",
        help="Use USM Shared or pure numpy",
    )

    args = parser.parse_args()
    nopt = args.size
    repeat = args.repeat

    output = {}
    output["name"] = name
    output["sizes"] = sizes
    output["step"] = step
    output["repeat"] = repeat
    output["metrics"] = []

    if args.usm:
        print(
            "Warn: Use of USM data not supported since DBSCAN has both host and device executions. Using numpy input data instead of USM\n"
        )

    if args.test:
        data, p_assignments, eps, minpts = gen_data_np(
            nopt, args.dims, args.minpts, args.eps
        )
        p_nclusters = dbscan_python(nopt, args.dims, data, eps, minpts, p_assignments)

        # if args.usm is True:
        #     data, assignments, eps, minpts = gen_data_usm(nopt, args.dims, args.minpts, args.eps)
        #     n_nclusters = alg(nopt, args.dims, data, eps, minpts, assignments)
        # else:
        data, n_assignments, eps, minpts = gen_data_np(
            nopt, args.dims, args.minpts, args.eps
        )
        n_nclusters = alg(nopt, args.dims, data, eps, minpts, n_assignments)

        if np.allclose(n_nclusters, p_nclusters) and np.allclose(
            n_assignments, p_assignments
        ):
            print(
                "Test succeeded. Python clusters = ",
                p_nclusters,
                ", numba clusters = ",
                n_nclusters,
                "\n",
            )
            print(
                "n_assignments = ", n_assignments, "\n p_assignments = ", p_assignments
            )
        else:
            print(
                "Test failed. Python clusters = ",
                p_nclusters,
                ", numba clusters = ",
                n_nclusters,
                "\n",
            )
            print(
                "n_assignments = ", n_assignments, "\n p_assignments = ", p_assignments
            )
        return

    with open("perf_output.csv", "w", 1) as mops_fd, open(
        "runtimes.csv", "w", 1
    ) as runtimes_fd:
        for _ in xrange(args.steps):
            data, assignments, eps, minpts = gen_data_np(
                nopt, args.dims, args.minpts, args.eps
            )
            nclusters = alg(nopt, args.dims, data, eps, minpts, assignments)  # warmup

            t0 = now()
            for _ in xrange(repeat):
                nclusters = alg(nopt, args.dims, data, eps, minpts, assignments)
            mops, time = get_mops(t0, now(), nopt)
            result_mops = mops * repeat / 1e6

            print(
                "ERF: {:15s} | Size: {:10d} | MOPS: {:15.2f} | TIME: {:10.6f}".format(
                    name, nopt, result_mops, time
                ),
                flush=True,
            )
            output["metrics"].append((nopt, mops, time))

            mops_fd.write(
                "{},{},{},{},{},{}\n".format(
                    nopt, args.dims, eps, minpts, nclusters, result_mops
                )
            )
            runtimes_fd.write(
                "{},{},{},{},{},{}\n".format(
                    nopt, args.dims, eps, minpts, nclusters, time
                )
            )

            nopt *= args.step
            repeat = max(repeat - args.step, 1)
    json.dump(output, open(args.json, "w"), indent=2, sort_keys=True)
