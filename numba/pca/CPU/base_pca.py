# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

from sklearn.datasets import make_classification, make_regression


try:
    import itimer as it
    now = it.itime
    get_mops = it.itime_mops_now
except:
    from timeit import default_timer
    now = default_timer
    get_mops = lambda t0, t1, n: (n / (t1 - t0),t1-t0)

# try:
#     import itimer as it
#     now = it.itime
#     get_mops = it.itime_mops_now
# except:
#     from timeit import default_timer
#     now = default_timer
#     get_mops = lambda t0, n: n / (1.e6 * (now() - t0))

######################################################
# GLOBAL DECLARATIONS THAT WILL BE USED IN ALL FILES #
######################################################


# make xrange available in python 3
try:
    xrange
except NameError:
    xrange = range


###############################################

def gen_c_data(nopt, dims):
    return make_classification(n_samples=nopt, n_features=dims, random_state=0)


def gen_r_data(nopt, dims):
    return make_regression(n_samples=nopt, n_features=dims, random_state=0)


##############################################

def run(name, alg, sizes=10, step=2, nopt=2**10):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', required=False, default=sizes, help="Number of steps")
    parser.add_argument('--step', required=False, default=step, help="Factor for each step")
    parser.add_argument('--size', required=False, default=nopt, help="Initial data size")
    parser.add_argument('--repeat', required=False, default=1, help="Iterations inside measured region")
    parser.add_argument('--text', required=False, default="", help="Print with each result")
    parser.add_argument('--dims', required=False, type=int, default=2**7, help='Dimensions')
    parser.add_argument('--rand', required=False, type=bool, default=True, help='Rand?')

    args = parser.parse_args()
    sizes = int(args.steps)
    step = int(args.step)
    repeat = int(args.repeat)
    dims = int(args.dims)
    rand = args.rand

    f = open("perf_output.csv", 'w',1)
    f2 = open("runtimes.csv",'w',1)

    nopt = int(args.size)
    for i in xrange(sizes):
        data, _ = gen_r_data(nopt, args.dims)
        iterations = xrange(repeat)

        op = alg(data)  # warmup
        t0 = now()
        for _ in iterations:
            op = alg(data)

        mops,time = get_mops(t0, now(), nopt)
        f.write(str(nopt) + "," + str(mops * 2 * repeat) + "\n")
        f2.write(str(nopt) + "," + str(time) + "\n")
        print(str(nopt) + "," + str(time) + "\n")
        nopt *= step
        repeat -= step
        if repeat < 1:
            repeat = 1

    f.close()
    f2.close()
