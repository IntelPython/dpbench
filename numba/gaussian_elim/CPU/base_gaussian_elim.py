import numpy as np
import os
import json
from timeit import default_timer as now

import dpctl, dpctl.memory as dpmem


######################################################
# GLOBAL DECLARATIONS THAT WILL BE USED IN ALL FILES #
######################################################

# make xrange available in python 3
try:
    xrange
except NameError:
    xrange = range

###############################################


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


def gen_matrix(size):
    """
    Example of target matrix m with size = 4

    10.0 9.9 9.8 9.7
    9.9 10.0 9.9 9.8
    9.8 9.9 10.0 9.9
    9.7 9.8 9.9 10.0

    """

    m = np.empty(size * size, dtype=float)

    lamda = -0.01
    coef = np.empty(2 * size - 1)

    for i in range(size):
        coef_i = 10 * np.exp(lamda * i)
        j = size - 1 + i
        coef[j] = coef_i
        j = size - 1 - i
        coef[j] = coef_i

    for i in range(size):
        for j in range(size):
            m[i * size + j] = coef[size - 1 - i + j]
    
    return m


def gen_matrix_usm(size):
    m_buf = gen_matrix(size)

    with dpctl.device_context(get_device_selector()):
        m_usm = dpmem.MemoryUSMShared(size * size * np.dtype('float').itemsize)
        m_usm.copy_from_host(m_buf.view("u1"))
    
    return np.array(size * size, buffer=m_usm, dtype='i4')


def gen_vec(size, value):
    return np.full(size, value, dtype=float)


def gen_vec_usm(size, value):
    v_buf = gen_vec(size, value)

    with dpctl.device_context(get_device_selector()):
        v_usm = dpmem.MemoryUSMShared(size * np.dtype('float').itemsize)
        v_usm.copy_from_host(v_buf.view("u1"))
    
    return np.array(size, buffer=v_usm, dtype='float')


# Return result from a solved matrix
def backward_sub(a, b, x, size):
    x[size-1] = b[size-1] / a[(size-1)*size + size - 1]
    for i in range(size-2, -1, -1):
        x[i] = b[i]

        for j in range(i+1, size):
            x[i] -= a[i*size + j] * x[j]
        
        x[i] = x[i] / a[i * size + i]


def run(name, alg, steps=5, step=2, size=10):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps',  required=False, default=steps, help="Number of steps")
    parser.add_argument('--step',   required=False, default=step, help="Factor for each step")
    parser.add_argument('--size',   required=False, default=size, help="Matrix size: rows or columns number")
    parser.add_argument('--usm',    required=False, action='store_true', help="Use USM Shared or pure numpy")
    parser.add_argument('--repeat', required=False, default=1, help="Iterations inside measured region")
    parser.add_argument('--json',  required=False, default=__file__.replace('py', 'json'),
                        help="Output json data filename")

    args = parser.parse_args()
    steps= int(args.steps)
    step = int(args.step)
    size = int(args.size)
    repeat = int(args.repeat)
 
    output = {}
    output['name']      = name
    output['sizes']     = steps
    output['step']      = step
    output['repeat']    = repeat
    output['metrics']   = []

    f2 = open("runtimes.csv", 'w', 1)

    def gen_data():
        if args.usm is True:
            solve_matrix = gen_matrix_usm(size)
            coef_vec = gen_vec_usm(size, 1.0)
            extra_matrix = gen_vec_usm(size, 0.0)
        else:
            solve_matrix = gen_matrix(size)
            coef_vec = gen_vec(size, 1.0)
            extra_matrix = gen_vec(size, 0.0)
        
        return solve_matrix, coef_vec, extra_matrix
    
    for i in xrange(steps):
        solution_vec = gen_vec(size, 0.0)

        solve_matrix, coef_vec, extra_matrix = gen_data()

        # Compilation
        alg(size, solve_matrix, coef_vec, extra_matrix)

        solve_matrix, coef_vec, extra_matrix = gen_data()
        
        iterations = xrange(repeat)
        t0 = now()
        for _ in iterations:
            alg(size, solve_matrix, coef_vec, extra_matrix)
        time = now() - t0

        backward_sub(solve_matrix, coef_vec, solution_vec, size)

        print("SOLUTION: ")
        print(solution_vec)

        f2.write(str(size) + "," + str(time) + "\n")
        
        size *= step
        mops = 0.
        nopt = 0
        print("ERF: {:15s} | Size: {:10d} | MOPS: {:15.2f} | TIME: {:10.6f}".format(name, nopt, mops, time), flush=True)
        output['metrics'].append((nopt, mops, time))
        repeat -= step
        if repeat < 1:
            repeat = 1
    json.dump(output, open(args.json, 'w'), indent=2, sort_keys=True)

    f2.close()
