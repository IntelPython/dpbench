import numpy as np
import os
import json
from timeit import default_timer as now
from dpbench_datagen.gaussian_elim import gen_matrix, gen_vec
from device_selector import get_device_selector
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

BLOCK_SIZE_0 = 256
BLOCK_SIZE_1_X = 16
BLOCK_SIZE_1_Y = 16


def gen_matrix_usm(size):
    m_buf = gen_matrix(size)

    with dpctl.device_context(get_device_selector(is_gpu=True)):
        m_usm = dpmem.MemoryUSMShared(size * size * np.dtype("float").itemsize)
        m_usm.copy_from_host(m_buf.view("u1"))

    return np.array(size * size, buffer=m_usm, dtype="i4")


def gen_vec_usm(size, value):
    v_buf = gen_vec(size, value)

    with dpctl.device_context(get_device_selector(is_gpu=True)):
        v_usm = dpmem.MemoryUSMShared(size * np.dtype("float").itemsize)
        v_usm.copy_from_host(v_buf.view("u1"))

    return np.array(size, buffer=v_usm, dtype="float")


def set_block_size(size):
    # Determine block sizes
    local_work_size_buf_1 = [BLOCK_SIZE_0]
    local_work_size_buf_2 = [BLOCK_SIZE_1_X, BLOCK_SIZE_1_Y]

    global_work_size_1 = [size]
    global_work_size_2 = [size, size]

    if local_work_size_buf_1[0]:
        global_work_size_1[0] = (
            int(np.ceil(global_work_size_1[0] / local_work_size_buf_1[0]))
            * local_work_size_buf_1[0]
        )

    if local_work_size_buf_2[0]:
        global_work_size_2[0] = (
            int(np.ceil(global_work_size_2[0] / local_work_size_buf_2[0]))
            * local_work_size_buf_2[0]
        )
        global_work_size_2[1] = (
            int(np.ceil(global_work_size_2[1] / local_work_size_buf_2[1]))
            * local_work_size_buf_2[1]
        )

    return (
        global_work_size_1,
        local_work_size_buf_1,
        global_work_size_2,
        local_work_size_buf_2,
    )


# Return result from a solved matrix
def backward_sub(a, b, x, size):
    x[size - 1] = b[size - 1] / a[(size - 1) * size + size - 1]
    for i in range(size - 2, -1, -1):
        x[i] = b[i]

        for j in range(i + 1, size):
            x[i] -= a[i * size + j] * x[j]

        x[i] = x[i] / a[i * size + i]


def run(name, alg, steps=5, step=2, size=10):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps", required=False, default=steps, help="Number of steps"
    )
    parser.add_argument(
        "--step", required=False, default=step, help="Factor for each step"
    )
    parser.add_argument(
        "--size",
        required=False,
        default=size,
        help="Matrix size: rows or columns number",
    )
    parser.add_argument(
        "--usm",
        required=False,
        action="store_true",
        help="Use USM Shared or pure numpy",
    )
    parser.add_argument(
        "--repeat", required=False, default=1, help="Iterations inside measured region"
    )
    parser.add_argument(
        "--json",
        required=False,
        default=__file__.replace("py", "json"),
        help="Output json data filename",
    )
    parser.add_argument(
        "--test",
        required=False,
        action="store_true",
        help="Check for correctness by comparing output with naieve Python version",
    )

    args = parser.parse_args()
    steps = int(args.steps)
    step = int(args.step)
    size = int(args.size)
    repeat = int(args.repeat)

    output = {}
    output["name"] = name
    output["sizes"] = steps
    output["step"] = step
    output["repeat"] = repeat
    output["metrics"] = []

    f2 = open("runtimes.csv", "a", 1)

    def gen_data(n):
        if args.usm is True:
            solve_matrix = gen_matrix_usm(n)
            coef_vec = gen_vec_usm(n, 1.0)
            extra_matrix = gen_vec_usm(n, 0.0)
        else:
            solve_matrix = gen_matrix(n)
            coef_vec = gen_vec(n, 1.0)
            extra_matrix = gen_vec(n, 0.0)

        return solve_matrix, coef_vec, extra_matrix

    if args.test:
        reference_result = [5.02e-02, 5.00e-04, 5.00e-04, 5.02e-02]
        ref_size = 4

        (
            global_work_size_1,
            local_work_size_buf_1,
            global_work_size_2,
            local_work_size_buf_2,
        ) = set_block_size(ref_size)
        solution_vec = gen_vec(ref_size, 0.0)
        solve_matrix, coef_vec, extra_matrix = gen_data(ref_size)

        import pdb

        pdb.set_trace()

        alg(
            ref_size,
            solve_matrix,
            coef_vec,
            extra_matrix,
            global_work_size_1,
            local_work_size_buf_1,
            global_work_size_2,
            local_work_size_buf_2,
        )

        backward_sub(solve_matrix, coef_vec, solution_vec, ref_size)

        if np.allclose(solution_vec, reference_result):
            print(
                "Test succeeded. Python result: ",
                reference_result,
                "Numba result: ",
                solution_vec,
                "\n",
            )
        else:
            print(
                "Test failed. Python result: ",
                reference_result,
                "Numba result: ",
                solution_vec,
                "\n",
            )
        return

    for _ in xrange(steps):
        (
            global_work_size_1,
            local_work_size_buf_1,
            global_work_size_2,
            local_work_size_buf_2,
        ) = set_block_size(size)

        solution_vec = gen_vec(size, 0.0)

        solve_matrix, coef_vec, extra_matrix = gen_data(size)
        # Warm up
        alg(
            size,
            solve_matrix,
            coef_vec,
            extra_matrix,
            global_work_size_1,
            local_work_size_buf_1,
            global_work_size_2,
            local_work_size_buf_2,
        )

        solve_matrix, coef_vec, extra_matrix = gen_data(size)

        iterations = xrange(repeat)
        times = np.empty(repeat)

        for i in iterations:
            t0 = now()
            alg(
                size,
                solve_matrix,
                coef_vec,
                extra_matrix,
                global_work_size_1,
                local_work_size_buf_1,
                global_work_size_2,
                local_work_size_buf_2,
            )
            time = now() - t0
            times[i] = time

        backward_sub(solve_matrix, coef_vec, solution_vec, size)

        result_time = np.median(times)

        f2.write(str(size) + "," + str(result_time) + "\n")

        size *= step
        mops = 0.0
        print(
            "ERF: {:15s} | Size: {:10d} | MOPS: {:15.2f} | TIME: {:10.6f}".format(
                name, size, mops, result_time
            ),
            flush=True,
        )
        output["metrics"].append((size, mops, result_time))

    json.dump(output, open(args.json, "a"), indent=2, sort_keys=True)

    f2.close()
