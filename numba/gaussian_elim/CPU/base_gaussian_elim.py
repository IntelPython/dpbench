import numpy as np
import os
import json
from timeit import default_timer as now
from dpbench_datagen.gaussian_elim import gen_matrix, gen_vec

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


def get_device_selector(is_gpu=True):
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

    return global_work_size_1, local_work_size_buf_1, global_work_size_2, local_work_size_buf_2


def gen_data(size):
        solve_matrix = gen_matrix(size)
        coef_vec = gen_vec(size, 1.0)
        extra_matrix = gen_vec(size, 0.0)

        return solve_matrix, coef_vec, extra_matrix

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
        "--repeat", required=False, default=1, help="Iterations inside measured region"
    )
    parser.add_argument(
        "--json",
        required=False,
        default=__file__.replace("py", "json"),
        help="Output json data filename",
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

    for _ in xrange(steps):
        solution_vec = gen_vec(size, 0.0)

        solve_matrix, coef_vec, extra_matrix = gen_data(size)

        global_work_size_1, local_work_size_buf_1, global_work_size_2, local_work_size_buf_2 = set_block_size(size)

        # Compilation
        alg(size, solve_matrix, coef_vec, extra_matrix, global_work_size_1, local_work_size_buf_1, global_work_size_2, local_work_size_buf_2)

        solve_matrix, coef_vec, extra_matrix = gen_data(size)

        iterations = xrange(repeat)
        times = np.empty(repeat)

        for i in iterations:
            t0 = now()
            # alg(size, solve_matrix, coef_vec, extra_matrix, global_work_size_1, local_work_size_buf_1, global_work_size_2, local_work_size_buf_2)
            time = now() - t0
            times[i] = time
        backward_sub(solve_matrix, coef_vec, solution_vec, size)

        print("SOLUTION: ")
        print(solution_vec)

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
        output["metrics"].append((size, mops, time))

    json.dump(output, open(args.json, "a"), indent=2, sort_keys=True)

    f2.close()
