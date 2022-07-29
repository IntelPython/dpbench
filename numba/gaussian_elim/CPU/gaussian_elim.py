import base_gaussian_elim
import dpctl
import numba_dpex
import numpy as np

import numba


@numba_dpex.kernel
def compute_ratio_kernel(m, a, size, t):
    i = numba_dpex.get_global_id(0)

    if i < size - 1 - t:
        m[size * (i + t + 1) + t] = (
            a[size * (i + t + 1) + t] / a[size * t + t]
        )  # ratio


@numba_dpex.kernel
def forward_sub_kernel(m, a, b, size, t):

    global_id_x = numba_dpex.get_global_id(0)
    global_id_y = numba_dpex.get_global_id(1)

    if global_id_x < size - 1 - t and global_id_y < size - t:
        a[size * (global_id_x + 1 + t) + (global_id_y + t)] -= (
            m[size * (global_id_x + 1 + t) + t]
            * a[size * t + (global_id_y + t)]
        )

        if global_id_y == 0:
            b[global_id_x + 1 + t] -= (
                m[size * (global_id_x + 1 + t) + (global_id_y + t)] * b[t]
            )


def run_gaussian_elim(
    size,
    solve_matrix,
    coef_vec,
    extra_matrix,
    global_work_size_1,
    local_work_size_buf_1,
    global_work_size_2,
    local_work_size_buf_2,
):
    # Setup and Run kernels
    for t in range(size - 1):
        with dpctl.device_context("opencl:cpu"):
            compute_ratio_kernel[
                global_work_size_1[0], local_work_size_buf_1[0]
            ](extra_matrix, solve_matrix, size, t)
            forward_sub_kernel[
                [global_work_size_2[0], global_work_size_2[1]],
                [local_work_size_buf_2[0], local_work_size_buf_2[1]],
            ](extra_matrix, solve_matrix, coef_vec, size, t)


base_gaussian_elim.run("Numba gaussian_elim", run_gaussian_elim)
