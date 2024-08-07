# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from math import cos, log, pi, sin, sqrt
from timeit import default_timer

import numba_dpex as dpex
from numba_dpex import kernel_api as kapi

now = default_timer


@dpex.kernel
def _rambo(item: kapi.Item, C1, F1, Q1, nout, output):
    dtype = C1.dtype
    i = item.get_id(0)
    for j in range(nout):
        C = dtype.type(2.0) * C1[i, j] - dtype.type(1.0)
        S = sqrt(dtype.type(1) - C * C)
        F = dtype.type(2.0 * pi) * F1[i, j]
        Q = -log(Q1[i, j])

        output[i, j, 0] = Q
        output[i, j, 1] = Q * S * sin(F)
        output[i, j, 2] = Q * S * cos(F)
        output[i, j, 3] = Q * C


def rambo(nevts, nout, C1, F1, Q1, output):
    dpex.call_kernel(_rambo, kapi.Range(nevts), C1, F1, Q1, nout, output)


def initialize(nevts, nout):
    import numpy as np

    dtype = np.float64

    C1 = np.empty((nevts, nout), dtype=dtype)
    F1 = np.empty((nevts, nout), dtype=dtype)
    Q1 = np.empty((nevts, nout), dtype=dtype)

    np.random.seed(777)
    for i in range(nevts):
        for j in range(nout):
            C1[i, j] = np.random.rand()
            F1[i, j] = np.random.rand()
            Q1[i, j] = np.random.rand() * np.random.rand()

    return (C1, F1, Q1, np.empty((nevts, nout, 4), dtype))


def copy_to_func():
    """Returns the copy-method that should be used
    for copying the benchmark arguments."""

    def _copy_to_func_impl(ref_array):
        import dpnp

        if ref_array.flags["C_CONTIGUOUS"]:
            order = "C"
        elif ref_array.flags["F_CONTIGUOUS"]:
            order = "F"
        else:
            order = "K"
        return dpnp.asarray(
            ref_array,
            dtype=ref_array.dtype,
            order=order,
            like=None,
            usm_type=None,
            sycl_queue=None,
        )

    return _copy_to_func_impl


nevts = 16777216
nout = 4
C1, F1, Q1, output = initialize(nevts, nout)

C1_d = copy_to_func()(C1)
F1_d = copy_to_func()(F1)
Q1_d = copy_to_func()(Q1)
output_d = copy_to_func()(output)

rambo(nevts, nout, C1_d, F1_d, Q1_d, output_d)

t0 = now()
rambo(nevts, nout, C1_d, F1_d, Q1_d, output_d)
t1 = now()

print("TIME: {:10.6f}".format((t1 - t0)), flush=True)
