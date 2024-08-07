# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import math
from timeit import default_timer

import numba_dpex as dpex
from numba_dpex import kernel_api as kapi

now = default_timer


@dpex.kernel
def l2_norm_kernel(item: kapi.Item, a, d):
    i = item.get_id(0)
    a_rows = a.shape[1]
    d[i] = 0.0
    for k in range(a_rows):
        d[i] += a[i, k] * a[i, k]
    d[i] = math.sqrt(d[i])


def l2_norm(a, d):
    dpex.call_kernel(l2_norm_kernel, kapi.Range(a.shape[0]), a, d)


def initialize(npoints, dims, seed):
    import numpy as np
    import numpy.random as default_rng

    default_rng.seed(seed)

    return (
        default_rng.random((npoints, dims)).astype(np.float64),
        np.zeros(npoints).astype(np.float64),
    )


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


a, d = initialize(536870912, 3, 777777)

a_D = copy_to_func()(a)
d_D = copy_to_func()(d)

l2_norm(a_D, d_D)

t0 = now()
l2_norm(a_D, d_D)
t1 = now()

print("TIME: {:10.6f}".format((t1 - t0)), flush=True)
