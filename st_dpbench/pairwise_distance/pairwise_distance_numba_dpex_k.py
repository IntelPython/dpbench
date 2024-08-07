# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from timeit import default_timer

import dpnp as np
import numba_dpex as dpex
from numba_dpex import kernel_api as kapi

now = default_timer


@dpex.kernel
def _pairwise_distance_kernel(item: kapi.Item, X1, X2, D):
    i = item.get_id(1)
    j = item.get_id(0)

    X1_cols = X1.shape[1]

    d = X1.dtype.type(0.0)
    for k in range(X1_cols):
        tmp = X1[i, k] - X2[j, k]
        d += tmp * tmp
    D[i, j] = np.sqrt(d)


def pairwise_distance(X1, X2, D):
    dpex.call_kernel(
        _pairwise_distance_kernel,
        kapi.Range(X2.shape[0], X1.shape[0]),
        X1,
        X2,
        D,
    )


def initialize(npoints, dims, seed):
    import numpy as np
    import numpy.random as default_rng

    default_rng.seed(seed)

    return (
        default_rng.random((npoints, dims)).astype(np.float64),
        default_rng.random((npoints, dims)).astype(np.float64),
        np.empty((npoints, npoints), np.float64),
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


X1, X2, D = initialize(44032, 3, 7777777)
X1_d = copy_to_func()(X1)
X2_d = copy_to_func()(X2)
D_d = copy_to_func()(D)

pairwise_distance(X1_d, X2_d, D_d)

t0 = now()
pairwise_distance(X1_d, X2_d, D_d)
t1 = now()

print("TIME: {:10.6f}".format((t1 - t0)), flush=True)
