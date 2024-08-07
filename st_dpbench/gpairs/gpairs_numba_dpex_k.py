# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from timeit import default_timer

import dpnp
import numba_dpex as dpex
import numpy as np
from numba_dpex import kernel_api as kapi

now = default_timer


def _generate_rbins(dtype, nbins, rmax, rmin):
    rbins = np.logspace(np.log10(rmin), np.log10(rmax), nbins).astype(dtype)

    return (rbins**2).astype(dtype)


def initialize(nopt, seed, nbins, rmax, rmin):
    import numpy.random as default_rng

    default_rng.seed(seed)
    dtype = np.float64
    x1 = np.random.randn(nopt).astype(dtype)
    y1 = np.random.randn(nopt).astype(dtype)
    z1 = np.random.randn(nopt).astype(dtype)
    w1 = np.random.rand(nopt).astype(dtype)
    w1 = w1 / np.sum(w1)

    x2 = np.random.randn(nopt).astype(dtype)
    y2 = np.random.randn(nopt).astype(dtype)
    z2 = np.random.randn(nopt).astype(dtype)
    w2 = np.random.rand(nopt).astype(dtype)
    w2 = w2 / np.sum(w2)

    rbins = _generate_rbins(dtype=dtype, rmin=rmin, rmax=rmax, nbins=nbins)
    results = np.zeros_like(rbins).astype(dtype)
    return (x1, y1, z1, w1, x2, y2, z2, w2, rbins, results)


@dpex.kernel
def count_weighted_pairs_3d_diff_ker(
    item: kapi.Item,
    n,
    nbins,
    x1,
    y1,
    z1,
    w1,
    x2,
    y2,
    z2,
    w2,
    rbins_squared,
    result,
):
    i = item.get_id(0)

    px = x1[i]
    py = y1[i]
    pz = z1[i]
    pw = w1[i]
    for j in range(n):
        qx = x2[j]
        qy = y2[j]
        qz = z2[j]
        qw = w2[j]
        dx = px - qx
        dy = py - qy
        dz = pz - qz
        wprod = pw * qw
        dsq = dx * dx + dy * dy + dz * dz

        if dsq <= rbins_squared[nbins - 1]:
            for k in range(nbins - 1, -1, -1):
                if dsq > rbins_squared[k]:
                    result[i, k + 1] += wprod
                    break
                if k == 0:
                    result[i, k] += wprod
                    break

    for j in range(nbins - 2, -1, -1):
        for k in range(j + 1, nbins, 1):
            result[i, k] += result[i, j]


@dpex.kernel
def count_weighted_pairs_3d_diff_agg_ker(item: kapi.Item, result, n):
    col_id = item.get_id(0)
    for i in range(1, n):
        result[0, col_id] += result[i, col_id]


def gpairs(nopt, nbins, x1, y1, z1, w1, x2, y2, z2, w2, rbins, results):
    # allocate per-work item private result vector in device global memory
    results_disjoint = dpnp.zeros_like(results, shape=(nopt, rbins.shape[0]))

    # call gpairs compute kernel
    dpex.call_kernel(
        count_weighted_pairs_3d_diff_ker,
        kapi.Range(nopt),
        nopt,
        nbins,
        x1,
        y1,
        z1,
        w1,
        x2,
        y2,
        z2,
        w2,
        rbins,
        results_disjoint,
    )

    # aggregate the results from the compute kernel
    dpex.call_kernel(
        count_weighted_pairs_3d_diff_agg_ker,
        kapi.Range(nbins),
        results_disjoint,
        nopt,
    )

    # copy to results vector
    results[:] = results_disjoint[0]


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


nopt = 524288
seed = 1234
nbins = 20
rmax = 50
rmin = 0.1
(x1, y1, z1, w1, x2, y2, z2, w2, rbins, results) = initialize(
    nopt, seed, nbins, rmax, rmin
)

gpairs(
    nopt,
    nbins,
    copy_to_func()(x1),
    copy_to_func()(y1),
    copy_to_func()(z1),
    copy_to_func()(w1),
    copy_to_func()(x2),
    copy_to_func()(y2),
    copy_to_func()(z2),
    copy_to_func()(w2),
    copy_to_func()(rbins),
    copy_to_func()(results),
)

t0 = now()
gpairs(
    nopt,
    nbins,
    copy_to_func()(x1),
    copy_to_func()(y1),
    copy_to_func()(z1),
    copy_to_func()(w1),
    copy_to_func()(x2),
    copy_to_func()(y2),
    copy_to_func()(z2),
    copy_to_func()(w2),
    copy_to_func()(rbins),
    copy_to_func()(results),
)
t1 = now()

print("TIME: {:10.6f}".format((t1 - t0)), flush=True)
