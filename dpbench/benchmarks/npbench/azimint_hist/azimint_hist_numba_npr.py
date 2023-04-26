# SPDX-FileCopyrightText: 2014 Jérôme Kieffer et al.
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Jérôme Kieffer and Giannis Ashiotis. Pyfai: a python library for
high performance azimuthal integration on gpu, 2014. In Proceedings of the
7th European Conference on Python in Science (EuroSciPy 2014).
"""

import numba as nb
import numpy as np


@nb.jit(nopython=True, parallel=True, fastmath=True)
def get_bin_edges_prange(a, bins):
    bin_edges = np.zeros((bins + 1,), dtype=np.float64)
    a_min = a.min()
    a_max = a.max()
    delta = (a_max - a_min) / bins
    for i in nb.prange(bin_edges.shape[0]):
        bin_edges[i] = a_min + i * delta

    bin_edges[-1] = a_max  # Avoid roundoff error on last point
    return bin_edges


@nb.jit(nopython=True, fastmath=True)
def compute_bin(x, bin_edges):
    # assuming uniform bins for now
    n = bin_edges.shape[0] - 1
    a_min = bin_edges[0]
    a_max = bin_edges[-1]

    # special case to mirror NumPy behavior for last bin
    if x == a_max:
        return n - 1  # a_max always in last bin

    return int(n * (x - a_min) / (a_max - a_min))


@nb.jit(nopython=True, parallel=True, fastmath=True)
def histogram_prange(a, bins, weights):
    hist = np.zeros((bins,), dtype=a.dtype)
    bin_edges = get_bin_edges_prange(a, bins)

    for i in nb.prange(a.shape[0]):
        bin = compute_bin(a[i], bin_edges)
        hist[bin] += weights[i]

    return hist, bin_edges


@nb.jit(nopython=True, parallel=True, fastmath=True)
def azimint_hist(data, radius, npt):
    histu = np.histogram(radius, npt)[0]
    # histw = np.histogram(radius, npt, weights=data)[0]
    histw = histogram_prange(radius, npt, weights=data)[0]
    return histw / histu
