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

import dace as dc
import numpy as np

N, bins, npt = (dc.symbol(s, dtype=dc.int64) for s in ("N", "bins", "npt"))


@dc.program
def get_bin_edges(a: dc.float64[N], bin_edges: dc.float64[bins + 1]):
    a_min = np.amin(a)
    a_max = np.amax(a)
    delta = (a_max - a_min) / bins
    for i in dc.map[0:bins]:
        bin_edges[i] = a_min + i * delta

    bin_edges[bins] = a_max  # Avoid roundoff error on last point


@dc.program
def compute_bin(x: dc.float64, bin_edges: dc.float64[bins + 1]):
    # assuming uniform bins for now
    a_min = bin_edges[0]
    a_max = bin_edges[bins]
    return dc.int64(bins * (x - a_min) / (a_max - a_min))


@dc.program
def histogram(a: dc.float64[N], bin_edges: dc.float64[bins + 1]):
    hist = np.ndarray((bins,), dtype=np.int64)
    hist[:] = 0
    get_bin_edges(a, bin_edges)

    for i in dc.map[0:N]:
        bin = min(compute_bin(a[i], bin_edges), bins - 1)
        hist[bin] += 1

    return hist


@dc.program
def histogram_weights(
    a: dc.float64[N], bin_edges: dc.float64[bins + 1], weights: dc.float64[N]
):
    hist = np.ndarray((bins,), dtype=weights.dtype)
    hist[:] = 0
    get_bin_edges(a, bin_edges)

    for i in dc.map[0:N]:
        bin = min(compute_bin(a[i], bin_edges), bins - 1)
        hist[bin] += weights[i]

    return hist


@dc.program
def azimint_hist(data: dc.float64[N], radius: dc.float64[N]):
    # histu = np.histogram(radius, npt)[0]
    bin_edges_u = np.ndarray((npt + 1,), dtype=np.float64)
    histu = histogram(radius, bin_edges_u)
    # histw = np.histogram(radius, npt, weights=data)[0]
    bin_edges_w = np.ndarray((npt + 1,), dtype=np.float64)
    histw = histogram_weights(radius, bin_edges_w, data)
    return histw / histu
