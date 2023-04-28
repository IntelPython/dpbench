# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def _generate_rbins(dtype, nbins, rmax, rmin):
    rbins = np.logspace(np.log10(rmin), np.log10(rmax), nbins).astype(dtype)

    return (rbins**2).astype(dtype)


def initialize(nopt, seed, nbins, rmax, rmin, types_dict):
    import numpy.random as default_rng

    default_rng.seed(seed)
    dtype = types_dict["float"]
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
