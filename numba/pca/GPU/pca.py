# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT


import base_pca
import numba
import numpy as np
import dpctl

@numba.njit(parallel=True, fastmath=True)
def pca_impl(data):
    tdata = data.T
    m = np.empty(tdata.shape[0])
    for i in numba.prange(tdata.shape[0]):
        m[i] = np.mean(tdata[i])
    c = data - m
    v = np.cov(c.T)

    values, vectors = np.linalg.eig(v)

    a = vectors.T
    b = c.T

    arr = np.dot(a, b)

    return arr.T

def call_pca(data):
    with dpctl.device_context("opencl:gpu"):
        pca_impl(data)
        
base_pca.run("Numba", pca_impl)
