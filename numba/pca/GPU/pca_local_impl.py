# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import dpctl
import base_pca
import numba
import numpy as np

@numba.njit(parallel=True, fastmath=True)
def matmul(X,Y):
    result = np.zeros((X.shape[0],Y.shape[1]))
    for i in numba.prange(X.shape[0]):
        for j in range(Y.shape[1]):
            for k in range(Y.shape[0]):
                result[i,j] += X[i,k] * Y[k,j]
    return result

@numba.njit(parallel=True, fastmath=True)
def compute_mean_axis_0(data):
    tdata = data.T
    m = np.empty(tdata.shape[0])
    for i in numba.prange(tdata.shape[0]):
        sum = 0.0
        for j in range(tdata.shape[1]):
            sum += tdata[i,j]
        m[i] = sum/tdata.shape[1]
    return m

def covariance(M):
    with dpctl.device_context(base_pca.get_device_selector()):
        mean = compute_mean_axis_0(M.T)
        X = M-mean[:, None]
        Y = (M-mean[:, None]).T
        res = matmul(X,Y)
        return res/(M.shape[1]-1)

def pca_impl(data):
    with dpctl.device_context(base_pca.get_device_selector()):
        m = compute_mean_axis_0(data)
        c = data - m
        v = covariance(c.T)
        values, vectors = np.linalg.eig(v)
        a = vectors.T
        b = c.T
        arr = matmul(a,b)
        return arr.T

base_pca.run("Numba", pca_impl)
