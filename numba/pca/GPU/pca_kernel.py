# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT


import base_pca
import dppy.core as ocldrv
import numpy
import numpy as np

import numba
from numba import dppy, jit, prange


@numba.njit(parallel={"spirv": True}, fastmath=True)
def matmul(X, Y):
    result = np.zeros((X.shape[0], Y.shape[1]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            for k in range(Y.shape[0]):
                result[i][j] += X[i][k] * Y[k][j]
    return result


@dppy.kernel
def compute_mean_axis_0(tdata, m):
    i = dppy.get_global_id(0)
    sum = 0.0
    for j in range(tdata.shape[1]):
        sum += tdata[i, j]
    m[i] = sum / tdata.shape[1]
    return m


@numba.njit(parallel={"spirv": True}, fastmath=True)
def gen_rand_data(data):
    tdata = data.T
    m = np.empty(tdata.shape[0])
    return tdata, m


def call_ocl(data):
    # device_env = dppl.runtime.get_gpu_device()
    device_env = ocldrv.runtime.get_gpu_device()

    tdata, m = gen_rand_data(data)

    dtdata = device_env.copy_array_to_device(tdata)
    dm = device_env.copy_array_to_device(m)

    # get_output_mom2[nevts,](dC1, dF1, dQ1, doutput, nout)
    compute_mean_axis_0[device_env](dtdata, dm)

    device_env.copy_array_from_device(dm)

    return m


def covariance(M):
    mean = compute_mean_axis_0(M.T)
    X = M - mean[:, None]
    Y = (M - mean[:, None]).T
    res = matmul(X, Y)
    return res / (M.shape[1] - 1)


def pca_impl(data):
    m = compute_mean_axis_0(data)
    c = data - m
    v = covariance(c.T)
    values, vectors = np.linalg.eig(v)
    a = vectors.T
    b = c.T
    arr = matmul(a, b)
    return arr


base_pca.run("Numba", pca_impl)


@jit(nopython=True, parallel={"spirv": True})
def gen_rand_data(nevts, nout):
    C1 = numpy.empty((nevts, nout))
    F1 = numpy.empty((nevts, nout))
    Q1 = numpy.empty((nevts, nout))

    numpy.random.seed(123456)
    for i in range(nevts):
        for j in range(nout):
            C1[i, j] = numpy.random.rand()
            F1[i, j] = numpy.random.rand()
            Q1[i, j] = numpy.random.rand() * numpy.random.rand()

    return C1, F1, Q1
