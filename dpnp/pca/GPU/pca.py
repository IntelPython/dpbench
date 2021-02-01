# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import dpctl
import base_pca
import dpnp as np

def pca_impl(data):
    tdata = data.T
    m = np.empty(tdata.shape[0])
    # for i in range(tdata.shape[0]):
    #     m[i] = np.mean(tdata[i])
    m = np.mean(tdata,axis=1)
    c = data - m
    v = np.cov(c.T)

    values, vectors = np.linalg.eig(v)

    a = vectors.T
    b = c.T

    arr = np.dot(a, b)

    return arr.T

def pca_dpctl(data):
    with dpctl.device_context("opencl:gpu"):
        pca_impl(data)

base_pca.run("Numba", pca_dpctl)
