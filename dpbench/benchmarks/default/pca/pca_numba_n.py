# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import numba as nb
import numpy as np


@nb.njit(parallel=False, fastmath=True)
def mean_axis_0(data):
    tdata = data.T
    m = np.empty(tdata.shape[0])
    for i in range(tdata.shape[0]):
        sum = 0.0
        for j in range(tdata.shape[1]):
            sum += tdata[i, j]
        m[i] = sum / tdata.shape[1]
    return m


@nb.njit(parallel=False, fastmath=True)
def pca(data, dims_rescaled_data=2):
    # mean center the data (data -= data.mean(axis=0))
    data -= mean_axis_0(data)

    # calculate the covariance matrix
    v = np.cov(data, rowvar=False)

    # calculate eigenvectors & eigenvalues of the covariance matrix
    evalues, evectors = np.linalg.eigh(v)

    # sort eigenvalues and eigenvectors in decreasing order
    idx = np.argsort(evalues)[::-1]
    evectors = evectors[:, idx]
    evalues = evalues[idx]

    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evectors = evectors[:, :dims_rescaled_data]

    # carry out the transformation on the data using eigenvectors
    tdata = np.dot(evectors.T.astype(data.dtype), data.T).T

    # return the transformed data, eigenvalues, and eigenvectors
    return tdata, evalues, evectors
