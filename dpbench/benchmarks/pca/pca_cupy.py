# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import cupy as np


def pca(data, dims_rescaled_data=2):
    # mean center the data
    data -= data.mean(axis=0)

    # calculate the covariance matrix
    v = np.cov(data, rowvar=False, dtype=data.dtype)

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
    tdata = np.dot(evectors.T, data.T).T

    # return the transformed data, eigenvalues, and eigenvectors
    return tdata, evalues, evectors
