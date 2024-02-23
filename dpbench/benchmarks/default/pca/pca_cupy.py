# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import cupy as cp


def pca(data, dims_rescaled_data=2):
    # mean center the data
    data -= data.mean(axis=0)

    # calculate the covariance matrix
    v = cp.cov(data, rowvar=False, dtype=data.dtype)

    # calculate eigenvectors & eigenvalues of the covariance matrix
    evalues, evectors = cp.linalg.eigh(v)

    # sort eigenvalues and eigenvectors in decreasing order
    idx = cp.argsort(evalues)[::-1]
    evectors = evectors[:, idx]
    evalues = evalues[idx]

    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evectors = evectors[:, :dims_rescaled_data]

    # carry out the transformation on the data using eigenvectors
    tdata = cp.dot(evectors.T, data.T).T

    cp.cuda.stream.get_current_stream().synchronize()

    # return the transformed data, eigenvalues, and eigenvectors

    return tdata, evalues, evectors
