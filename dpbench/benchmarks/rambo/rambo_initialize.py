# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0


def initialize(nevts, nout):
    import numpy as np

    C1 = np.empty((nevts, nout))
    F1 = np.empty((nevts, nout))
    Q1 = np.empty((nevts, nout))

    np.random.seed(777)
    for i in range(nevts):
        for j in range(nout):
            C1[i, j] = np.random.rand()
            F1[i, j] = np.random.rand()
            Q1[i, j] = np.random.rand() * np.random.rand()

    return (
        nevts,
        nout,
        C1,
        F1,
        Q1,
        np.empty((nevts, nout, 4)),
    )
