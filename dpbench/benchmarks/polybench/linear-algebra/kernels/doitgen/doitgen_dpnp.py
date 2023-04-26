# SPDX-FileCopyrightText: 2010-2016 Ohio State University
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import dpnp as np


def kernel(NR, NQ, NP, A, C4):
    # for r in range(NR):
    #     for q in range(NQ):
    #         sum[:] = A[r, q, :] @ C4
    #         A[r, q, :] = sum
    A[:] = np.reshape(np.reshape(A, (NR, NQ, 1, NP)) @ C4, (NR, NQ, NP))
