# SPDX-FileCopyrightText: 2010-2016 Ohio State University
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import dpnp as np
from numba_dpex import dpjit


@dpjit
def kernel(A):
    A[:] = np.linalg.cholesky(A) + np.triu(A, k=1)
