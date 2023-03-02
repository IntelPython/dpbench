# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache 2.0
# SPDX-License-Identifier: Apache-2.0

"""
Principle Component Analysis

Input
---------
data : array
       random regression problem

Output
-------
data: array
      transformation on the data using eigenvectors

evalues: array
         Eigen values

evectors: array
          Eigen vectors

Method
------
PCA implementation using covariance approach.
Step 1) Calculate covariance matrix
Step 2) Compute eigen values and eigen vectors from the covariance matrix
"""

from .pca_dpnp import pca as pca_dpnp
from .pca_initialize import initialize
from .pca_numba_dpex_n import pca as pca_numba_dpex_n
from .pca_numba_n import pca as pca_numba_n
from .pca_numba_np import pca as pca_numba_np
from .pca_numpy import pca as pca_numpy

__all__ = [
    "initialize",
    "pca_numpy",
    "pca_dpnp",
    "pca_numba_n",
    "pca_numba_np",
    "pca_numba_dpex_n",
]
