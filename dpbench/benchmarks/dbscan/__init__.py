# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
DBSCAN is a data clustering technique that that uses a
density-based technique to compute clusters.

Input
---------
n_samples: int
    number of samples
n_features: int
    number of features in each sample
data: float
    input data
eps: float
    The maximum distance between two samples for
    one to be considered as in the neighborhood of the other.
min_pts: int
    The number of samples (or total weight) in a neighborhood
    for a point to be considered as a core point.
    This includes the point itself.
assignments: float
    Random set of input assignments.

Output
-------
clusters: int
    number of clusters

"""

from .dbscan_initialize import initialize
from .dbscan_numba_dpex_k import dbscan as dbscan_numba_dpex_k
from .dbscan_numba_dpex_p import dbscan as dbscan_numba_dpex_p
from .dbscan_numba_n import dbscan as dbscan_numba_n
from .dbscan_numba_npr import dbscan as dbscan_numba_npr
from .dbscan_python import dbscan as dbscan_python
from .dbscan_sycl_native_ext import dbscan_sycl

__all__ = [
    "initialize",
    "dbscan_numba_dpex_k",
    "dbscan_numba_dpex_p",
    "dbscan_numba_n",
    "dbscan_numba_npr",
    "dbscan_python",
    "dbscan_sycl",
]
