# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
K-means is a clustering algorithm that partitions observations
from a dataset into a requested number of geometric clusters of
points closest to the cluster's own center of mass.
Using an initial estimate of the centroids, the algorithm
iteratively updates the positions of the centroids until a
fixed point.

Input
---------
arrayP: float
    input set of points
arrayPcluster: int
    cluster where points belongs
arrayC: float
    array of centriods
arrayCsum: float
    array of centriods sum
arrayCnumpoint: int
    number of points in cluster
niters: int
    number of iterations
npoints: int
    number of points
ndims: int
    number of dims
ncentroids: int
    number of centroids

Output
-------
clusters: int
    number of clusters

"""

from .kmeans_initialize import initialize
from .kmeans_numba_dpex_k import kmeans as kmeans_numba_dpex_k
from .kmeans_numba_dpex_p import kmeans as kmeans_numba_dpex_p
from .kmeans_numba_n import kmeans as kmeans_numba_n
from .kmeans_numba_npr import kmeans as kmeans_numba_npr
from .kmeans_python import kmeans as kmeans_python
from .kmeans_sycl_native_ext import kmeans_sycl

__all__ = [
    "initialize",
    "kmeans_numba_dpex_k",
    "kmeans_numba_dpex_p",
    "kmeans_numba_n",
    "kmeans_numba_npr",
    "kmeans_python",
    "kmeans_sycl",
]
