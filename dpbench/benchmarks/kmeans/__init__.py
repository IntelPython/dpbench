# Copyright 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache 2.0

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
