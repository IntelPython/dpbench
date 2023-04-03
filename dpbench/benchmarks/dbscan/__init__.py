# Copyright 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache 2.0

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
