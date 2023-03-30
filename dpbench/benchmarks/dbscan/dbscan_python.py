# Copyright 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache 2.0

import numpy as np
from sklearn.cluster import DBSCAN


def dbscan(n_samples, n_features, data, eps, min_pts, assignments):
    data = np.reshape(data, (n_samples, n_features))

    labels = DBSCAN(eps=eps, min_samples=min_pts).fit_predict(data)
    np.copyto(assignments, labels)
    return len(set(assignments)) - (1 if -1 in labels else 0)
