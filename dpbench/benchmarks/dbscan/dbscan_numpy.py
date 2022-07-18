# Copyright 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache 2.0


def dbscan(n, dim, data, eps, min_pts, assignments):
    import numpy as np
    from sklearn.cluster import DBSCAN

    data = np.reshape(data, (n, dim))

    labels = DBSCAN(eps=eps, min_samples=min_pts).fit_predict(data)
    np.copyto(assignments, labels)
    return len(set(assignments)) - (1 if -1 in labels else 0)
