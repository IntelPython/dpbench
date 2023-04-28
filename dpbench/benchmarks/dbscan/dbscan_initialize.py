# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0


def initialize(n_samples, n_features, centers, seed, types_dict):
    from typing import NamedTuple

    import numpy as np
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import StandardScaler

    DEFAULT_EPS = 0.6
    DEFAULT_MINPTS = 20

    class DataSize(NamedTuple):
        n_samples: int
        n_features: int

    class Params(NamedTuple):
        eps: float
        minpts: int

    OPTIMAL_PARAMS = {
        DataSize(n_samples=2**8, n_features=2): Params(eps=0.173, minpts=4),
        DataSize(n_samples=2**8, n_features=3): Params(eps=0.35, minpts=6),
        DataSize(n_samples=2**8, n_features=10): Params(eps=0.8, minpts=20),
        DataSize(n_samples=2**9, n_features=2): Params(eps=0.15, minpts=4),
        DataSize(n_samples=2**9, n_features=3): Params(eps=0.1545, minpts=6),
        DataSize(n_samples=2**9, n_features=10): Params(eps=0.7, minpts=20),
        DataSize(n_samples=2**10, n_features=2): Params(eps=0.1066, minpts=4),
        DataSize(n_samples=2**10, n_features=3): Params(eps=0.26, minpts=6),
        DataSize(n_samples=2**10, n_features=10): Params(eps=0.6, minpts=20),
        DataSize(n_samples=2**11, n_features=2): Params(eps=0.095, minpts=4),
        DataSize(n_samples=2**11, n_features=3): Params(eps=0.18, minpts=6),
        DataSize(n_samples=2**11, n_features=10): Params(eps=0.6, minpts=20),
        DataSize(n_samples=2**12, n_features=2): Params(eps=0.0715, minpts=4),
        DataSize(n_samples=2**12, n_features=3): Params(eps=0.17, minpts=6),
        DataSize(n_samples=2**12, n_features=10): Params(eps=0.6, minpts=20),
        DataSize(n_samples=2**13, n_features=2): Params(eps=0.073, minpts=4),
        DataSize(n_samples=2**13, n_features=3): Params(eps=0.149, minpts=6),
        DataSize(n_samples=2**13, n_features=10): Params(eps=0.6, minpts=20),
        DataSize(n_samples=2**14, n_features=2): Params(eps=0.0695, minpts=4),
        DataSize(n_samples=2**14, n_features=3): Params(eps=0.108, minpts=6),
        DataSize(n_samples=2**14, n_features=10): Params(eps=0.6, minpts=20),
        DataSize(n_samples=2**15, n_features=2): Params(eps=0.0695, minpts=4),
        DataSize(n_samples=2**15, n_features=3): Params(eps=0.108, minpts=6),
        DataSize(n_samples=2**15, n_features=10): Params(eps=0.6, minpts=20),
        DataSize(n_samples=2**16, n_features=2): Params(eps=0.0695, minpts=4),
        DataSize(n_samples=2**16, n_features=3): Params(eps=0.108, minpts=6),
        DataSize(n_samples=2**16, n_features=10): Params(eps=0.6, minpts=20),
    }

    X, *_ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        random_state=seed,
    )
    X = StandardScaler().fit_transform(X)

    data_size = DataSize(n_samples=n_samples, n_features=n_features)
    params = OPTIMAL_PARAMS.get(
        data_size, Params(eps=DEFAULT_EPS, minpts=DEFAULT_MINPTS)
    )

    return (
        X.flatten().astype(types_dict["float"]),
        params.eps,
        params.minpts,
    )
