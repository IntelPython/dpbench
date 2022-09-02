# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0


def initialize(nopt, seed):
    import numpy as np
    import numpy.random as default_rng

    dtype = np.float64
    S0L = 10.0
    S0H = 50.0
    XL = 10.0
    XH = 50.0
    TL = 1.0
    TH = 2.0
    RISK_FREE = 0.1
    VOLATILITY = 0.2

    default_rng.seed(seed)
    price = default_rng.uniform(S0L, S0H, nopt).astype(dtype)
    strike = default_rng.uniform(XL, XH, nopt).astype(dtype)
    t = default_rng.uniform(TL, TH, nopt).astype(dtype)
    rate = RISK_FREE
    volatility = VOLATILITY
    call = np.zeros(nopt, dtype=np.float64)
    put = -np.ones(nopt, dtype=np.float64)

    return (nopt, price, strike, t, rate, volatility, call, put)
