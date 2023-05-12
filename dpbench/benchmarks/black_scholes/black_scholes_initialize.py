# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0


def initialize(nopt, seed, types_dict):
    import numpy as np
    import numpy.random as default_rng

    dtype: np.dtype = types_dict["float"]
    S0L = dtype.type(10.0)
    S0H = dtype.type(50.0)
    XL = dtype.type(10.0)
    XH = dtype.type(50.0)
    TL = dtype.type(1.0)
    TH = dtype.type(2.0)
    RISK_FREE = dtype.type(0.1)
    VOLATILITY = dtype.type(0.2)

    default_rng.seed(seed)
    price = default_rng.uniform(S0L, S0H, nopt).astype(dtype)
    strike = default_rng.uniform(XL, XH, nopt).astype(dtype)
    t = default_rng.uniform(TL, TH, nopt).astype(dtype)
    rate = RISK_FREE
    volatility = VOLATILITY
    call = np.zeros(nopt, dtype=dtype)
    put = -np.ones(nopt, dtype=dtype)

    return (price, strike, t, rate, volatility, call, put)
