# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from math import erf, exp, log, sqrt

from numba import prange
from numba_dpex import dpjit


@dpjit
def black_scholes(nopt, price, strike, t, rate, volatility, call, put):
    dtype = price.dtype
    mr = -rate
    sig_sig_two = volatility * volatility * dtype.type(2)

    # TODO: get rid of it once prange supports dtype
    # https://github.com/IntelPython/numba-dpex/issues/1063
    float025 = dtype.type(0.25)
    float1 = dtype.type(1.0)
    float05 = dtype.type(0.5)

    for i in prange(nopt):
        P = price[i]
        S = strike[i]
        T = t[i]

        a = log(P / S)
        b = T * mr

        z = T * sig_sig_two
        c = float025 * z
        y = float1 / sqrt(z)

        w1 = (a - b + c) * y
        w2 = (a - b - c) * y

        d1 = float05 + float05 * erf(w1)
        d2 = float05 + float05 * erf(w2)

        Se = exp(b) * S

        r = P * d1 - Se * d2
        call[i] = r
        put[i] = r - P + Se
