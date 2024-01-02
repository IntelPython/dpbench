# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from math import ceil, erf, exp, log, sqrt

from numba import cuda


@cuda.jit
def _black_scholes_kernel(nopt, price, strike, t, rate, volatility, call, put):
    dtype = price.dtype
    mr = -rate
    sig_sig_two = volatility * volatility * dtype.type(2)

    i = cuda.grid(1)

    P = price[i]
    S = strike[i]
    T = t[i]

    a = log(P / S)
    b = T * mr

    z = T * sig_sig_two
    c = dtype.type(0.25) * z
    y = dtype.type(1.0) / sqrt(z)

    w1 = (a - b + c) * y
    w2 = (a - b - c) * y

    d1 = dtype.type(0.5) + dtype.type(0.5) * erf(w1)
    d2 = dtype.type(0.5) + dtype.type(0.5) * erf(w2)

    Se = exp(b) * S

    r = P * d1 - Se * d2
    call[i] = r
    put[i] = r - P + Se


def black_scholes(nopt, price, strike, t, rate, volatility, call, put):
    nthreads = 256
    nblocks = ceil(nopt // nthreads)

    _black_scholes_kernel[nblocks, nthreads](
        nopt, price, strike, t, rate, volatility, call, put
    )
