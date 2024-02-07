# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from math import erf, exp, log, sqrt

import numba_mlir.kernel as nb


@nb.kernel(gpu_fp64_truncate="auto")
def _black_scholes_kernel(nopt, price, strike, t, rate, volatility, call, put):
    mr = -rate
    sig_sig_two = volatility * volatility * 2

    i = nb.get_global_id(0)

    P = price[i]
    S = strike[i]
    T = t[i]

    a = log(P / S)
    b = T * mr

    z = T * sig_sig_two
    c = 0.25 * z
    y = 1.0 / sqrt(z)

    w1 = (a - b + c) * y
    w2 = (a - b - c) * y

    d1 = 0.5 + 0.5 * erf(w1)
    d2 = 0.5 + 0.5 * erf(w2)

    Se = exp(b) * S

    r = P * d1 - Se * d2
    call[i] = r
    put[i] = r - P + Se


def black_scholes(nopt, price, strike, t, rate, volatility, call, put):
    _black_scholes_kernel[nopt, nb.DEFAULT_LOCAL_SIZE](
        nopt, price, strike, t, rate, volatility, call, put
    )