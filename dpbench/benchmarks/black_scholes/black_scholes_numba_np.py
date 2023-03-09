# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from math import erf

import numba as nb
import numpy as np


@nb.vectorize(nopython=True)
def _nberf(x):
    return erf(x)


@nb.njit(parallel=True, fastmath=True)
def black_scholes(nopt, price, strike, t, rate, volatility, call, put):
    mr = -rate
    sig_sig_two = volatility * volatility * 2

    P = price
    S = strike
    T = t

    a = np.log(P / S)
    b = T * mr

    z = T * sig_sig_two
    c = 0.25 * z
    y = np.true_divide(1.0, np.sqrt(z))

    w1 = (a - b + c) * y
    w2 = (a - b - c) * y

    d1 = 0.5 + 0.5 * _nberf(w1)
    d2 = 0.5 + 0.5 * _nberf(w2)

    Se = np.exp(b) * S

    r = P * d1 - Se * d2
    call[:] = r  # temporary `r` is necessary for faster `put` computation
    put[:] = r - P + Se
