# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from scipy.special import erf

invsqrt = lambda x: np.true_divide(1.0, np.sqrt(x))


def black_scholes(nopt, price, strike, t, rate, volatility, call, put):
    """Documentation for black_scholes function

    The Black-Scholes program computes the price of a portfolio of
    options using partial differential equations.
    This function is an implementation of Black-Scholes in pure Python using numpy vector operations.
    """

    mr = -rate
    sig_sig_two = volatility * volatility * 2

    P = price
    S = strike
    T = t

    a = np.log(P / S)
    b = T * mr

    z = T * sig_sig_two
    c = 0.25 * z
    y = invsqrt(z)

    w1 = (a - b + c) * y
    w2 = (a - b - c) * y

    d1 = 0.5 + 0.5 * erf(w1)
    d2 = 0.5 + 0.5 * erf(w2)

    Se = np.exp(b) * S

    call[:] = P * d1 - Se * d2
    put[:] = call - P + Se
