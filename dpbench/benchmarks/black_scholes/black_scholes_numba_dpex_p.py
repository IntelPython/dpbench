# Copyright 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache 2.0


from math import erf, exp, log, sqrt

import numba


# blackscholes implemented as a parallel loop using numba.prange
@numba.njit(parallel=True, fastmath=True)
def black_scholes_kernel(nopt, price, strike, t, rate, vol, call, put):
    """
    Blackscholes implementation using numba paralle-for loops
    """

    mr = -rate
    sig_sig_two = vol * vol * 2

    for i in numba.prange(nopt):
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


def black_scholes(nopt, price, strike, t, rate, vol, call, put):
    """Documentation for black_scholes function

    The Black-Scholes program computes the price of a portfolio of
    options using partial differential equations.
    The entire computation performed by Black-Scholes is data-parallel
    where each option can be priced independent of other options.
    This function is an implementation of Black-Scholes in Python.
    It is jit-compiled using numba and executes in parallel on device.
    """

    black_scholes_kernel(nopt, price, strike, t, rate, vol, call, put)
