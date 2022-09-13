# Copyright 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache 2.0


from math import erf

import numba as nb
from numpy import exp, log, sqrt


@nb.vectorize(nopython=True)
def _nberf(x):
    return erf(x)


@nb.njit(parallel=True, fastmath=True)
def black_scholes(nopt, price, strike, t, rate, vol, call, put):
    """Documentation for black_scholes function

    The Black-Scholes program computes the price of a portfolio of 
    options using partial differential equations.
    The entire computation performed by Black-Scholes is data-parallel 
    where each option can be priced independent of other options.
    This function is an implementation of Black-Scholes in Python 
    using numpy vector operations.
    It is jit-compiled using numba and executes in parallel.
    """
    
    mr = -rate
    sig_sig_two = vol * vol * 2

    P = price
    S = strike
    T = t

    a = log(P / S)
    b = T * mr

    z = T * sig_sig_two
    c = 0.25 * z
    y = 1.0 / sqrt(z)

    w1 = (a - b + c) * y
    w2 = (a - b - c) * y

    d1 = 0.5 + 0.5 * _nberf(w1)
    d2 = 0.5 + 0.5 * _nberf(w2)

    Se = exp(b) * S

    r = P * d1 - Se * d2
    call[:] = r  # temporary `r` is necessary for faster `put` computation
    put[:] = r - P + Se
