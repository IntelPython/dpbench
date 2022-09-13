# Copyright 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache 2.0

from math import erf, exp, log, sqrt


def invsqrt(x):
    return 1.0 / sqrt(x)


def black_scholes(nopt, price, strike, t, rate, vol, call, put):
    """Documentation for black_scholes function

    The Black-Scholes program computes the price of a portfolio of 
    options using partial differential equations.
    This function is an implementation of Black-Scholes in pure Python.
    """
    
    mr = -rate
    sig_sig_two = vol * vol * 2

    for i in range(nopt):
        P = float(price[i])
        S = strike[i]
        T = t[i]

        a = log(P / S)
        b = T * mr

        z = T * sig_sig_two
        c = 0.25 * z
        y = invsqrt(z)

        w1 = (a - b + c) * y
        w2 = (a - b - c) * y

        d1 = 0.5 + 0.5 * erf(w1)
        d2 = 0.5 + 0.5 * erf(w2)

        Se = exp(b) * S

        call[i] = P * d1 - Se * d2
        put[i] = call[i] - P + Se
