# Copyright 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache 2.0

from .black_scholes_numba_npr import black_scholes as bs_npr


def black_scholes(nopt, price, strike, t, rate, volatility, call, put):
    bs_npr(nopt, price, strike, t, rate, volatility, call, put)
