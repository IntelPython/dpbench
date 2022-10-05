# Copyright 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache 2.0

from .black_scholes_numba_np import black_scholes as bs_np


def black_scholes(nopt, price, strike, t, rate, volatility, call, put):
    bs_np(nopt, price, strike, t, rate, volatility, call, put)
