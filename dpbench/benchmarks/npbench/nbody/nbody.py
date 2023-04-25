# SPDX-FileCopyrightText: 2020 Philip Mocz
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np


def initialize(N, tEnd, dt):
    from numpy.random import default_rng

    rng = default_rng(42)
    mass = 20.0 * np.ones((N, 1)) / N  # total mass of particles is 20
    pos = rng.random((N, 3))  # randomly selected positions and velocities
    vel = rng.random((N, 3))
    Nt = int(np.ceil(tEnd / dt))
    return mass, pos, vel, Nt
