# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import dace as dc
import numpy as np

NA, NB, Nkz, NE, Nqz, Nw, Norb, N3D = (
    dc.symbol(s, dc.int64)
    for s in ("NA", "NB", "Nkz", "NE", "Nqz", "Nw", "Norb", "N3D")
)


@dc.program
def scattering_self_energies(
    neigh_idx: dc.int32[NA, NB],
    dH: dc.complex128[NA, NB, N3D, Norb, Norb],
    G: dc.complex128[Nkz, NE, NA, Norb, Norb],
    D: dc.complex128[Nqz, Nw, NA, NB, N3D, N3D],
    Sigma: dc.complex128[Nkz, NE, NA, Norb, Norb],
):
    for k in range(Nkz):
        for E in range(NE):
            for q in range(Nqz):
                for w in range(Nw):
                    for i in range(N3D):
                        for j in range(N3D):
                            for a in range(NA):
                                for b in range(NB):
                                    if E - w >= 0:
                                        dHG = (
                                            G[k, E - w, neigh_idx[a, b]]
                                            @ dH[a, b, i]
                                        )
                                        dHD = dH[a, b, j] * D[q, w, a, b, i, j]
                                        Sigma[k, E, a] += dHG @ dHD
