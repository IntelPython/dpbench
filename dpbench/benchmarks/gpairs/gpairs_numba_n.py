# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import numba as nb


@nb.njit(parallel=False, fastmath=True)
def gpairs(nopt, nbins, x1, y1, z1, w1, x2, y2, z2, w2, rbins, results):
    n1 = x1.shape[0]
    n2 = x2.shape[0]
    nbins = rbins.shape[0]

    for i in range(n1):
        px = x1[i]
        py = y1[i]
        pz = z1[i]
        pw = w1[i]
        for j in range(n2):
            qx = x2[j]
            qy = y2[j]
            qz = z2[j]
            qw = w2[j]
            dx = px - qx
            dy = py - qy
            dz = pz - qz
            wprod = pw * qw
            dsq = dx * dx + dy * dy + dz * dz

            k = nbins - 1
            while dsq <= rbins[k]:
                results[k] += wprod
                k = k - 1
                if k < 0:
                    break
