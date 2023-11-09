# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
"""Numba-dpex implementation for gaussian elimination."""

import dpctl
import numba_dpex


@numba_dpex.kernel()
def gaussian_kernel_1(m, a, size, t):
    """Find the multiplier matrix.

    Args:
        m: multiplier matrix.
        a: input matrix.
        size: size of matrix.
        t: current iteration.
    """
    if (
        numba_dpex.get_local_id(2)
        + numba_dpex.get_group_id(2) * numba_dpex.get_local_size(2)
        >= size - 1 - t
    ):
        return

    m[
        size
        * (
            numba_dpex.get_local_size(2) * numba_dpex.get_group_id(2)
            + numba_dpex.get_local_id(2)
            + t
            + 1
        )
        + t
    ] = (
        a[
            size
            * (
                numba_dpex.get_local_size(2) * numba_dpex.get_group_id(2)
                + numba_dpex.get_local_id(2)
                + t
                + 1
            )
            + t
        ]
        / a[size * t + t]
    )


@numba_dpex.kernel()
def gaussian_kernel_2(m, a, b, size, t):
    """Perform Gaussian elimination using gaussian operations for a iteration.

    Args:
        m: multiplier matrix.
        a: input matrix.
        b: column matrix.
        size: size of matrices.
        t: current iteration.
    """
    if (
        numba_dpex.get_local_id(2)
        + numba_dpex.get_group_id(2) * numba_dpex.get_local_size(2)
        >= size - 1 - t
    ):
        return

    if (
        numba_dpex.get_local_id(1)
        + numba_dpex.get_group_id(1) * numba_dpex.get_local_size(1)
        >= size - t
    ):
        return

    xidx = numba_dpex.get_group_id(2) * numba_dpex.get_local_size(
        2
    ) + numba_dpex.get_local_id(2)
    yidx = numba_dpex.get_group_id(1) * numba_dpex.get_local_size(
        1
    ) + numba_dpex.get_local_id(1)

    a[size * (xidx + 1 + t) + (yidx + t)] -= (
        m[size * (xidx + 1 + t) + t] * a[size * t + (yidx + t)]
    )
    if yidx == 0:
        b[xidx + 1 + t] -= m[size * (xidx + 1 + t) + (yidx + t)] * b[t]


def gaussian(a, b, m, size, block_sizeXY, result):
    """Perform Gaussian elimination using gaussian operations.

    Args:
        a: input matrix.
        b: column matrix.
        m: multiplier matrix.
        size: size of matrices.
        block_sizeXY: grid size.
        result: result matrix.
    """
    device = dpctl.SyclDevice()
    block_size = device.max_work_group_size
    grid_size = int((size / block_size) + 0 if not (size % block_size) else 1)

    blocksize2d = block_sizeXY
    gridsize2d = int(
        (size / blocksize2d) + (0 if not (size % blocksize2d) else 1)
    )

    global_range = numba_dpex.Range(1, 1, grid_size * block_size)
    local_range = numba_dpex.Range(1, 1, block_size)

    dim_blockXY = numba_dpex.Range(1, blocksize2d, blocksize2d)
    dim_gridXY = numba_dpex.Range(
        1, gridsize2d * blocksize2d, gridsize2d * blocksize2d
    )

    for t in range(size - 1):
        gaussian_kernel_1[numba_dpex.NdRange(global_range, local_range)](
            m, a, size, t
        )

        gaussian_kernel_2[numba_dpex.NdRange(dim_gridXY, dim_blockXY)](
            m, a, b, size, t
        )

    for i in range(size):
        result[size - i - 1] = b[size - i - 1]
        for j in range(i):
            result[size - i - 1] -= (
                a[size * (size - i - 1) + (size - j - 1)] * result[size - j - 1]
            )
        result[size - i - 1] = (
            result[size - i - 1] / a[size * (size - i - 1) + (size - i - 1)]
        )
