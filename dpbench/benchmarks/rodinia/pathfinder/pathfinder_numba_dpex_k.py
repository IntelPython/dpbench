# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
"""Numba-dpex implementation for Pathfinder."""

import dpnp
import numba_dpex


@numba_dpex.func
def IN_RANGE(x, min, max):
    """Find if x is in range.

    Args:
        x: Element to be checked.
        min: Range min.
        max: Range max.

    Returns:
        true/false: If in range or not.
    """
    return (x) >= (min) and (x) <= (max)


@numba_dpex.func
def min_dpex(a, b):
    """Find the min.

    Args:
        a: first element.
        b: second element.

    Returns:
        t: min of two.
    """
    t = a if a <= b else b
    return t


@numba_dpex.kernel
def _pathfinder_kernel(
    iteration,
    gpuWall,
    gpuSrc,
    gpuResults,
    cols,
    startStep,
    border,
    block_size,
):
    """Kernel to compute the smallest element per iteration.

    Args:
        iteration: current iteration.
        gpuWall: Matrix elements excluding that row.
        gpuSrc: input row.
        gpuResults: Intermediate results.
        cols: number of columns.
        rows: number of rows.
        startStep: starting point.
        border: max/min border.
        block_size: block size for computation.
    """
    prev = numba_dpex.local.array((256), dtype=dpnp.int64)
    result = numba_dpex.local.array((256), dtype=dpnp.int64)

    bx = numba_dpex.get_group_id(2)
    tx = numba_dpex.get_local_id(2)

    small_block_cols = block_size - iteration * 2

    blkX = small_block_cols * bx - border
    blkXmax = blkX + block_size - 1

    xidx = blkX + tx

    validXmin = -blkX if blkX < 0 else 0
    validXmax = (
        block_size - 1 - (blkXmax - cols + 1)
        if blkXmax > cols - 1
        else block_size - 1
    )

    W = tx - 1
    E = tx + 1

    W = validXmin if W < validXmin else W
    E = validXmax if E > validXmax else E

    isValid = IN_RANGE(tx, validXmin, validXmax)

    if IN_RANGE(xidx, 0, cols - 1):
        prev[tx] = gpuSrc[xidx]

    numba_dpex.barrier(numba_dpex.LOCAL_MEM_FENCE)

    for i in range(iteration):
        computed = False
        if IN_RANGE(tx, i + 1, block_size - i - 2) and isValid:
            computed = True
            left = prev[W]
            up = prev[tx]
            right = prev[E]
            shortest = min_dpex(left, up)
            shortest = min_dpex(shortest, right)
            index = cols * (startStep + i) + xidx
            result[tx] = shortest + gpuWall[index]

        numba_dpex.barrier(numba_dpex.LOCAL_MEM_FENCE)
        if i == iteration - 1:
            break
        if computed:
            prev[tx] = result[tx]
        numba_dpex.barrier(numba_dpex.LOCAL_MEM_FENCE)

    if computed:
        gpuResults[xidx] = result[tx]


def pathfinder(data, rows, cols, pyramid_height, block_size, result):
    """Compute smallest distance from top row to bottom.

    Args:
         data: data matrix.
         rows: number of rows.
         cols: number of cols.
         pyramid_height: pyramid height.
         block_size: block size for parallel 2d-kernel.
         result: result of operation.
    """
    # create a temp list that hold first row of data as first element and empty numpy array as second element
    borderCols = pyramid_height
    smallBlockCol = block_size - (pyramid_height) * 2
    blockCols = int(
        cols / smallBlockCol + (0 if cols % smallBlockCol == 0 else 1)
    )
    size = rows * cols

    dimBlock = numba_dpex.Range(1, 1, block_size)
    dimGrid = numba_dpex.Range(1, 1, blockCols * block_size)

    gpuResult = dpnp.zeros((2, cols), dtype=dpnp.int64)
    gpuWall = dpnp.array((size - cols), dtype=dpnp.int64)

    gpuResult[0] = data[:cols]
    gpuWall = data[cols:]

    dimBlock = numba_dpex.Range(1, 1, block_size)
    dimGrid = numba_dpex.Range(1, 1, blockCols * block_size)

    src = 1
    dst = 0
    for t in range(0, rows - 1, pyramid_height):
        temp = src
        src = dst
        dst = temp

        iteration = min(pyramid_height, rows - t - 1)

        _pathfinder_kernel[numba_dpex.NdRange(dimGrid, dimBlock)](
            iteration,
            gpuWall,
            gpuResult[src],
            gpuResult[dst],
            cols,
            t,
            borderCols,
            block_size,
        )

    k = 0
    for i in gpuResult[dst]:
        result[k] = i
        k += 1
