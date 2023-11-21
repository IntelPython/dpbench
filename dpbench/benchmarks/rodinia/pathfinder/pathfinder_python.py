# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
"""Pathfinder python serial implementation."""


def _pathfinder_kernel(device_src, device_dest, cols, current_element):
    """Compute shortest distance per column element.

    Args:
         device_src: src row.
         device_dest: destination row.
         cols: number of cols.
         current_element: Current column element.
    """
    left_ind = current_element - 1 if current_element >= 1 else 0
    right_ind = current_element + 1 if current_element < cols - 1 else cols - 1
    up_ind = current_element

    left = device_src[left_ind]
    up = device_src[up_ind]
    right = device_src[right_ind]
    shortest = min(left, up, right)

    device_dest[current_element] += shortest


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
    device_dest = data[:cols]  # first row
    device_src = [0] * cols

    t = 1
    while t < rows:
        device_src = device_dest
        device_dest = data[t * cols : (t + 1) * cols]

        for i in range(cols):
            _pathfinder_kernel(device_src, device_dest, cols, i)
        t += 1

    for i in range(cols):
        result[i] = device_dest[i]
