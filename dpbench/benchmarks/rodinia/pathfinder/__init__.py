# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Pathfinder Implementation.

This pathfinding algorithm finds the shortest path from the first row to the last.

Input
---------
rows<int_64> : Indicates the number of rows.

cols<int_64> : Indicates the number of cols.

pyramid height<int_64> : Indicates pyramid height.

block_size<int_64> : Indicates block size for parallel computation.

Output

--------

result<array<int_64>> : Indicates the minimum distance from first row to last.

Method:

The elements are fed to the kernel row-wise and the minimum distance is computed based
on the minimum weight of the neighbors above.
This is done for all rows until last and result is returned.


"""
