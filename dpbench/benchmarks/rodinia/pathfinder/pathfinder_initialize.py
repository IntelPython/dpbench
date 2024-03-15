# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
"""Pathfinder initialization."""
LOW = 0
HIGH = 10.0
SEED = 9


def initialize(rows, cols, types_dict=None):
    """Initialize the input and output matrices for pathfinder.

    Args:
        rows: number of rows.
        cols: number of cols.
        types_dict: data type of operand.

    Returns:
        data: input matrix.
        result: result matrix.
    """
    import numpy as np
    import numpy.random as rnd

    rnd.seed(SEED)

    return (
        rnd.randint(LOW, HIGH, (rows * cols), dtype=np.int64),
        np.empty(cols, dtype=np.int64),
    )
