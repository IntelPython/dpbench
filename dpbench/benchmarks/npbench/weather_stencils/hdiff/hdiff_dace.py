# SPDX-FileCopyrightText: 2014-2021 ETH Zurich
# SPDX-FileCopyrightText: 2007 Free Software Foundation, Inc. <https://fsf.org/>
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: GPL-3.0-or-later

import dace as dc
import numpy as np

I, J, K = (dc.symbol(s, dtype=dc.int64) for s in ("I", "J", "K"))


# Adapted from https://github.com/GridTools/gt4py/blob/1caca893034a18d5df1522ed251486659f846589/tests/test_integration/stencil_definitions.py#L194
@dc.program
def hdiff(
    in_field: dc.float64[I + 4, J + 4, K],
    out_field: dc.float64[I, J, K],
    coeff: dc.float64[I, J, K],
):
    # I, J, K = out_field.shape[0], out_field.shape[1], out_field.shape[2]
    lap_field = 4.0 * in_field[1 : I + 3, 1 : J + 3, :] - (
        in_field[2 : I + 4, 1 : J + 3, :]
        + in_field[0 : I + 2, 1 : J + 3, :]
        + in_field[1 : I + 3, 2 : J + 4, :]
        + in_field[1 : I + 3, 0 : J + 2, :]
    )

    # res = lap_field[1:, 1:J + 1, :] - lap_field[:-1, 1:J + 1, :]
    res1 = lap_field[1:, 1 : J + 1, :] - lap_field[: I + 1, 1 : J + 1, :]
    flx_field = np.where(
        (
            res1
            * (
                in_field[2 : I + 3, 2 : J + 2, :]
                - in_field[1 : I + 2, 2 : J + 2, :]
            )
        )
        > 0,
        0,
        res1,
    )

    # c1 = (res1 * (in_field[2:I+3, 2:J+2, :] - in_field[1:I+2, 2:J+2, :])) <= 0
    # flx_field = np.ndarray((I+1, J, K), dtype=np.float64)
    # flx_field[:] = 0.0
    # flx_field[:] = np.positive(res1, where=c1)
    # np.positive(res1, out=flx_field, where=c1)

    # res = lap_field[1:I + 1, 1:, :] - lap_field[1:I + 1, :-1, :]
    res2 = lap_field[1 : I + 1, 1:, :] - lap_field[1 : I + 1, : J + 1, :]
    fly_field = np.where(
        (
            res2
            * (
                in_field[2 : I + 2, 2 : J + 3, :]
                - in_field[2 : I + 2, 1 : J + 2, :]
            )
        )
        > 0,
        0,
        res2,
    )
    # c2 = (res2 * (in_field[2:I+2, 2:J+3, :] - in_field[2:I+2, 1:J+2, :])) <= 0
    # fly_field = np.ndarray((I, J+1, K), dtype=np.float64)
    # fly_field[:] = 0.0
    # fly_field[:] = np.positive(res2, where=c2)
    # np.positive(res2, out=fly_field, where=c2)

    out_field[:, :, :] = in_field[2 : I + 2, 2 : J + 2, :] - coeff[:, :, :] * (
        flx_field[1:, :, :]
        - flx_field[:-1, :, :]
        + fly_field[:, 1:, :]
        - fly_field[:, :-1, :]
    )
