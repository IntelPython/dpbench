# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import dace as dc
import numpy as np

C_in, C_out, H, K, N, W = (
    dc.symbol(s, dc.int64) for s in ("C_in", "C_out", "H", "K", "N", "W")
)


# Deep learning convolutional operator (stride = 1)
@dc.program
def conv2d(
    input: dc.float32[N, H, W, C_in], weights: dc.float32[K, K, C_in, C_out]
):
    # K = weights.shape[0]  # Assuming square kernel
    # N = input.shape[0]
    # H_out = input.shape[1] - K + 1
    # W_out = input.shape[2] - K + 1
    # C_out = weights.shape[3]
    # output = np.empty((N, H_out, W_out, C_out), dtype=np.float32)
    output = np.ndarray((N, H - K + 1, W - K + 1, C_out), dtype=np.float32)

    # Loop structure adapted from https://github.com/SkalskiP/ILearnDeepLearning.py/blob/ba0b5ba589d4e656141995e8d1a06d44db6ce58d/01_mysteries_of_neural_networks/06_numpy_convolutional_neural_net/src/layers/convolutional.py#L88
    # for i, j in dc.map[0:H-K+1, 0:W-K+1]:
    for i in range(H - K + 1):
        for j in range(W - K + 1):
            output[:, i, j, :] = np.sum(
                input[:, i : i + K, j : j + K, :, np.newaxis]
                * weights[np.newaxis, :, :, :],
                axis=(1, 2, 3),
            )

    return output


@dc.program
def conv2d_bias(
    input: dc.float32[N, H, W, C_in],
    weights: dc.float32[K, K, C_in, C_out],
    bias: dc.float32[C_out],
):
    return conv2d(input, weights) + bias
