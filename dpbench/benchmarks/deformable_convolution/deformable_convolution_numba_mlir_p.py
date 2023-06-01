# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
from numba import prange
from numba_mlir import njit


@njit(parallel=True, inline="always", fastmath=True)
def bilinear(input, offset_y, offset_x):
    height, width = input.shape
    start_x = int(math.floor(offset_x))
    start_x_weight = 1 - (offset_x - start_x)
    start_y = int(math.floor(offset_y))
    start_y_weight = 1 - (offset_y - start_y)

    output = 0
    if (
        offset_x >= width
        or offset_y >= height
        or offset_x <= -1
        or offset_y <= -1
    ):
        return output

    if start_x >= 0 and start_y >= 0:
        w = start_x_weight * start_y_weight
        output += w * input[start_y, start_x]

    if start_x + 1 < width and start_y >= 0:
        w = (1 - start_x_weight) * start_y_weight
        output += w * input[start_y, start_x + 1]

    if start_x >= 0 and start_y + 1 < height:
        w = start_x_weight * (1 - start_y_weight)
        output += w * input[start_y + 1, start_x]

    if start_x + 1 < width and start_y + 1 < height:
        w = (1 - start_x_weight) * (1 - start_y_weight)
        output += w * input[start_y + 1, start_x + 1]

    return output


@njit(parallel=True, fastmath=True)
def deform(
    input, offset, output, stride, pad, dilation, groups, deformable_groups
):
    k_height, k_width, _, out_height, out_width = offset.shape
    channels, _, _ = input.shape

    k_h_m = (k_height - 1) // 2
    k_w_m = (k_width - 1) // 2

    for ckhkw in prange(channels * k_height * k_width):
        for h in prange(out_height):
            for w in prange(out_width):
                c = ckhkw // (k_height * k_width)
                khkw = ckhkw % (k_height * k_width)
                kh = khkw // k_width
                kw = khkw % k_width

                offset_y = (
                    offset[kh, kw, 1, h, w]
                    + h * stride[0]
                    + (kh - k_h_m) * dilation[0]
                    - (pad[0] - k_h_m)
                )
                offset_x = (
                    offset[kh, kw, 0, h, w]
                    + w * stride[1]
                    + (kw - k_w_m) * dilation[1]
                    - (pad[1] - k_w_m)
                )

                output[c, kh, kw, h, w] = bilinear(input[c], offset_y, offset_x)


@njit(parallel=True, fastmath=True)
def deformable_convolution_b1(
    input,
    output,
    offset,
    weights,
    bias,
    tmp,
    stride,
    pad,
    dilation,
    groups,
    deformable_groups,
):
    out_channels, height, width = output.shape
    _, in_channels, k_height, k_width = weights.shape

    deform(input, offset, tmp, stride, pad, dilation, groups, deformable_groups)

    tmp = tmp.reshape((in_channels * k_height * k_width, height * width))

    _weights = weights.reshape((out_channels, in_channels * k_height * k_width))
    _output = output.reshape((out_channels, height * width))
    np.dot(_weights, tmp, _output)

    _bias = bias.reshape((out_channels, 1))
    _output[:] = _output + _bias


@njit(parallel=True)
def deformable_convolution(
    input,
    output,
    offset,
    weights,
    bias,
    tmp,
    stride_y,
    stride_x,
    pad_y,
    pad_x,
    dilation_y,
    dilation_x,
    groups,
    deformable_groups,
):
    batch, _, _, _ = input.shape
    for b in range(batch):
        deformable_convolution_b1(
            input[b],
            output[b],
            offset,
            weights,
            bias,
            tmp,
            (stride_y, stride_x),
            (pad_y, pad_x),
            (dilation_y, dilation_x),
            groups,
            deformable_groups,
        )
