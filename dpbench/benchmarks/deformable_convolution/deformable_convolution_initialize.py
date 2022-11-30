# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0


def initialize(batch, 
               in_channels, in_height, in_width, 
               out_channels, out_height, out_width, 
               kernel_height, kernel_width, 
               stride_y, stride_x,
               dilation_y, dilation_x,
               pad_y, pad_x,
               groups,
               deformable_groups,
               dtype,
               seed):
    import numpy as np
    import numpy.random as default_rng
    default_rng.seed(seed)

    return (
        default_rng.random((batch, in_channels, in_height, in_width)).astype(dtype),
        # np.ones((batch, in_channels, in_height, in_width)).astype(dtype),
        np.zeros((batch, out_channels, out_height, out_width)).astype(dtype),
        # np.zeros((kernel_height, kernel_width, 2, out_height, out_width)).astype(dtype),
        2*default_rng.random((kernel_height, kernel_width, 2, out_height, out_width)).astype(dtype) - 1,
        # default_rng.random((out_channels, in_channels, kernel_height, kernel_width)).astype(dtype),
        np.ones((out_channels, in_channels, kernel_height, kernel_width)).astype(dtype),
        default_rng.random(out_channels).astype(dtype),
        # np.ones((out_channels,)).astype(dtype),
        np.zeros((in_channels, kernel_height, kernel_width, out_height, out_width)).astype(dtype)
    )
