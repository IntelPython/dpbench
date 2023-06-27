# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0


def initialize(
    batch,
    in_chw,
    out_chw,
    kernel_hw,
    stride_hw,
    dilation_hw,
    pad_hw,
    groups,
    deformable_groups,
    seed,
    types_dict,
):
    import numpy as np
    import numpy.random as default_rng

    dtype: np.dtype = types_dict["float"]

    default_rng.seed(seed)

    input_size = [batch] + in_chw  # nchw
    output_size = [batch] + out_chw  # nchw
    offset_size = kernel_hw + [2, out_chw[1], out_chw[2]]  # kh, kw, 2, oh, ow
    weights_size = [out_chw[0], in_chw[0]] + kernel_hw  # oc, ic, kh, kw
    bias_size = out_chw[0]  # oc
    tmp_size = [
        in_chw[0],
        kernel_hw[0],
        kernel_hw[1],
        out_chw[1],
        out_chw[2],
    ]  # ic, kh, kw, oh, ow

    input = default_rng.random(input_size).astype(dtype)
    output = np.empty(output_size, dtype=dtype)
    offset = 2 * default_rng.random(offset_size).astype("float32") - 1
    weights = default_rng.random(weights_size).astype(dtype)
    bias = default_rng.random(bias_size).astype(dtype)
    tmp = np.empty(tmp_size, dtype=dtype)

    return (
        input,
        output,
        offset,
        weights,
        bias,
        tmp,
    )
