# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0


def initialize(nevts, nout):
    import numpy as np

    return (
        nevts,
        nout,
        np.empty((nevts, nout, 4)),
    )
