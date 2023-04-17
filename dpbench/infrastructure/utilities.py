# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import sqlite3
import timeit
from numbers import Number
from typing import Union

import numpy as np


# From https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v: Union[str, bool]) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def time_to_ms(raw: float) -> int:
    return int(round(raw * 1000))


def relative_error(
    ref: Union[Number, np.ndarray], val: Union[Number, np.ndarray]
) -> float:
    if np.linalg.norm(ref) == 0.0:
        return 0.0
    return np.linalg.norm(ref - val) / np.linalg.norm(ref)


def validate(ref, val, framework="Unknown"):
    if not isinstance(ref, (tuple, list)):
        ref = [ref]
    if not isinstance(val, (tuple, list)):
        val = [val]
    valid = True
    for r, v in zip(ref, val):
        if not np.allclose(r, v):
            try:
                import cupy

                if isinstance(v, cupy.ndarray):
                    relerror = relative_error(r, cupy.asnumpy(v))
                else:
                    relerror = relative_error(r, v)
            except Exception:
                relerror = relative_error(r, v)
            if relerror < 1e-05:
                continue
            valid = False
            print("Relative error: {}".format(relerror))
            # return False
    if not valid:
        print("{} did not validate!".format(framework))
    return valid
