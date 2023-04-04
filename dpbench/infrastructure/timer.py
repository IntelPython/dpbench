# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import gc
import time


class timer:
    """A contextmanager class to capture the timing for a section of code.

    :Example:
        .. code-block:: python
            from import dpbench.infrastructure import timer

            with timer(GCOff=True) as t:
                s = [x for x in range(10000)]

            print(t.get_elapsed_time())

    """

    def __init__(self, GCOff: bool = False) -> None:
        self.GCOff = GCOff

    def __enter__(self):
        if not self.GCOff:
            self.gcold = gc.isenabled()
            gc.disable()
        self._t = time.perf_counter_ns()
        return self

    def __exit__(self, type, value, traceback):
        self._t = time.perf_counter_ns() - self._t
        if self.gcold:
            gc.enable()

    def get_elapsed_time(self):
        return self._t
