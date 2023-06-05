# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import numpy as np

from dpbench.infrastructure.datamodel import Result
from dpbench.infrastructure.enums import ErrorCodes, ValidationStatusCodes


@dataclass
class BenchmarkResults:
    """A helper class to store the results and timing from running a
    benchmark.
    """

    repeats: int
    impl_postfix: str
    preset: str
    input_size: int = 0

    setup_time: float = 0.0
    warmup_time: float = 0.0
    teardown_time: float = 0.0

    min_exec_time: float = 0.0
    quartile25_exec_time: float = 0.0
    median_exec_time: float = 0.0
    quartile75_exec_time: float = 0.0
    max_exec_time: float = 0.0

    validation_state: ValidationStatusCodes = ValidationStatusCodes.NA
    error_state: ErrorCodes = ErrorCodes.UNIMPLEMENTED
    error_msg: str = "Not implemented"

    @property
    def exec_times(self):
        """Returns an array of execution timings measured in nanoseconds

        Returns:
            numpy.ndarray: An array of execution times for each repetition of
            the main execution.
        """
        return self._exec_times

    @exec_times.setter
    def exec_times(self, exec_times):
        if not exec_times:
            exec_times = [0.0]
        quartiles = np.percentile(exec_times, [25, 50, 75])
        self.min_exec_time = min(exec_times)
        self.quartile25_exec_time = quartiles[0]
        self.median_exec_time = quartiles[1]
        self.quartile75_exec_time = quartiles[2]
        self.max_exec_time = max(exec_times)

    def Result(self, run_id: int, benchmark_name, framework_version) -> Result:
        if self.error_state == ErrorCodes.UNIMPLEMENTED:
            error_state_str = "Unimplemented"
        elif self.error_state == ErrorCodes.NO_FRAMEWORK:
            error_state_str = "Framework unavailable"
        elif self.error_state == ErrorCodes.FAILED_EXECUTION:
            error_state_str = "Failed Execution"
        elif self.error_state == ErrorCodes.FAILED_VALIDATION:
            error_state_str = "Failed Validation"
        elif self.error_state == ErrorCodes.EXECUTION_TIMEOUT:
            error_state_str = "Execution Timeout"
        elif self.error_state == ErrorCodes.SUCCESS:
            error_state_str = "Success"
        else:
            error_state_str = "N/A"

        return Result(
            run_id=run_id,
            benchmark=benchmark_name,
            implementation=self.impl_postfix,
            # TODO: platform
            platform="TODO",
            framework_version=framework_version,
            error_state=error_state_str,
            problem_preset=self.preset,
            input_size=self.input_size,
            setup_time=self.setup_time,
            warmup_time=self.warmup_time,
            repeats=str(self.repeats),
            min_exec_time=self.min_exec_time,
            max_exec_time=self.max_exec_time,
            median_exec_time=self.median_exec_time,
            quartile25_exec_time=self.quartile25_exec_time,
            quartile75_exec_time=self.quartile75_exec_time,
            teardown_time=self.teardown_time,
            validated="Success"
            if self.validation_state == ValidationStatusCodes.SUCCESS
            else "Fail",
        )

    def _format_ns(self, time_in_ns: int):
        time = int(time_in_ns)
        assert time >= 0
        suff = [
            ("s", 1000_000_000),
            ("ms", 1000_000),
            ("\u03BCs", 1000),
            ("ns", 0),
        ]
        for s, scale in suff:
            if time >= scale:
                scaled_time = float(time) / scale if scale > 0 else time
                return f"{scaled_time}{s} ({time} ns)"

    def print(self, framework_name: str = "", framework_version: str = ""):
        print(
            "================ implementation "
            + self.impl_postfix
            + " ========================\n"
            + "implementation:",
            self.impl_postfix,
        )

        if self.error_state == ErrorCodes.SUCCESS:
            print("framework:", framework_name)
            print("framework version:", framework_version)
            print("setup time:", self._format_ns(self.setup_time))
            print("warmup time:", self._format_ns(self.warmup_time))
            print("teardown time:", self._format_ns(self.teardown_time))
            print("max execution times:", self._format_ns(self.max_exec_time))
            print("min execution times:", self._format_ns(self.min_exec_time))
            print(
                "median execution times:",
                self._format_ns(self.median_exec_time),
            )
            print("repeats:", self.repeats)
            print("preset:", self.preset)
            print("validated:", self.validation_state)
        else:
            print("error states:", self.error_state)
            print("error msg:", self.error_msg)
