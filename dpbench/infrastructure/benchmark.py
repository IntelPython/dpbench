# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import json
import logging
import os
import pathlib
import sqlite3
import tempfile
from collections import namedtuple
from datetime import datetime
from functools import partial
from multiprocessing import Manager, Process
from typing import Any, Dict

import numpy as np

import dpbench.config as cfg
from dpbench.infrastructure import timer

from . import timeout_decorator as tout
from .datamodel import Result, store_results
from .dpcpp_framework import DpcppFramework
from .dpnp_framework import DpnpFramework
from .enums import ErrorCodes, ValidationStatusCodes
from .framework import Framework
from .numba_dpex_framework import NumbaDpexFramework
from .numba_framework import NumbaFramework

# A global namedtuple to store a function implementing a benchmark along with
# the name of the implementation.
BenchmarkImplFn = namedtuple("BenchmarkImplFn", "name fn")


def _reset_output_args(bench_info, orig_data, fmwrk, inputs):
    output_args = bench_info.output_args
    array_args = bench_info.array_args
    for arg in inputs.keys():
        if arg in output_args and arg in array_args:
            original_data = orig_data[arg]
            inputs.update({arg: fmwrk.copy_to_func()(original_data)})


def _setup_func(initialized_data, array_args, framework):
    copied_args = {}
    for arg in array_args:
        npdata = initialized_data[arg]
        copied_args.update({arg: framework.copy_to_func()(npdata)})
    return copied_args


def _array_size(array: Any) -> int:
    try:
        return array.nbytes
    except AttributeError:
        return array.size * array.itemsize


# TODO: move definition of class on top, so we can properly linter it.
def _exec(
    bench_info,
    orig_data,
    impl_fn,
    out_filename,
    fmwrk,
    repeat,
    get_args,
    results_dict,
    copy_output,
):
    """Executes a benchmark for a given implementation.

    A helper function to execute a benchmark. The function is called in a
    separate sub-process by a BenchmarkRunner instance. The ``_exec`` function
    first runs the benchmark implementation function once as a warmup and then
    performs the specified number of repetitions. The output results are reset
    before each repetition and the final output is serialized into a npz
    (compressed NumPy data file) file.

    All timing results and the path to the serialized results are written to
    the results_dict input argument that is managed by the calling process.
    """
    input_args = bench_info.input_args
    array_args = bench_info.array_args
    inputs = dict()

    with timer.timer() as t:
        args = get_args(orig_data, array_args, fmwrk)

    results_dict["setup_time"] = t.get_elapsed_time()

    input_size = 0
    for arg in array_args:
        input_size += _array_size(orig_data[arg])

    results_dict["input_size"] = input_size

    for arg in input_args:
        if arg not in array_args:
            inputs.update({arg: orig_data[arg]})
        else:
            inputs.update({arg: args[arg]})

    # Warmup
    def warmup(impl_fn, inputs):
        fmwrk.execute(impl_fn, inputs)

    with timer.timer() as t:
        try:
            warmup(impl_fn, inputs)
        except Exception:
            logging.exception("Benchmark execution failed at the warmup step.")
            results_dict["error_state"] = ErrorCodes.FAILED_EXECUTION
            results_dict["error_msg"] = "Execution failed"
            return

    results_dict["warmup_time"] = t.get_elapsed_time()
    _reset_output_args(
        bench_info=bench_info, orig_data=orig_data, fmwrk=fmwrk, inputs=inputs
    )
    exec_times = [0] * repeat

    retval = None
    for i in range(repeat):
        with timer.timer() as t:
            retval = fmwrk.execute(impl_fn, inputs)
        exec_times[i] = t.get_elapsed_time()
        # Do not reset the output from the last repeat
        if i < repeat - 1:
            _reset_output_args(
                bench_info=bench_info,
                orig_data=orig_data,
                fmwrk=fmwrk,
                inputs=inputs,
            )

    results_dict["exec_times"] = exec_times

    # Get the output data
    if copy_output:
        out_args = bench_info.output_args
        array_args = bench_info.array_args
        output_arrays = dict()
        with timer.timer() as t:
            for out_arg in out_args:
                if out_arg in array_args:
                    output_arrays.update(
                        {out_arg: fmwrk.copy_from_func()(inputs[out_arg])}
                    )

        out_filename = tempfile.gettempdir() + "/" + out_filename
        np.savez_compressed(out_filename, **output_arrays)
        results_dict["outputs"] = out_filename + ".npz"
        results_dict["teardown_time"] = t.get_elapsed_time()

        # Special case: if the benchmark implementation returns anything, then
        # add that to the results dict
        if retval is not None:
            results_dict["return-value"] = convert_to_numpy(retval, fmwrk)

    results_dict["error_state"] = ErrorCodes.SUCCESS
    results_dict["error_msg"] = ""


def convert_to_numpy(value: any, fmwrk: Framework) -> any:
    """Calls copy_from_func on all array values."""
    if isinstance(value, tuple):
        retval_list = list(value)
        for i, _ in enumerate(retval_list):
            retval_list[i] = fmwrk.copy_from_func()(retval_list[i])
        value = tuple(retval_list)
    else:
        value = fmwrk.copy_from_func()(value)

    return value


class BenchmarkResults:
    """A helper class to store the results and timing from running a
    benchmark.
    """

    def __init__(self, bench, repeat, impl_postfix, preset):
        """Initialize defaults."""
        self._fmwrk = bench.get_framework(impl_postfix)
        self._setup_time = 0.0
        self._warmup_time = 0.0
        self._teardown_time = 0.0
        self._validation_state = ValidationStatusCodes.NA
        self._error_state = ErrorCodes.UNIMPLEMENTED
        self._error_msg = "Not implemented"
        self._results = dict()
        self._bench = bench
        self._repeats = repeat
        self._impl_postfix = impl_postfix
        self._preset = preset
        self._input_size = 0

        self.exec_times = np.zeros(repeat, np.float64)

    @property
    def benchmark(self):
        return self._bench

    @benchmark.setter
    def benchmark(self, bench):
        self._bench = bench

    @property
    def benchmark_name(self):
        """Returns the name of the benchmark.

        Returns:
            str: Name of the benchmark
        """
        return self._bench.bname

    @benchmark_name.setter
    def benchmark_name(self, bname):
        """Sets the name of the benchmark."""
        self._bench.bname = bname

    @property
    def benchmark_impl_postfix(self):
        """Returns the implementation type (postfix) of the benchmark.

        Returns:
            str: The implementation postfix for the benchmark's run
        """
        return self._impl_postfix

    @benchmark_impl_postfix.setter
    def benchmark_impl_postfix(self, impl_postfix: str):
        self._impl_postfix = impl_postfix

    @property
    def framework_name(self):
        """Returns the name of the Framework used to execute the benchmark

        Returns:
            str: The name of the Framework used for execution
        """
        if self._fmwrk:
            return self._fmwrk.fname
        else:
            return "Not available"

    @property
    def framework(self):
        return self._fmwrk

    @framework.setter
    def framework(self, fmwrk=None):
        self._fmwrk = fmwrk

    @property
    def framework_version(self):
        """Returns the version of the Framework used to execute the benchmark

        Returns:
            str: The version of the Framework used for execution
        """
        if self._fmwrk:
            return self._fmwrk.version()
        else:
            return "Not available"

    @property
    def setup_time(self):
        """Returns the time in nanoseconds used to setup the benchmark.

        Setting up a benchmark involves copying the data from NumPy to either
        other NumPy arrays or to a Framework-specific data container.

        Returns:
            float: Time in nanosecond spent on copying data from NumPy to
            Framework-specific data container.
        """
        return self._setup_time

    @setup_time.setter
    def setup_time(self, setup_time):
        self._setup_time = setup_time

    @property
    def warmup_time(self):
        return self._warmup_time

    @warmup_time.setter
    def warmup_time(self, warmup_time):
        self._warmup_time = warmup_time

    @property
    def teardown_time(self):
        """Returns the time in nanoseconds used to teardown the benchmark.

        Tearing down a benchmark involves copying the data from any
        Framework-specific data container to a NumPy array on the host system.

        Returns:
            float: Time in nanosecond spent on copying data from any
            Framework-specific data container to NumPy.
        """
        return self._teardown_time

    @teardown_time.setter
    def teardown_time(self, teardown_time):
        self._teardown_time = teardown_time

    @property
    def num_repeats(self):
        """Returns the number of repetitions of the main execution.

        Returns:
            int: Number of times the main program was executed.
        """
        return self._repeats

    @num_repeats.setter
    def num_repeats(self, repeats):
        self._repeats = repeats

    @property
    def preset(self):
        return self._preset

    @preset.setter
    def preset(self, preset):
        self._preset = preset

    @property
    def input_size(self):
        return self._input_size

    @input_size.setter
    def input_size(self, problem_size):
        self._input_size = problem_size

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
        self._exec_times = exec_times
        self._exec_time_quartiles = np.percentile(exec_times, [25, 50, 75])

    @property
    def min_exec_time(self):
        """Minimum execution time for the benchmark out of the set of
        repetitions.

        Returns:
            float: Time in nanoseconds showing the fastest run out of all
            repeats.
        """
        return self._exec_times.min()

    @property
    def max_exec_time(self):
        """Maximum execution time for the benchmark out of the set of
        repetitions.

        Returns:
            float: Time in nanoseconds showing the slowest run out of all
            repeats.
        """
        return self._exec_times.max()

    @property
    def quartile25_exec_time(self):
        """25th quartile execution time for the benchmark out of the set of
        repetitions.

        Returns:
            float: Time in nanoseconds showing the 25th quartile run out of all
            repeats.
        """
        return self._exec_time_quartiles[0]

    @property
    def median_exec_time(self):
        """Median execution time for the benchmark out of the set of
        repetitions.

        Returns:
            float: Time in nanoseconds showing the median run out of all
            repeats.
        """
        return self._exec_time_quartiles[1]

    @property
    def quartile75_exec_time(self):
        """75th quartile execution time for the benchmark out of the set of
        repetitions.

        Returns:
            float: Time in nanoseconds showing the 75th quartile run out of all
            repeats.
        """
        return self._exec_time_quartiles[2]

    @property
    def results(self):
        """Returns as a list the output data from the benchmark.

        Returns:
            list: List of the output arguments generated by the benchmark.
        """
        return self._results

    @results.setter
    def results(self, results=None):
        self._results = results

    @property
    def validation_state(self) -> ValidationStatusCodes:
        return self._validation_state

    @validation_state.setter
    def validation_state(self, validated):
        self._validation_state = validated

    @property
    def error_state(self):
        return self._error_state

    @error_state.setter
    def error_state(self, error_state=ErrorCodes.UNIMPLEMENTED):
        self._error_state = error_state

    @property
    def error_msg(self):
        return self._error_msg

    @error_msg.setter
    def error_msg(self, error_msg="Not implemented"):
        self._error_msg = error_msg

    def Result(self, run_id: int) -> Result:
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
            benchmark=self.benchmark_name,
            implementation=self.benchmark_impl_postfix,
            platform="TODO",
            framework_version=self.framework_name
            + " "
            + self.framework_version,
            error_state=error_state_str,
            problem_preset=self.preset,
            input_size=self.input_size,
            # todo: platform
            setup_time=self.setup_time,
            warmup_time=self.warmup_time,
            repeats=str(self.num_repeats),
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


# TODO: move Benchmark implementation above for proper linter.
class BenchmarkRunner:
    def __init__(
        self,
        bench: "Benchmark",
        impl_postfix,
        preset,
        repeat=10,
        timeout=200.0,
        copy_output=True,
    ):
        self.bench = bench
        self.preset = preset
        self.repeat = repeat
        self.timeout = timeout
        self.impl_fn = self.bench.get_impl(impl_postfix)
        self.fmwrk = self.bench.get_framework(impl_postfix)
        self.results = BenchmarkResults(
            self.bench, self.repeat, impl_postfix, self.preset
        )

        if not self.impl_fn:
            self.results.error_state = ErrorCodes.UNIMPLEMENTED
            self.results.error_msg = "No implementation"
        elif not self.fmwrk:
            self.results.error_state = ErrorCodes.NO_FRAMEWORK
            self.results.error_msg = "No framework"
        else:
            # Execute the benchmark
            with Manager() as manager:
                results_dict = manager.dict()
                orig_data = self.bench.get_data(preset=preset)
                impl_fn = self.bench.get_impl(impl_postfix)
                run_datetime = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
                out_filename = (
                    self.bench.bname
                    + "_"
                    + impl_postfix
                    + "_"
                    + preset
                    + "."
                    + run_datetime
                )
                p = Process(
                    target=_exec,
                    args=(
                        self.bench.info,
                        orig_data,
                        impl_fn,
                        out_filename,
                        self.fmwrk,
                        repeat,
                        _setup_func,
                        results_dict,
                        copy_output,
                    ),
                )
                p.start()
                res = p.join(timeout)
                if res is None and p.exitcode is None:
                    logging.error(
                        "Terminating process due to timeout in the execution "
                        f"phase of {self.bench.bname} "
                        f"for the {impl_postfix} implementation"
                    )
                    p.kill()
                    self.results.error_state = ErrorCodes.EXECUTION_TIMEOUT
                    self.results.error_msg = "Execution timed out"
                else:
                    self.results.error_state = results_dict.get(
                        "error_state", ErrorCodes.FAILED_EXECUTION
                    )
                    self.results.error_msg = results_dict.get(
                        "error_msg", "Unexpected crash"
                    )
                    self.results.input_size = results_dict.get("input_size")
                    if self.results.error_state == ErrorCodes.SUCCESS:
                        self.results.setup_time = results_dict["setup_time"]
                        self.results.warmup_time = results_dict["warmup_time"]
                        self.results.exec_times = np.asarray(
                            results_dict["exec_times"]
                        )

                        if "outputs" in results_dict:
                            output_npz = results_dict["outputs"]
                            if output_npz:
                                with np.load(output_npz) as npzfile:
                                    for outarr in npzfile.files:
                                        self.results.results.update(
                                            {outarr: npzfile[outarr]}
                                        )
                                os.remove(output_npz)
                            self.results.teardown_time = results_dict[
                                "teardown_time"
                            ]

                        if "return-value" in results_dict:
                            self.results.results.update(
                                {"return-value": results_dict["return-value"]}
                            )

    def get_results(self) -> BenchmarkResults:
        return self.results


class Benchmark(object):
    """A class for reading and benchmark information and initializing
    benchmark data.
    """

    def _set_implementation_fn_list(
        self,
    ) -> list[BenchmarkImplFn]:
        """Loads all implementation functions into list.

        Returns:
            A list of (name, value) pair that represents the name of an
            implementation function and a corresponding function object.
        """

        result: list[BenchmarkImplFn] = []

        for impl in self.info.implementations:
            try:
                mod = importlib.import_module(impl.package_path)
                canonical_name = f"{self.info.module_name}_{impl.postfix}"
                implfn = BenchmarkImplFn(
                    name=canonical_name, fn=getattr(mod, impl.func_name)
                )
                result.append(implfn)
            except Exception:
                logging.exception(
                    f"Failed to import benchmark module: {impl.module_name}"
                )
                continue

        return result

    def _set_reference_implementation(self) -> BenchmarkImplFn:
        """Sets the reference implementation for the benchmark.

        The reference implementation is either a pure Python implementation
        if available, or else a NumPy implementation. We give preference to
        the NumPy implementation over Python if both are present.
        If neither is found, then the reference implementation is set to None.

        Returns:
            BenchmarkImplFn: The reference benchmark implementation.

        """

        ref_impl = None

        python_impl = [
            impl for impl in self.impl_fnlist if "python" in impl.name
        ]
        numpy_impl = [impl for impl in self.impl_fnlist if "numpy" in impl.name]

        if numpy_impl:
            ref_impl = numpy_impl[0]
        elif python_impl:
            ref_impl = python_impl[0]
        else:
            raise RuntimeError("No reference implementation")

        return ref_impl

    def _set_impl_to_framework_map(self, impl_fnlist) -> dict[str, Framework]:
        """Create a dictionary mapping each implementation function name to a
        corresponding Framework object.

        Args:
            impl_fnlist : list of implementation functions

        Returns:
            Dict: Dictionary mapping implementation function to a Framework
        """

        impl_to_fw_map = dict()

        for bimpl in impl_fnlist:
            if "_numba" in bimpl.name and "_dpex" not in bimpl.name:
                impl_to_fw_map.update({bimpl.name: NumbaFramework("numba")})
            elif "_numpy" in bimpl.name:
                impl_to_fw_map.update({bimpl.name: Framework("numpy")})
            elif "_python" in bimpl.name:
                impl_to_fw_map.update({bimpl.name: Framework("python")})
            elif "_dpex" in bimpl.name:
                try:
                    fw = NumbaDpexFramework("numba_dpex")
                    impl_to_fw_map.update({bimpl.name: fw})
                except Exception:
                    logging.exception(
                        "Framework could not be "
                        + "created for numba_dpex due to:"
                    )
            elif "_sycl" in bimpl.name:
                try:
                    fw = DpcppFramework("dpcpp")
                    impl_to_fw_map.update({bimpl.name: fw})
                except Exception:
                    logging.exception(
                        "Framework could not be created for dpcpp due to:"
                    )
            elif "_dpnp" in bimpl.name:
                try:
                    fw = DpnpFramework("dpnp")
                    impl_to_fw_map.update({bimpl.name: fw})
                except Exception:
                    logging.exception(
                        "Framework could not be created for dpcpp due to:"
                    )

        return impl_to_fw_map

    def _get_validation_data(self, preset):
        if preset in self.refdata.keys():
            return self.refdata[preset]

        ref_impl_postfix = self.ref_impl_fn.name[
            (len(self.bname) - len(self.ref_impl_fn.name) + 1) :  # noqa: E203
        ]

        ref_results = BenchmarkRunner(
            bench=self,
            impl_postfix=ref_impl_postfix,
            preset=preset,
            repeat=1,
            copy_output=True,
        ).get_results()

        if ref_results.error_state == ErrorCodes.SUCCESS:
            self.refdata.update({preset: ref_results.results})
            return ref_results.results
        else:
            logging.error(
                "Validation data unavailable as reference implementation "
                + "could not be executed."
            )
            return None

    def _validate_results(self, preset, frmwrk, frmwrk_out):
        ref_out = self._get_validation_data(preset)
        if not ref_out:
            return False
        try:
            validator_fn = frmwrk.validator()
            for key in ref_out.keys():
                valid = validator_fn(ref_out[key], frmwrk_out[key])
                if not valid:
                    logging.error(
                        (
                            "Output did not match for {0}. "
                            + "Expected: {1} Actual: {2}"
                        ).format(key, ref_out[key], frmwrk_out[key])
                    )
            return valid
        except Exception:
            return False

    def __init__(
        self,
        config: cfg.Benchmark,
    ):
        """Reads benchmark information.
        :param bname: The benchmark name.
        "param config_path: Optional location of the config JSON file for the
        benchmark. If none is provided, the default config inside the
        package's bench_info directory is used.
        """

        self.bdata = dict()
        self.refdata = dict()

        self.info: cfg.Benchmark = config
        self.bname = self.info.module_name
        self.init_mod_path = (
            self.info.init.package_path if self.info.init else None
        )
        self.init_fn_name: str = (
            self.info.init.func_name if self.info.init else None
        )

        self.initialize_fn = (
            getattr(
                importlib.import_module(self.init_mod_path), self.init_fn_name
            )
            if self.info.init
            else None
        )

        self.impl_fnlist = self._set_implementation_fn_list()
        self.ref_impl_fn = self._set_reference_implementation()
        self.impl_to_fw_map = self._set_impl_to_framework_map(self.impl_fnlist)

    def get_impl_fnlist(self) -> list[BenchmarkImplFn]:
        """Returns a list of function objects each for a single implementation
        of the benchmark.

        Returns:
            list[BenchmarkImplFn]: A list of 2-tuple. The first element of
            the tuple is the string function name and the second element is
            the actual function object.
        """
        return self.impl_fnlist

    def has_impl(self, impl_postfix: str):
        if not impl_postfix:
            return False

        impls = [
            impl
            for impl in self.impl_fnlist
            if self.bname + "_" + impl_postfix == impl.name
        ]
        if len(impls) == 1:
            return True
        else:
            return False

    def get_impl(self, impl_postfix: str):
        if not impl_postfix:
            return None

        fn = [
            impl.fn
            for impl in self.impl_fnlist
            if self.bname + "_" + impl_postfix == impl.name
        ]
        if len(fn) > 1:
            logging.error(
                "Unable to select any implementation as there are "
                + "multiple implementations for "
                + impl_postfix
            )
            return None
        elif not fn:
            logging.error(
                "No implementation exists for postfix: " + impl_postfix
            )
            return None
        else:
            return fn[0]

    def get_framework(self, impl_postfix: str) -> Framework:
        try:
            return self.impl_to_fw_map[self.bname + "_" + impl_postfix]
        except KeyError:
            logging.exception(
                "No framework found for the implementation "
                + self.bname
                + "_"
                + impl_postfix
            )
            return None

    def get_data(self, preset: str = "L") -> Dict[str, Any]:
        """Initializes the benchmark data.
        :param preset: The data-size preset (S, M, L, paper).
        """

        # 0. Skip if preset is already loaded
        if preset in self.bdata.keys():
            return self.bdata[preset]

        # 1. Create data dictionary
        data = dict()

        # 2. Check if the provided preset configuration is available in the
        #    config file.
        if preset not in self.info.parameters.keys():
            raise NotImplementedError(
                "{b} doesn't have a {p} preset.".format(b=self.bname, p=preset)
            )

        # 3. Store the input preset args in the "data" dict.
        parameters = self.info.parameters[preset]
        for k, v in parameters.items():
            data[k] = v

        if self.info.init:
            # 4. Call the initialize_fn with the input args and store the results
            #    in the "data" dict.

            init_input_args_list = self.info.init.input_args
            init_input_args_val_list = []
            for arg in init_input_args_list:
                init_input_args_val_list.append(data[arg])

            init_kws = dict(zip(init_input_args_list, init_input_args_val_list))
            initialized_output = self.initialize_fn(**init_kws)

            # 5. Store the initialized output in the "data" dict. Note that the
            #    implementation depends on Python dicts being ordered. Thus, the
            #    code will not work with Python older than 3.7.
            if isinstance(initialized_output, tuple):
                for idx, out in enumerate(self.info.init.output_args):
                    data.update({out: initialized_output[idx]})
            elif len(self.info.init.output_args) == 1:
                out = self.info.init.output_args[0]
                data.update({out: initialized_output})
            else:
                raise ValueError("Unsupported initialize output")

        # 6. Update the benchmark data (self.bdata) with the generated data
        #    for the provided preset.
        self.bdata[preset] = data
        return self.bdata[preset]

    def run(
        self,
        conn: sqlite3.Connection = None,
        implementation_postfix: str = None,
        preset: str = "S",
        repeat: int = 10,
        validate: bool = True,
        timeout: float = 200.0,
        run_id: int = None,
    ) -> list[BenchmarkResults]:
        results: list[BenchmarkResults] = []

        implementation_postfixes = []

        if implementation_postfix:
            implementation_postfixes.append(implementation_postfix)
        else:
            for impl in self.impl_fnlist:
                impl_postfix = impl.name[
                    (len(self.bname) - len(impl.name) + 1) :  # noqa: E203
                ]

                implementation_postfixes.append(impl_postfix)

        # TODO: do we call ref benchmark function twice?
        for implementation_postfix in implementation_postfixes:
            # copy_output is true only if validation is needed.
            runner = BenchmarkRunner(
                bench=self,
                impl_postfix=implementation_postfix,
                preset=preset,
                repeat=repeat,
                timeout=timeout,
                copy_output=validate,
            )
            result = runner.get_results()
            if validate and result.error_state == ErrorCodes.SUCCESS:
                if self._validate_results(
                    preset, result.framework, result.results
                ):
                    result.validation_state = ValidationStatusCodes.SUCCESS
                else:
                    result.validation_state = ValidationStatusCodes.FAILURE
                    result.error_state = ErrorCodes.FAILED_VALIDATION
                    result.error_msg = "Validation failed"
            if conn:
                store_results(conn, result.Result(run_id=run_id))
            results.append(result)

        return results
