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

from .benchmark_results import BenchmarkResults
from .datamodel import Result, store_results
from .enums import ErrorCodes, ValidationStatusCodes
from .frameworks import Framework, build_framework_map

# A global namedtuple to store a function implementing a benchmark along with
# the name of the implementation.
BenchmarkImplFn = namedtuple("BenchmarkImplFn", "name fn")


def _reset_output_args(bench, fmwrk, inputs, preset, precision):
    output_args = bench.info.output_args
    array_args = bench.info.array_args
    for arg in inputs.keys():
        if arg in output_args and arg in array_args:
            original_data = bench.get_data(
                preset=preset, framework=fmwrk, global_precision=precision
            )[arg]
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
    bench: "Benchmark",
    fmwrk,
    impl_postfix,
    preset,
    repeat,
    precision,
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

    Args:
        bench : A Benchmark object representing the benchmark to be executed.
        fmwrk : A Framework for which the benchmark is to be executed.
        impl_postfix : The identifier for the benchmark implementation.
        preset : A problem size entry defined in the bench_info JSON.
        timeout : Number of seconds after which the execution is killed.
        repeat : Number of repetitions of the benchmark execution.
        precision: The precsion to use for benchmark input data.
        args : Input arguments to benchmark implementation function.
        results_dict : A dictionary where timing and other results are stored.
        copy_output : A flag that controls copying output.
    """
    input_args = bench.info.input_args
    array_args = bench.info.array_args
    impl_fn = bench.get_impl(impl_postfix)
    inputs = dict()

    with timer.timer() as t:
        args = get_args(
            bench.get_data(
                preset=preset, framework=fmwrk, global_precision=precision
            ),
            array_args,
            fmwrk,
        )

    results_dict["setup_time"] = t.get_elapsed_time()

    input_size = 0
    for arg in array_args:
        input_size += _array_size(bench.bdata[preset][arg])

    results_dict["input_size"] = input_size

    for arg in input_args:
        if arg not in array_args:
            inputs.update(
                {
                    arg: bench.get_data(
                        preset=preset,
                        framework=fmwrk,
                        global_precision=precision,
                    )[arg]
                }
            )
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
        bench=bench,
        fmwrk=fmwrk,
        inputs=inputs,
        preset=preset,
        precision=precision,
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
                bench=bench,
                fmwrk=fmwrk,
                inputs=inputs,
                preset=preset,
                precision=precision,
            )

    results_dict["exec_times"] = exec_times

    # Get the output data
    if copy_output:
        _exec_copy_output(
            bench, fmwrk, impl_postfix, preset, retval, inputs, results_dict
        )

    results_dict["error_state"] = ErrorCodes.SUCCESS
    results_dict["error_msg"] = ""


def _exec_copy_output(
    bench: "Benchmark",
    fmwrk,
    impl_postfix,
    preset,
    retval,
    inputs: dict,
    results_dict,
):
    out_args = bench.info.output_args
    array_args = bench.info.array_args
    output_arrays = dict()
    with timer.timer() as t:
        for out_arg in out_args:
            if out_arg in array_args:
                output_arrays.update(
                    {out_arg: fmwrk.copy_from_func()(inputs[out_arg])}
                )
    run_datetime = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    out_filename = (
        bench.bname + "_" + impl_postfix + "_" + preset + "." + run_datetime
    )
    out_filename = tempfile.gettempdir() + "/" + out_filename
    np.savez_compressed(out_filename, **output_arrays)
    results_dict["outputs"] = out_filename + ".npz"
    results_dict["teardown_time"] = t.get_elapsed_time()

    # Special case: if the benchmark implementation returns anything, then
    # add that to the results dict
    if retval is not None:
        results_dict["return-value"] = convert_to_numpy(retval, fmwrk)


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


# TODO: move Benchmark implementation above for proper linter.
class BenchmarkRunner:
    def __init__(
        self,
        bench: "Benchmark",
        impl_postfix,
        preset,
        repeat=10,
        timeout=200.0,
        precision=None,
        copy_output=True,
    ):
        self.bench = bench
        self.preset = preset
        self.repeat = repeat
        self.timeout = timeout
        self.fmwrk = None
        self.results = BenchmarkResults(self.repeat, impl_postfix, self.preset)
        self.output = {}
        self.impl_fn = self.bench.get_impl(impl_postfix)
        if not self.impl_fn:
            self.results.error_state = ErrorCodes.UNIMPLEMENTED
            self.results.error_msg = "No implementation"
            return

        self.fmwrk = self.bench.get_framework(impl_postfix)
        if not self.fmwrk:
            self.results.error_state = ErrorCodes.NO_FRAMEWORK
            self.results.error_msg = "No framework"
            return
        else:
            # Execute the benchmark
            with Manager() as manager:
                results_dict = manager.dict()
                p = Process(
                    target=_exec,
                    args=(
                        self.bench,
                        self.fmwrk,
                        impl_postfix,
                        preset,
                        repeat,
                        precision,
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
                        self.results.exec_times = results_dict["exec_times"]

                        if "outputs" in results_dict:
                            output_npz = results_dict["outputs"]
                            if output_npz:
                                with np.load(output_npz) as npzfile:
                                    for outarr in npzfile.files:
                                        self.output.update(
                                            {outarr: npzfile[outarr]}
                                        )
                                os.remove(output_npz)
                            self.results.teardown_time = results_dict[
                                "teardown_time"
                            ]

                        if "return-value" in results_dict:
                            self.output.update(
                                {"return-value": results_dict["return-value"]}
                            )


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

    def _set_impl_to_framework_map(self) -> dict[str, Framework]:
        """Create a dictionary mapping each implementation function name to a
        corresponding Framework object.

        Returns:
            Dict: Dictionary mapping implementation function to a Framework
        """

        framework_map = build_framework_map()

        return {
            f"{self.info.module_name}_{impl.postfix}": framework_map[
                impl.postfix
            ]
            for impl in self.info.implementations
        }

    def _get_validation_data(self, preset, precision):
        if preset in self.refdata.keys():
            return self.refdata[preset]

        ref_impl_postfix = self.ref_impl_fn.name[
            (len(self.bname) - len(self.ref_impl_fn.name) + 1) :  # noqa: E203
        ]

        ref_runner = BenchmarkRunner(
            bench=self,
            impl_postfix=ref_impl_postfix,
            preset=preset,
            repeat=1,
            precision=precision,
            copy_output=True,
        )

        if ref_runner.results.error_state == ErrorCodes.SUCCESS:
            self.refdata.update({preset: ref_runner.output})
            return ref_runner.output
        else:
            logging.error(
                "Validation data unavailable as reference implementation "
                + "could not be executed."
            )
            return None

    def _validate_results(self, preset, frmwrk, frmwrk_out, precision):
        ref_out = self._get_validation_data(preset, precision)
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
        self.impl_to_fw_map = self._set_impl_to_framework_map()

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

    def _enforce_precision(
        self, arg_name: str, arg: Any, precision: str, framework: Framework
    ) -> Any:
        """Enforce selected precision on data.

        Args:
           arg_name: Name of data argument.
           arg: Input data argument whose precision is changed.
           precision: The precsion used for input data argument..
           framework: A Framework for which the data is initilized.

        Returns: Copy of Input data with precision updated or
                 the original input data.
        """

        if framework.is_array_type(arg):
            for _type, _precision_strings in cfg.GLOBAL.dtypes.items():
                if framework.is_type_fn()(arg, _type):
                    precision_fn = framework.change_precision_fn(arg)
                    dtype_obj = framework.dtype_obj()(
                        _precision_strings[precision]
                    )
                    return precision_fn(dtype_obj)

        logging.warning(
            "Precision unchanged for " + arg_name + " due to unsupported type."
        )
        return arg

    def _get_types_dict(
        self, framework: Framework, global_precision: str, config_precision: str
    ) -> dict[str, Any]:
        """Constructs a dictionary of types with selected precision.

        Args:
           framework: A Framework for which the data is initialized.
           global_precision: The precsion specified through runner.
           config_precision: The precision specified through config.

        Returns: Dictionary with types as str as key
                 and type object as value.
        """

        # if types_dict_name is provided, precision must be specified
        # either globally or in config file
        precision = (
            global_precision
            if global_precision is not None
            else config_precision
        )

        if precision is None or precision == "":
            raise ValueError(
                "Precision info unavailable. "
                "Types dict requires precision info either "
                "through config file or as arg to run_benchmark/s"
            )

        types_dict = dict()
        for _type, _precision_strings in cfg.GLOBAL.dtypes.items():
            try:
                types_dict[_type] = framework.dtype_obj()(
                    _precision_strings[precision]
                )
            except KeyError:
                raise KeyError(
                    "Precision "
                    + precision
                    + " not supported for "
                    + self.bname
                )

        return types_dict

    def get_data(
        self, preset: str, framework: Framework, global_precision: str
    ) -> Dict[str, Any]:
        """Initializes the benchmark data.

        Args:
           preset: The data-size preset (S, M, L).
           framework: A Framework for which the data is initialized.
           global_precision: The precision to use for benchmark data.

        Returns: Dictionary with benchmark inputs as key
                 and initialized data as value.
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
            self.get_data_init(framework, global_precision, data)

        # 8. Update the benchmark data (self.bdata) with the generated data
        #    for the provided preset.
        self.bdata[preset] = data
        return self.bdata[preset]

    def get_data_init(
        self,
        framework: Framework,
        global_precision: str,
        data: str,
    ):
        """Population benchmark data with outputs from initialization function.

        Args:
           preset: The data-size preset (S, M, L).
           framework: A Framework for which the data is initialized.
           global_precision: The precision to use for benchmark data.
           data: Dictionary to put data into.

        Returns: Dictionary with benchmark inputs as key
                 and initialized data as value.
        """
        # 4. Store types of selected precision dict in "data" dict.
        if self.info.init.types_dict_name != "":
            data[self.info.init.types_dict_name] = self._get_types_dict(
                framework, global_precision, self.info.init.precision
            )

        # 5. Call the initialize_fn with the input args and store the results
        #    in the "data" dict.
        init_input_args_list = self.info.init.input_args
        init_input_args_val_list = []
        for arg in init_input_args_list:
            init_input_args_val_list.append(data[arg])

        init_kws = dict(zip(init_input_args_list, init_input_args_val_list))
        initialized_output = self.initialize_fn(**init_kws)

        # 6. Store the initialized output in the "data" dict. Note that the
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

        # 7. If global precision or precision through config is set
        #   enforce precision on all initialized arguments.
        if self.info.init.types_dict_name == "" and (
            global_precision is not None or self.info.init.precision != ""
        ):
            enforce_pres = (
                global_precision
                if global_precision is not None
                else self.info.init.precision
            )
            for out in self.info.init.output_args:
                data.update(
                    {
                        out: self._enforce_precision(
                            out, data[out], enforce_pres, framework
                        )
                    }
                )

    def run(
        self,
        implementation_postfix: str = None,
        preset: str = "S",
        repeat: int = 10,
        validate: bool = True,
        timeout: float = 200.0,
        precision: str = None,
        conn: sqlite3.Connection = None,
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
                precision=precision,
                copy_output=validate,
            )
            result = runner.results
            if validate and result.error_state == ErrorCodes.SUCCESS:
                if self._validate_results(
                    preset, runner.fmwrk, runner.output, precision
                ):
                    result.validation_state = ValidationStatusCodes.SUCCESS
                else:
                    result.validation_state = ValidationStatusCodes.FAILURE
                    result.error_state = ErrorCodes.FAILED_VALIDATION
                    result.error_msg = "Validation failed"
            if conn:
                store_results(
                    conn,
                    result.Result(
                        run_id=run_id,
                        benchmark_name=self.bname,
                        framework_version=runner.fmwrk.fname
                        + " "
                        + runner.fmwrk.version()
                        if runner.fmwrk
                        else "n/a",
                    ),
                )
            results.append((result, runner.fmwrk))

        return results
