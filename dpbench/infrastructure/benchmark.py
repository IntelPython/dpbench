# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import logging
from typing import Any, Dict

import numpy as np

import dpbench.config as cfg
from dpbench.infrastructure import timer

from .enums import ErrorCodes
from .frameworks import Framework


def _set_input_args(
    bench: "Benchmark", framework: Framework, np_input_data: dict
):
    inputs = dict()

    for arg in bench.info.input_args:
        if arg in bench.info.array_args:
            inputs[arg] = framework.copy_to_func()(np_input_data[arg])
        else:
            inputs[arg] = np_input_data[arg]

    return inputs


def _reset_output_args(
    bench: "Benchmark", framework: Framework, inputs: dict, np_input_data: dict
):
    for arg in bench.info.output_args:
        overwritten_data = inputs.get(arg, None)
        if overwritten_data is None or arg not in bench.info.array_args:
            continue
        inputs[arg] = framework.copy_to_func()(np_input_data[arg])


def _array_size(array: Any) -> int:
    try:
        return array.nbytes
    except AttributeError:
        return array.size * array.itemsize


def _exec_simple(
    bench: "Benchmark",
    framework: Framework,
    impl_postfix: str,
    preset: str,
):
    np_input_data = bench.get_input_data(preset=preset)
    inputs = _set_input_args(bench, framework, np_input_data)
    impl_fn = bench.get_implementation(impl_postfix)

    try:
        retval = framework.execute(impl_fn, inputs)
        results_dict = {}

        _exec_copy_output(bench, framework, retval, inputs, results_dict)

        return results_dict["outputs"]
    except Exception:
        logging.exception("Benchmark execution failed at the warmup step.")
        return None


# TODO: move definition of class on top, so we can properly linter it.
def _exec(
    bench: "Benchmark",
    framework: Framework,
    impl_postfix: str,
    preset: str,
    repeat: int,
    results_dict: dict,
    copy_output: bool,
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
        framework : A Framework for which the benchmark is to be executed.
        impl_postfix : The identifier for the benchmark implementation.
        preset : A problem size entry defined in the bench_info JSON.
        timeout : Number of seconds after which the execution is killed.
        repeat : Number of repetitions of the benchmark execution.
        precision: The precision to use for benchmark input data.
        args : Input arguments to benchmark implementation function.
        results_dict : A dictionary where timing and other results are stored.
        copy_output : A flag that controls copying output.
    """
    np_input_data = bench.get_input_data(preset=preset)

    with timer.timer() as t:
        inputs = _set_input_args(bench, framework, np_input_data)
    results_dict["setup_time"] = t.get_elapsed_time()

    input_size = 0
    for arg in bench.info.array_args:
        input_size += _array_size(bench.bdata[preset][arg])

    results_dict["input_size"] = input_size

    impl_fn = bench.get_implementation(impl_postfix)

    # Warmup
    with timer.timer() as t:
        try:
            framework.execute(impl_fn, inputs)
        except Exception:
            logging.exception("Benchmark execution failed at the warmup step.")
            results_dict["error_state"] = ErrorCodes.FAILED_EXECUTION
            results_dict["error_msg"] = "Execution failed"
            return

    results_dict["warmup_time"] = t.get_elapsed_time()

    _reset_output_args(bench, framework, inputs, np_input_data)

    exec_times = [0] * repeat

    retval = None
    for i in range(repeat):
        with timer.timer() as t:
            retval = framework.execute(impl_fn, inputs)
        exec_times[i] = t.get_elapsed_time()

        # Do not reset the output from the last repeat
        if i < repeat - 1:
            _reset_output_args(bench, framework, inputs, np_input_data)

    results_dict["exec_times"] = exec_times

    # Get the output data
    results_dict["teardown_time"] = 0.0
    if copy_output:
        _exec_copy_output(bench, framework, retval, inputs, results_dict)

    results_dict["error_state"] = ErrorCodes.SUCCESS
    results_dict["error_msg"] = ""


def _exec_copy_output(
    bench: "Benchmark",
    fmwrk: Framework,
    retval,
    inputs: dict,
    results_dict: dict,
):
    output_arrays = dict()
    with timer.timer() as t:
        for out_arg in bench.info.output_args:
            if out_arg in bench.info.array_args:
                output_arrays[out_arg] = fmwrk.copy_from_func()(inputs[out_arg])

    # Special case: if the benchmark implementation returns anything, then
    # add that to the results dict
    if retval is not None:
        output_arrays["return-value"] = convert_to_numpy(retval, fmwrk)

    results_dict["outputs"] = output_arrays
    results_dict["teardown_time"] = t.get_elapsed_time()


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


class Benchmark(object):
    """A class for reading and benchmark information and initializing
    benchmark data.
    """

    def __init__(
        self,
        config: cfg.Benchmark,
    ):
        """Reads benchmark information.

        Args:
            config: Benchmark configuration.
        """

        # [preset] = benchmark input data
        self.bdata = dict()
        self.info: cfg.Benchmark = config

        self.initialize_fn = (
            getattr(
                importlib.import_module(self.init_mod_path), self.init_fn_name
            )
            if self.info.init
            else None
        )

    @property
    def bname(self):
        return self.info.module_name

    @property
    def init_mod_path(self):
        return self.info.init.package_path if self.info.init else None

    @property
    def init_fn_name(self):
        return self.info.init.func_name if self.info.init else None

    def get_implementation(self, implementation_postfix: str):
        implementation = None

        for impl in self.info.implementations:
            if impl.postfix == implementation_postfix:
                implementation = impl
                break

        if implementation is None:
            logging.error(
                f"Could not find implementation {implementation_postfix} for {self.bname}"
            )
            return None

        try:
            mod = importlib.import_module(implementation.package_path)
            implementation_function = getattr(mod, implementation.func_name)
        except Exception:
            logging.error(
                f"Failed to import benchmark module: {implementation.module_name}"
            )
            return None

        return implementation_function

    def _enforce_precision(
        self,
        arg_name: str,
        arg: Any,
        precision: str,
    ) -> Any:
        """Enforce selected precision on data.

        Args:
           arg_name: Name of data argument.
           arg: Input data argument whose precision is changed.
           precision: The precision used for input data argument..
           framework: A Framework for which the data is initialized.

        Returns: Copy of Input data with precision updated or
                 the original input data.
        """

        if isinstance(arg, np.ndarray):
            for _type, _precision_strings in cfg.GLOBAL.dtypes.items():
                if np.issubsctype(arg, _type):
                    return arg.astype(np.dtype(_precision_strings[precision]))

        logging.warning(
            "Precision unchanged for " + arg_name + " due to unsupported type."
        )
        return arg

    def _get_types_dict(
        self, global_precision: str, config_precision: str
    ) -> dict[str, Any]:
        """Constructs a dictionary of types with selected precision.

        Args:
           framework: A Framework for which the data is initialized.
           global_precision: The precision specified through runner.
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
                types_dict[_type] = np.dtype(_precision_strings[precision])
            except KeyError:
                raise KeyError(
                    "Precision "
                    + precision
                    + " not supported for "
                    + self.bname
                )

        return types_dict

    def get_input_data(self, preset: str) -> Dict[str, Any]:
        return self.bdata[preset]

    def initialize_input_data(
        self, preset: str, global_precision: str
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

        self._initialize_input_data_from_init(global_precision, data)

        # 8. Update the benchmark data (self.bdata) with the generated data
        #    for the provided preset.
        self.bdata[preset] = data
        return self.bdata[preset]

    def _initialize_input_data_from_init(
        self,
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
        if not self.info.init:
            return

        # 4. Store types of selected precision dict in "data" dict.
        if self.info.init.types_dict_name != "":
            data[self.info.init.types_dict_name] = self._get_types_dict(
                global_precision, self.info.init.precision
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
                            out,
                            data[out],
                            enforce_pres,
                        )
                    }
                )
