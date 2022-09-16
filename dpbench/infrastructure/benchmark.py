# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import pathlib
import warnings
from inspect import getmembers
from typing import Any, Dict

import dpbench.infrastructure as dpbi
from dpbench.infrastructure import timeout_decorator as tout
from dpbench.infrastructure import timer

from .framework import Framework


def get_supported_implementation_postfixes():
    """Returns as a dictionary all the supported postfixes for filenames
    that implement a specific version of a benchmark.

    Returns:
        Dict: Key is the string providing the supported postfix and value is a
        string describing when to use the postfix.
    """
    parent_folder = pathlib.Path(__file__).parent.absolute()
    impl_postfix_json = parent_folder.joinpath(
        "..", "configs", "impl_postfix.json"
    )

    try:
        with open(impl_postfix_json) as json_file:
            return json.load(json_file)["impl_postfix"]
    except Exception as e:
        warnings.warn("impl_postfix.json file not found")
        raise (e)


class BenchmarkRunner:
    """_summary_"""

    def __init__(self, bench, impl_postfix, preset, repeat, timeout):
        self.bench = bench
        self.impl_fn = self.bench.get_impl(impl_postfix)
        self.fmwrk = self.bench.get_framework(impl_postfix)
        self.preset = preset
        self.repeat = repeat
        self.timeout = timeout
        self.copied_args = dict()
        self.output = dict()

        if self.impl_fn:
            # Run setup step
            self._setup()
            # Execute the benchmark
            self._exec()
            # Copy back any data from a device to the host
            self._teardown()

    def _setup(self):
        initialized_data = self.bench.get_data(preset=self.preset)
        array_args = self.bench.info["array_args"]

        with timer.timer() as t:
            for arg in array_args:
                npdata = initialized_data[arg]
                self.copied_args.update(
                    {arg: self.fmwrk.copy_to_func()(npdata)}
                )

        self.setup_time = t.get_elapsed_time()

    def _reset_output_args(self, inputs):
        output_args = self.bench.info["output_args"]
        array_args = self.bench.info["array_args"]

        for arg in inputs.keys():
            if arg in output_args and arg in array_args:
                original_data = self.bench.get_data(preset=self.preset)[arg]
                inputs.update({arg: self.fmwrk.copy_to_func()(original_data)})

    def _exec(self):
        input_args = self.bench.info["input_args"]
        array_args = self.bench.info["array_args"]
        inputs = dict()
        for arg in input_args:
            if arg not in array_args:
                inputs.update(
                    {arg: self.bench.get_data(preset=self.preset)[arg]}
                )
            else:
                inputs.update({arg: self.copied_args[arg]})

        # Warmup
        @tout.exit_after(self.timeout)
        def warmup(impl_fn, input_list):
            impl_fn(*input_list)

        warmup(self.impl_fn, inputs.values())
        self._reset_output_args(inputs=inputs)

        # Get the output data
        for out_arg in self.bench.info["output_args"]:
            self.output.update({out_arg: inputs[out_arg]})

        for _ in range(self.repeat):
            self.impl_fn(*inputs.values())
            self._reset_output_args(inputs=inputs)

    def _teardown(self):

        array_args = self.bench.info["array_args"]
        out_args = self.bench.info["output_args"]

        for arg in out_args:
            if arg in array_args:
                copied_back = self.fmwrk.copy_from_func()(self.output[arg])
                self.output[arg] = copied_back

    def get_results(self):
        return self.output


class Benchmark(object):
    """A class for reading and benchmark information and initializing
    benchmark data.
    """

    def _check_if_valid_impl_postfix(self, impl_postfix: str) -> bool:
        """Checks if an implementation postfix is found in the
        impl_postfix.json.

        Args:
            impl_postfix (str): An implementation postfix

        Returns:
            bool: True if the postfix is found in the JSON file else False
        """
        impl_postfixes = get_supported_implementation_postfixes()
        if impl_postfix in impl_postfixes:
            return True
        else:
            return False

    def _set_implementation_fn_list(self, bmod, initialize_fname):
        """Selects all the callables from the __all__ list for the module
        excluding the initialize function that we know is not a benchmark
        implementation.

        Args:
            bmod : A benchmark module
            initialize_fname : Name of the initialization function
        """

        return [
            fn
            for fn in getmembers(bmod, callable)
            if initialize_fname not in fn[0]
        ]

    def _load_benchmark_info(self, bconfig_path: str = None):
        """Reads the benchmark configuration and loads into a member dict.

        Args:
            bconfig_path (str, optional): _description_. Defaults to None.
        """
        bench_filename = "{b}.json".format(b=self.bname)
        bench_path = None

        if bconfig_path:
            bench_path = bconfig_path.joinpath(bench_filename)
        else:
            parent_folder = pathlib.Path(__file__).parent.absolute()
            bench_path = parent_folder.joinpath(
                "..", "configs", "bench_info", bench_filename
            )

        try:
            with open(bench_path) as json_file:
                self.info = json.load(json_file)["benchmark"]
        except Exception:
            warnings.warn(
                "Benchmark JSON file {b} could not be opened.".format(
                    b=bench_filename
                )
            )
            raise

    def _set_data_initialization_fn(self, bmodule):
        """Sets the initialize function object to be used by the benchmark.

        Raises:
            RuntimeError: If the module's initialize function could not be
            loaded.
        """

        if "init" in self.info.keys() and self.info["init"]:
            self.init_fn_name = self.info["init"]["func_name"]
            self.initialize_fn = getattr(bmodule, self.init_fn_name)
        else:
            raise RuntimeError(
                "Initialization function not specified in JSON configuration"
                + " for "
                + self.bname
            )

    def _set_reference_implementation(self, impl_fnlist):
        """Sets the reference implementation for the benchmark.

        The reference implementation is either a pure Python implementation
        if available, or else a NumPy implementation. If neither is found, then
        the reference implementation is set to None.

        Args:
            impl_fnlist : The list of implementation function for the
            benchmark.
        """
        ref_impl_fn = None

        for fn in impl_fnlist:
            if "python" in fn[0]:
                ref_impl_fn = fn
            elif "numpy" in fn[0]:
                ref_impl_fn = fn

        return ref_impl_fn

    def _set_impl_to_framework_map(self, impl_fnlist):
        """Create a dictionary mapping each implementation function name to a
        corresponding Framework object.

        Args:
            impl_fnlist : list of implementation functions

        Returns:
            Dict: Dictionary mapping implementation function to a Framework
        """

        impl_to_fw_map = dict()

        for bimpl in impl_fnlist:

            if "_numba" in bimpl[0] and "_dpex" not in bimpl[0]:
                impl_to_fw_map.update({bimpl[0]: dpbi.NumbaFramework("numba")})
            elif "_numpy" in bimpl[0]:
                impl_to_fw_map.update({bimpl[0]: dpbi.Framework("numpy")})
            elif "_python" in bimpl[0]:
                impl_to_fw_map.update({bimpl[0]: dpbi.Framework("python")})
            elif "_dpex" in bimpl[0]:
                impl_to_fw_map.update(
                    {bimpl[0]: dpbi.NumbaDpexFramework("numba_dpex")}
                )
            elif "_sycl" in bimpl[0]:
                impl_to_fw_map.update({bimpl[0]: dpbi.DpcppFramework("dpcpp")})
            elif "_dpnp" in bimpl[0]:
                # FIXME: Fix the dpnp framework and implementations
                warnings.warn(
                    "DPNP Framework is broken, skipping dpnp implementation"
                )

        return impl_to_fw_map

    def __init__(self, bmodule: object, bconfig_path: str = None):
        """Reads benchmark information.
        :param bname: The benchmark name.
        "param config_path: Optional location of the config JSON file for the
        benchmark. If none is provided, the default config inside the
        package's bench_info directory is used.
        """
        self.bname = bmodule.__name__.split(".")[-1]
        self.bdata = dict()
        try:
            self._load_benchmark_info(bconfig_path)
            self._set_data_initialization_fn(bmodule)
            self.impl_fnlist = self._set_implementation_fn_list(
                bmodule, self.init_fn_name
            )
            self.ref_impl_fn = self._set_reference_implementation(
                self.impl_fnlist
            )
            self.impl_to_fw_map = self._set_impl_to_framework_map(
                self.impl_fnlist
            )
        except Exception:
            raise

    def get_impl_fnlist(self):
        """Returns a list of function objects each for a single implementation
        of the benchmark.

        Returns:
            list[tuple(str, object)]: A list of 2-tuple. The first element of
            the tuple is the string function name and the second element is
            the actual function object.
        """
        return self.impl_fnlist

    def has_impl(self, impl_postfix: str):
        impls = [
            impl
            for impl in self.impl_fnlist
            if self.bname + "_" + impl_postfix == impl[0]
        ]
        if len(impls) == 1:
            return True
        else:
            return False

    def get_impl(self, impl_postfix: str):

        fn = [
            impl[1]
            for impl in self.impl_fnlist
            if self.bname + "_" + impl_postfix == impl[0]
        ]
        if len(fn) > 1:
            warnings.warn(
                "Unable to select any implementation as there are "
                + "multiple implementations for "
                + impl_postfix
            )
            return None
        elif not fn:
            warnings.warn(
                "No implementation exists for postfix: " + impl_postfix
            )
            return None
        else:
            return fn[0]

    def get_framework(self, impl_postfix: str) -> Framework:
        try:
            return self.impl_to_fw_map[self.bname + "_" + impl_postfix]
        except KeyError:
            warnings.warn(
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

        if preset in self.bdata.keys():
            return self.bdata[preset]

        # 1. Create data dictionary
        data = dict()

        # 2. Check if the provided preset configuration is available in the
        #    config file.
        if preset not in self.info["parameters"].keys():
            raise NotImplementedError(
                "{b} doesn't have a {p} preset.".format(b=self.bname, p=preset)
            )

        # 3. Store the input preset args in the "data" dict.
        parameters = self.info["parameters"][preset]
        for k, v in parameters.items():
            data[k] = v

        # 4. Call the initialize_fn with the input args and store the results
        #    in the "data" dict.

        init_input_args_list = self.info["init"]["input_args"]
        init_input_args_val_list = []
        for arg in init_input_args_list:
            init_input_args_val_list.append(data[arg])

        initialized_output = self.initialize_fn(*init_input_args_val_list)

        # 5. Store the initialized output in the "data" dict. Note that the
        #    implementation depends on Python dicts being ordered. Thus, the
        #    code will not work with Python older than 3.7.
        for idx, out in enumerate(self.info["init"]["output_args"]):
            data.update({out: initialized_output[idx]})

        # 6. Update the benchmark data (self.bdata) with the generated data
        #    for the provided preset.
        self.bdata[preset] = data
        return self.bdata[preset]

    def run(
        self,
        implementation_postfix: str = None,
        preset: str = "S",
        repeat: int = 10,
        validate: bool = True,
        timeout: float = 200.0,
    ):

        if not self.has_impl(implementation_postfix):
            raise NotImplementedError(
                "The benchmark "
                + self.bname
                + " has no implementation for "
                + implementation_postfix
            )
        results = []
        if implementation_postfix:
            # Run the benchmark for a specific implementation
            runner = BenchmarkRunner(
                bench=self,
                impl_postfix=implementation_postfix,
                preset=preset,
                repeat=repeat,
                timeout=timeout,
            )

            results.append(runner.get_results())
        else:

            # Run the benchmark for all available implementations
            for impl in self.get_impl_fnlist():
                impl_postfix = impl[0][
                    (len(self.bname) - len(impl[0]) + 1) :  # noqa: E203
                ]
                runner = BenchmarkRunner(
                    bench=self,
                    impl_postfix=impl_postfix,
                    preset=preset,
                    repeat=repeat,
                    timeout=timeout,
                )
                results.append(runner.get_results())

        return results
