# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import pathlib
import warnings
from inspect import getmembers, isbuiltin, isfunction
from typing import Any, Dict


def get_supported_implementation_postfixes():
    """Returns as a dictionary all the supported postfixes for filenames
    that implement a specific version of a benchmark.

    Returns:
        Dict: Key is the string providing the supported postfix and value is a
        string describing when to use the postfix.
    """
    parent_folder = pathlib.Path(__file__).parent.absolute()
    impl_postfix_json = parent_folder.joinpath("..", "configs", "impl_postfix")

    try:
        with open(impl_postfix_json) as json_file:
            return json.load(json_file)["impl_postfix"]
    except Exception as e:
        warnings.warn("impl_postfix.json file not found")
        raise (e)


class Benchmark(object):
    """A class for reading and benchmark information and initializing
    benchmark data.
    """

    def _set_implementation_fn_list(self, bmod, initialize_fname):
        """Selects all the callables from the __all__ list for the module
        excluding the initialize function that we know is not a benchmark
        implementation.

        Args:
            bmod : A benchmark module
            initialize_fname : Name of the initialization function
        """

        self.impl_fnlist = [
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
            bench_path = pathlib.Path(bconfig_path).joinpath(bench_filename)
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
        """Loads the "initialize" function from the provided module.

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
            self._set_implementation_fn_list(bmodule, self.init_fn_name)
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

    def get_impl(self, impl_postfix: str):
        impl_postfixes = get_supported_implementation_postfixes()
        if impl_postfix in impl_postfixes:
            fn = [
                impl[1] for impl in self.impl_fnlist if impl_postfix in impl[0]
            ]
            if len(fn) > 1:
                raise RuntimeError(
                    "Multiple implementations for " + impl_postfix
                )
            return fn[0]
        else:
            raise RuntimeError(
                "Implementation postfix " + impl_postfix + " not supported."
            )

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
        initialized_output = self.initialize_fn(*data.values())

        # 5. Store the initialized output in the "data" dict. Note that the
        #    implementation depends on Python dicts being ordered. Thus, the
        #    code will not work with Python older than 3.7.
        for idx, out in enumerate(self.info["init"]["output_args"]):
            data.update({out: initialized_output[idx]})

        # 6. Update the benchmark data (self.bdata) with the generated data
        #    for the provided preset.
        self.bdata[preset] = data
        return self.bdata[preset]
