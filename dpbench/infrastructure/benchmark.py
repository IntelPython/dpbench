# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import pathlib
from typing import Any, Dict


class Benchmark(object):
    """A class for reading and benchmark information and initializing
    benchmark data."""

    def __init__(self, bname: str, bconfig_path: str = None):
        """Reads benchmark information.
        :param bname: The benchmark name.
        "param config_path: Optional location of the config JSON file for the
        benchmark. If none is provided, the default config inside the
        package's bench_info directory is used.
        """

        self.bname = bname
        self.bdata = dict()

        bench_filename = "{b}.json".format(b=bname)
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
        except Exception as e:
            print(
                "Benchmark JSON file {b} could not be opened.".format(
                    b=bench_filename
                )
            )
            raise (e)

    def get_data(self, preset: str = "L") -> Dict[str, Any]:
        """Initializes the benchmark data.
        :param preset: The data-size preset (S, M, L, paper).
        """

        if preset in self.bdata.keys():
            return self.bdata[preset]

        # 1. Create data dictionary
        data = dict()
        # 2. Add parameters to data dictionary
        if preset not in self.info["parameters"].keys():
            raise NotImplementedError(
                "{b} doesn't have a {p} preset.".format(b=self.bname, p=preset)
            )
        parameters = self.info["parameters"][preset]
        for k, v in parameters.items():
            data[k] = v
        # 3. Import initialization function
        if "init" in self.info.keys() and self.info["init"]:
            init_module=self.info["module_name"] + "_initialize"
            module_filename = "{m}.py".format(
                m=init_module
            )
            module_pypath = "dpbench.benchmarks.{r}.{m}".format(
                r=self.info["relative_path"].replace("/", "."),
                m=init_module,
            )
            exec_str = "from {m} import {i}".format(
                m=module_pypath, i=self.info["init"]["func_name"]
            )
            try:
                exec(exec_str, data)
            except Exception as e:
                print(
                    "Module Python file {m} could not be opened.".format(
                        m=module_filename
                    )
                )
                raise (e)
            # 4. Execute initialization
            init_str = "{oargs} = {i}({iargs})".format(
                oargs=",".join(self.info["init"]["output_args"]),
                i=self.info["init"]["func_name"],
                iargs=",".join(self.info["init"]["input_args"]),
            )
            try:
                exec(init_str, data)
            except Exception as e:
                print(
                    "Benchmark {m} could not be initialized with data.".format(
                        m=module_filename
                    )
                )
                raise (e)
            finally:
                del data[self.info["init"]["func_name"]]

        self.bdata[preset] = data
        return self.bdata[preset]
