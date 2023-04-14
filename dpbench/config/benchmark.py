# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark related configuration classes."""

from dataclasses import dataclass
from typing import Any, List

Parameters = dict[str, Any]

Presets = dict[str, Parameters]


@dataclass
class Init:
    """Configuration for benchmark initialization."""

    func_name: str
    input_args: List[str]
    output_args: List[str]

    @staticmethod
    def from_dict(obj: Any) -> "Init":
        """Convert object into Init dataclass."""
        _func_name = str(obj.get("func_name"))
        _input_args = obj.get("input_args")
        _output_args = obj.get("output_args")
        return Init(_func_name, _input_args, _output_args)


@dataclass
class Benchmark:
    """Configuration with benchmark information."""

    name: str
    short_name: str
    relative_path: str
    module_name: str
    func_name: str
    kind: str
    domain: str
    parameters: Presets
    init: Init
    input_args: List[str]
    array_args: List[str]
    output_args: List[str]

    @staticmethod
    def from_dict(obj: Any) -> "Benchmark":
        """Convert object into Benchamrk dataclass."""
        _name = str(obj.get("name"))
        _short_name = str(obj.get("short_name"))
        _relative_path = str(obj.get("relative_path"))
        _module_name = str(obj.get("module_name"))
        _func_name = str(obj.get("func_name"))
        _kind = str(obj.get("kind"))
        _domain = str(obj.get("domain"))
        _parameters = Presets(obj.get("parameters"))
        _init = Init.from_dict(obj.get("init"))
        _input_args = obj.get("input_args")
        _array_args = obj.get("input_args")
        _output_args = obj.get("input_args")
        return Benchmark(
            _name,
            _short_name,
            _relative_path,
            _module_name,
            _func_name,
            _kind,
            _domain,
            _parameters,
            _init,
            _input_args,
            _array_args,
            _output_args,
        )
