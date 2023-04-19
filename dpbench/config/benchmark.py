# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark related configuration classes."""

from dataclasses import dataclass, field
from typing import Any, List

Parameters = dict[str, Any]

Presets = dict[str, Parameters]


@dataclass
class Init:
    """Configuration for benchmark initialization."""

    func_name: str = ""
    package_path: str = ""
    module_name: str = ""
    input_args: List[str] = field(default_factory=list)
    output_args: List[str] = field(default_factory=list)

    @staticmethod
    def from_dict(obj: Any) -> "Init":
        """Convert object into Init dataclass."""
        _func_name = str(obj.get("func_name") or "")
        _package_path = str(obj.get("package_path") or "")
        _module_name = str(obj.get("module_name") or "")
        _input_args = obj.get("input_args")
        _output_args = obj.get("output_args")
        return Init(
            _func_name, _package_path, _module_name, _input_args, _output_args
        )

    def __post_init__(self):
        """Post initialization hook for dataclass. Not for direct use."""
        self.func_name = self.func_name or "initialize"


@dataclass
class BenchmarkImplementation:
    """Configuration for benchmark initialization."""

    postfix: str
    func_name: str
    module_name: str
    package_path: str

    @staticmethod
    def from_dict(obj: Any) -> "BenchmarkImplementation":
        """Convert object into Init dataclass."""
        _postfix = str(obj.get("postfix"))
        _func_name = str(obj.get("func_name"))
        _module_name = str(obj.get("module_name"))
        _package_path = str(obj.get("package_path"))
        return BenchmarkImplementation(
            _postfix, _func_name, _module_name, _package_path
        )


@dataclass
class Benchmark:
    """Configuration with benchmark information."""

    name: str = ""
    short_name: str = ""
    relative_path: str = ""
    module_name: str = ""
    package_path: str = ""
    func_name: str = ""
    kind: str = ""
    domain: str = ""
    parameters: Presets = field(default_factory=Presets)
    init: Init = None
    input_args: List[str] = field(default_factory=list)
    array_args: List[str] = field(default_factory=list)
    output_args: List[str] = field(default_factory=list)
    implementations: List[BenchmarkImplementation] = field(default_factory=list)

    @staticmethod
    def from_dict(obj: Any) -> "Benchmark":
        """Convert object into Benchmark dataclass."""
        _name = str(obj.get("name") or "")
        _short_name = str(obj.get("short_name") or "")
        _relative_path = str(obj.get("relative_path") or "")
        _module_name = str(obj.get("module_name") or "")
        _package_path = str(obj.get("package_path") or "")
        _func_name = str(obj.get("func_name") or "")
        _kind = str(obj.get("kind") or "")
        _domain = str(obj.get("domain") or "")
        _parameters = Presets(obj.get("parameters"))
        _init = obj.get("init")
        _init = Init.from_dict(_init) if _init else None
        _input_args = obj.get("input_args") or []
        _array_args = obj.get("array_args") or []
        _output_args = obj.get("output_args") or []
        _implementations = obj.get("implementations") or []
        return Benchmark(
            _name,
            _short_name,
            _relative_path,
            _module_name,
            _package_path,
            _func_name,
            _kind,
            _domain,
            _parameters,
            _init,
            _input_args,
            _array_args,
            _output_args,
            _implementations,
        )
