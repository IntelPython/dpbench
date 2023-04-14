# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Framework related configuration classes."""

from dataclasses import dataclass
from typing import Any


@dataclass
class Framework:
    """Configuration with framework information."""

    simple_name: str
    full_name: str
    prefix: str
    class_: str
    arch: str

    @staticmethod
    def from_dict(obj: Any) -> "Framework":
        """Convert object into Framework dataclass."""
        _simple_name = str(obj.get("simple_name"))
        _full_name = str(obj.get("full_name"))
        _prefix = str(obj.get("prefix"))
        _class = str(obj.get("class"))
        _arch = str(obj.get("arch"))
        return Framework(_simple_name, _full_name, _prefix, _class, _arch)
