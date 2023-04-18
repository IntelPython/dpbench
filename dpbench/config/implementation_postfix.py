# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Implementation related configuration classes."""

from dataclasses import dataclass
from typing import Any


@dataclass
class Implementation:
    """Configuration with implementation information."""

    postfix: str
    description: str

    @staticmethod
    def from_dict(obj: Any) -> "Implementation":
        """Convert object into Implementation dataclass."""
        _impl_postfix = str(obj.get("impl_postfix"))
        _description = str(obj.get("description"))
        return Implementation(_impl_postfix, _description)
