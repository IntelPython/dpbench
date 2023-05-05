# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Report subcommand package."""

import argparse

from ._namespace import Namespace


def add_config_arguments(parser: argparse.ArgumentParser):
    """Add arguments for the config subcommand.

    Args:
        parser: argument parser where arguments will be populated.
    """
    parser.add_argument(
        "--color",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set if print with color.",
    )
    pass


def execute_config(args: Namespace):
    """Execute config sub command.

    Print loaded configuration in pretty format

    Args:
        args: object with all input arguments.
        conn: database connection.
    """
    import dpbench.config as cfg

    cfg.GLOBAL = cfg.read_configs(
        benchmarks=args.benchmarks,
        implementations=args.implementations,
        with_npbench=True,
        with_polybench=True,
    )

    if args.color:
        from pprint import pformat

        from pygments import highlight
        from pygments.formatters import Terminal256Formatter
        from pygments.lexers import PythonLexer

        print(
            highlight(
                pformat(cfg.GLOBAL), PythonLexer(), Terminal256Formatter()
            ),
            end="",
        )
    else:
        from pprint import pprint

        pprint(cfg.GLOBAL)
