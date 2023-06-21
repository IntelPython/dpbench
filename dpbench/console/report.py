# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Report subcommand package."""

import argparse

import sqlalchemy

from ._namespace import CommaSeparateStringListAction, Namespace


def add_report_arguments(parser: argparse.ArgumentParser):
    """Add arguments for the report subcommand.

    Args:
        parser: argument parser where arguments will be populated.
    """
    parser.add_argument(
        "-c",
        "--comparisons",
        type=str,
        action=CommaSeparateStringListAction,
        nargs="?",
        default=[],
        help="Comma separated list of implementation pairs that need to be"
        + " compared.",
    )


def execute_report(args: Namespace, conn: sqlalchemy.Engine):
    """Execute report sub command.

    Report subcommand reads configuration and generates report.

    Args:
        args: object with all input arguments.
        conn: database connection.
    """
    if len(args.comparisons) % 2 != 0:
        raise ValueError(
            "--comparisons must contain pairs, but odd number of"
            + " elements was provided"
        )

    import dpbench.config as cfg
    from dpbench.infrastructure.reporter import print_report, update_run_id

    cfg.GLOBAL = cfg.read_configs(
        benchmarks=args.benchmarks,
        implementations=args.implementations,
        load_implementations=False,
    )

    comparison_pairs = [
        tuple(args.comparisons[i : i + 2])
        for i in range(0, len(args.comparisons), 2)
    ]

    args.run_id = update_run_id(conn, args.run_id)
    print_report(
        conn=conn,
        run_id=args.run_id,
        comparison_pairs=comparison_pairs,
    )
