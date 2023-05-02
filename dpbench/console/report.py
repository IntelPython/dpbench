# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Report subcommand package."""

import argparse

import sqlalchemy

from ._namespace import Namespace


def add_report_arguments(parser: argparse.ArgumentParser):
    """Add arguments for the report subcommand.

    Args:
        parser: argument parser where arguments will be populated.
    """
    pass


def execute_report(args: Namespace, conn: sqlalchemy.Engine):
    """Execute report sub command.

    Report subcommand reads configuration and generates report.

    Args:
        args: object with all input arguments.
        conn: database connection.
    """
    import dpbench.config as cfg
    from dpbench.infrastructure.reporter import update_run_id
    from dpbench.infrastructure.runner import print_report

    cfg.GLOBAL = cfg.read_configs(
        benchmarks=args.benchmarks,
        implementations=args.implementations,
        load_implementations=False,
    )

    args.run_id = update_run_id(conn, args.run_id)
    print_report(
        conn=conn,
        run_id=args.run_id,
        implementations=args.implementations,
    )
