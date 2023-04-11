# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Entry point for dpbench cli tool."""

import argparse
from importlib.metadata import version

from ._namespace import CommaSeparateStringAction, Namespace
from .report import add_report_arguments, execute_report
from .run import add_run_arguments, execute_run


def parse_args() -> Namespace:
    """Parse console arguments into dpbench Namespace."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-b",
        "--benchmarks",
        type=str,
        action=CommaSeparateStringAction,
        nargs="?",
        default={},
        help="Comma separated list of benchmarks. Leave empty to load all "
        + "benchmarks.",
    )
    parser.add_argument(
        "-i",
        "--implementations",
        type=str,
        action=CommaSeparateStringAction,
        nargs="?",
        default={"python", "numpy"},
        help="Comma separated list of implementations. Use "
        + "--all-implementations to load all available implementations.",
    )
    parser.add_argument(
        "-a",
        "--all-implementations",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, all available implementations will be loaded.",
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s " + version("dpbench")
    )
    parser.add_argument(
        "-r",
        "--run-id",
        type=int,
        nargs="?",
        default=None,
        help="run_id to perform actions on. Use --last-run to use latest"
        + " available run, or leave empty to create new one.",
    )
    parser.add_argument(
        "--last-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Sets run_id to the latest run_id from the database.",
    )
    parser.add_argument(
        "-d",
        "--results-db",
        type=str,
        nargs="?",
        default="results.db",
        help="Path to a database to store results.",
    )

    subparsers = parser.add_subparsers(dest="program")

    run_parser = subparsers.add_parser(
        "run", description="Subcommand to run benchmark executions."
    )

    add_run_arguments(run_parser)

    report_parser = subparsers.add_parser(
        "report",
        description="Subcommand to generate report for the existing run.",
    )

    add_report_arguments(report_parser)

    return parser.parse_args(namespace=Namespace())


def main():
    """Main function to run on dpbench console tool."""
    args = parse_args()

    if args.program in {"run", "report"}:
        import dpbench.infrastructure as dpbi
        from dpbench.infrastructure.reporter import update_run_id

        dpbi.create_results_table(db_file=args.results_db)
        conn = dpbi.create_connection(db_file=args.results_db)

    if args.last_run:
        if args.run_id is not None:
            raise ValueError("Last-run was requested but run_id was provided.")
        args.run_id = update_run_id(conn, None)

    if args.all_implementations:
        args.implementations = {}
    if args.program == "run":
        execute_run(args, conn)
    elif args.program == "report":
        execute_report(args, conn)


if __name__ == "__main__":
    main()
