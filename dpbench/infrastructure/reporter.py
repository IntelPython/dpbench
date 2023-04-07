# Copyright 2022 Intel Corp.
# SPDX-FileCopyrightText: 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache 2.0
# SPDX-License-Identifier: Apache-2.0

"""The module generates reports for implementation summary and timing summary
from a specific benchmark run.
"""

import pathlib
from typing import Final, Union

import pandas as pd
import sqlalchemy
from sqlalchemy import case, func
from sqlalchemy.orm import Session

from . import datamodel as dm

__all__ = [
    "generate_impl_summary_report",
    "generate_performance_report",
]


def update_run_id(conn: sqlalchemy.Engine, run_id: Union[int, None]) -> int:
    """checks if run_id was provided. Otherwise returns the latest available one"""
    if run_id is not None:
        return run_id

    with Session(conn) as session:
        run_id = (
            session.query(
                dm.Run.id,
            )
            .order_by(dm.Run.created_at.desc())
            .limit(1)
            .scalar()
        )

        print(
            f"WARNING: run_id was not provided, using the latest one {run_id}"
        )


def update_connection(
    results_db: Union[str, sqlalchemy.Engine] = "results.db",
) -> sqlalchemy.Engine:
    """checks if database provided as a path and returns sqlalchemy.Engine"""
    if not isinstance(results_db, sqlalchemy.Engine):
        return dm.create_connection(db_file=results_db)

    return results_db


def read_legends() -> pd.DataFrame:
    """reads implementation postfixes"""
    parent_folder = pathlib.Path(__file__).parent.absolute()
    impl_postfix_path = parent_folder.joinpath(
        "..", "configs", "impl_postfix.json"
    )
    return pd.read_json(impl_postfix_path)


def generate_header(conn: sqlalchemy.Engine, run_id: int):
    """prints header section"""
    with Session(conn) as session:
        created_at = (
            session.query(
                func.datetime(dm.Run.created_at, "unixepoch", "localtime"),
            )
            .where(dm.Run.id == run_id)
            .limit(1)
            .scalar()
        )

        print(f"Report for {created_at} run")
        print("==================================")


def generate_legend(legends: pd.DataFrame):
    """prints legend section"""
    formatters = {}
    for col in legends.select_dtypes("object"):
        len_max = legends[col].str.len().max()
        formatters[col] = lambda _, len_max=len_max: f"{_:<{len_max}s}"

    print("Legend")
    print("======")
    print(legends.to_string(formatters=formatters))
    print("")


def generate_summary(data: pd.DataFrame):
    """prints summary section"""
    print("Summary of current implementation")
    print("=================================")
    print(data.to_string())


def generate_impl_summary_report(
    results_db: Union[str, sqlalchemy.Engine] = "results.db",
    run_id: int = None,
):
    """generate implementation summary report with status of each benchmark"""
    conn = update_connection(results_db=results_db)
    run_id = update_run_id(conn, run_id)
    legends = read_legends()

    generate_header(conn, run_id)
    generate_legend(legends)

    columns = [
        dm.Result.input_size_human.label("input_size"),
        dm.Result.benchmark,
        dm.Result.problem_preset,
    ]

    for _, row in legends.iterrows():
        impl = row["impl_postfix"]
        columns.append(
            func.ifnull(
                func.max(
                    case(
                        (
                            dm.Result.implementation == impl,
                            dm.Result.error_state,
                        ),
                    )
                ),
                "N/A",
            ).label(impl),
        )

    sql = (
        sqlalchemy.select(*columns)
        .group_by(
            dm.Result.benchmark,
            dm.Result.problem_preset,
        )
        .where(
            dm.Result.run_id == run_id,
        )
    )

    df = pd.read_sql_query(
        sql=sql,
        con=conn.connect(),
    )

    generate_summary(df)


def generate_performance_report(
    results_db: Union[str, sqlalchemy.Engine] = "results.db",
    run_id: int = None,
):
    """generate performance report with median times for each benchmark"""
    conn = update_connection(results_db=results_db)
    run_id = update_run_id(conn, run_id)
    legends = read_legends()

    generate_header(conn, run_id)
    generate_legend(legends)

    columns = [
        dm.Result.input_size_human.label("input_size"),
        dm.Result.benchmark,
        dm.Result.problem_preset,
    ]

    for _, row in legends.iterrows():
        impl = row["impl_postfix"]
        columns.append(
            func.ifnull(
                func.max(
                    case(
                        (
                            dm.Result.implementation == impl,
                            dm.Result.median_exec_time,
                        ),
                    )
                ),
                None,
            ).label(impl),
        )

    sql = (
        sqlalchemy.select(*columns)
        .group_by(
            dm.Result.benchmark,
            dm.Result.problem_preset,
        )
        .where(dm.Result.run_id == run_id)
    )

    df = pd.read_sql_query(
        sql=sql,
        con=conn.connect(),
    )

    for index, row in df.iterrows():
        for _, legend_row in legends.iterrows():
            impl = legend_row["impl_postfix"]

            time = row[impl]
            if time:
                NANOSECONDS_IN_MILISECONDS: Final[float] = 1000 * 1000.0
                time /= NANOSECONDS_IN_MILISECONDS

                time = str(round(time, 2)) + "ms"
            else:
                time = "n/a"

            df.at[index, impl] = time

    generate_summary(df)
