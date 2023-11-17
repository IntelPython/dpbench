# Copyright 2022 Intel Corp.
# SPDX-FileCopyrightText: 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""The module generates reports for implementation summary and timing summary
from a specific benchmark run.
"""

import dataclasses
import logging
from typing import Final, Union

import pandas as pd
import sqlalchemy
from sqlalchemy import case, func
from sqlalchemy.orm import Session

import dpbench.config as cfg

from . import datamodel as dm

__all__ = [
    "generate_impl_summary_report",
    "generate_performance_report",
    "generate_comparison_report",
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

        logging.warn(f"using the latest run_id {run_id}")

        return run_id


def update_connection(
    results_db: Union[str, sqlalchemy.Engine] = "results.db",
) -> sqlalchemy.Engine:
    """checks if database provided as a path and returns sqlalchemy.Engine"""
    if not isinstance(results_db, sqlalchemy.Engine):
        return dm.create_connection(db_file=results_db)

    return results_db


def read_legends() -> pd.DataFrame:
    """reads implementation postfixes"""
    return pd.DataFrame.from_records(
        [dataclasses.asdict(impl) for impl in cfg.GLOBAL.implementations]
    )


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


def generate_legend(conn: sqlalchemy.Engine, run_id: int) -> list[str]:
    """prints legend section and returns implementation list"""
    sql = (
        sqlalchemy.select(
            dm.Postfix.postfix,
            dm.Postfix.description,
            dm.Postfix.device,
        )
        .order_by(dm.Postfix.postfix)
        .where(dm.Postfix.run_id == run_id)
    )

    legends = pd.read_sql_query(
        sql=sql,
        con=conn.connect(),
    )

    formatters = {}
    for col in legends.select_dtypes("object"):
        len_max = legends[col].str.len().max()
        formatters[col] = lambda _, len_max=len_max: f"{_:<{len_max}s}"

    print("Legend")
    print("======")
    print(legends.to_string(formatters=formatters))
    print("")

    return legends["postfix"].values.tolist()


def generate_summary(data: pd.DataFrame):
    """prints summary section"""
    print("Summary of current implementation")
    print("=================================")
    print(data.to_string())


def generate_impl_summary_report(
    conn: sqlalchemy.Engine,
    run_id: int,
    implementations: list[str],
):
    """generate implementation summary report with status of each benchmark"""
    columns = [
        func.max(dm.Result.input_size_human).label("input_size"),
        dm.Result.benchmark,
        dm.Result.problem_preset,
    ]

    for impl in implementations:
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
    conn: sqlalchemy.Engine,
    run_id: int,
    implementations: list[str],
):
    """generate performance report with median times for each benchmark"""
    columns = [
        func.max(dm.Result.input_size_human).label("input_size"),
        dm.Result.benchmark,
        dm.Result.problem_preset,
    ]

    for impl in implementations:
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

    NA = "n/a"

    df = (
        pd.read_sql_query(sql=sql, con=conn.connect())
        .astype({impl: "string" for impl in implementations})
        .replace("0.0", NA)
        .fillna(NA)
    )

    for index, row in df.iterrows():
        for impl in implementations:
            time = row[impl]

            if time == NA:
                continue

            NANOSECONDS_IN_MILISECONDS: Final[float] = 1000 * 1000.0
            time = float(time)
            time /= NANOSECONDS_IN_MILISECONDS

            time = str(round(time, 2)) + "ms"

            df.at[index, impl] = time

    generate_summary(df)


def generate_comparison_report(
    conn: sqlalchemy.Engine,
    run_id: int,
    implementations: list[str],
    comparison_pairs: list[tuple[str, str]],
):
    """generate comparison report with median times for each benchmark"""
    if len(comparison_pairs) == 0:
        return

    columns = [
        func.max(dm.Result.input_size_human).label("input_size"),
        dm.Result.benchmark,
        dm.Result.problem_preset,
    ]

    for impl in implementations:
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
        for target, reference in comparison_pairs:
            if row[reference] == 0 or row[target] == 0:
                boost = "n/a"
            else:
                boost = (
                    str(round((row[target] / row[reference]) * 100, 2)) + "%"
                )
            df.at[index, target + "_to_" + reference] = boost

    for impl in implementations:
        df = df.drop(impl, axis=1)

    generate_summary(df)


def get_failures_from_results(
    results_db: Union[str, sqlalchemy.Engine] = "results.db",
    run_id: int = None,
) -> list[tuple[str, str]]:
    conn = update_connection(results_db=results_db)
    run_id = update_run_id(conn, run_id)

    sql = sqlalchemy.select(
        dm.Result.benchmark,
        dm.Result.implementation,
    ).where(
        sqlalchemy.or_(
            dm.Result.error_state == "Failed Execution",
            dm.Result.error_state == "Execution Timeout",
        ),
        dm.Result.run_id == run_id,
    )

    with conn.connect() as connection:
        return connection.execute(sql).all()


def get_unexpected_failures(
    results_db: Union[str, sqlalchemy.Engine] = "results.db",
    run_id: int = None,
):
    expected_failures = {
        (b.module_name, impl)
        for b in cfg.GLOBAL.benchmarks
        for impl in b.expected_failure_implementations
    }

    failures = {f for f in get_failures_from_results(results_db, run_id)}

    fixed_failures = expected_failures.difference(failures)
    if len(fixed_failures):
        logging.warn(
            f"these benchmarks does not fail anymore: {fixed_failures}"
        )

    return failures.difference(expected_failures)


def print_report(
    conn: sqlalchemy.Engine,
    run_id: int,
    comparison_pairs: list[tuple[str, str]] = [],
):
    generate_header(conn, run_id)
    implementations = generate_legend(conn, run_id)

    generate_impl_summary_report(
        conn, run_id=run_id, implementations=implementations
    )

    generate_performance_report(
        conn,
        run_id=run_id,
        implementations=implementations,
    )

    generate_comparison_report(
        conn,
        run_id=run_id,
        implementations=implementations,
        comparison_pairs=comparison_pairs,
    )

    unexpected_failures = get_unexpected_failures(conn, run_id=run_id)

    if len(unexpected_failures) > 0:
        raise ValueError(
            f"Unexpected benchmark implementations failed: {unexpected_failures}.",
        )
