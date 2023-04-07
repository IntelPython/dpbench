# Copyright 2022 Intel Corp.
# SPDX-FileCopyrightText: 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache 2.0
# SPDX-License-Identifier: Apache-2.0

"""The module generates reports for implementation summary and timing summary
from a specific benchmark run.
"""

import pathlib
from typing import Union

import pandas as pd
import sqlalchemy
from sqlalchemy import case, func
from sqlalchemy.orm import Session

from . import datamodel as dm

__all__ = [
    "generate_impl_summary_report",
]


def generate_impl_summary_report(
    results_db: Union[str, sqlalchemy.Engine] = "results.db",
    run_id: int = None,
):
    conn = results_db
    if not isinstance(conn, sqlalchemy.Engine):
        conn = dm.create_connection(db_file=results_db)

    parent_folder = pathlib.Path(__file__).parent.absolute()
    impl_postfix_path = parent_folder.joinpath(
        "..", "configs", "impl_postfix.json"
    )
    legends = pd.read_json(impl_postfix_path)

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

    sql = sqlalchemy.select(*columns).group_by(
        dm.Result.benchmark,
        dm.Result.problem_preset,
    )
    if not run_id:
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

    sql = sql.where(dm.Result.run_id == run_id)

    df = pd.read_sql_query(
        sql=sql,
        con=conn.connect(),
    )

    print("Legend")
    print("======")
    print(legends)
    print("")
    print("Summary of current implementation")
    print("=================================")
    print(df.to_string())
