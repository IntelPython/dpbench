# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache 2.0

import sqlite3
from sys import implementation

_sql_create_results_table = """
CREATE TABLE IF NOT EXISTS results (
    timestamp integer NOT NULL,
    benchmark text NOT NULL,
    implementation text NOT NULL,
    platform text NOT NULL,
    framework_version text NOT NULL,
    error_state text NOT NULL,
    problem_preset text,
    setup_time real,
    warmup_time real,
    repeats text,
    min_exec_time real,
    max_exec_time real,
    median_exec_time real,
    quartile25_exec_time real,
    quartile75_exec_time real,
    teardown_time real,
    validated text,
    PRIMARY KEY (timestamp, benchmark, implementation)
);
"""

_sql_insert_into_results_table = """
INSERT INTO results(
    timestamp,
    benchmark,
    implementation,
    platform,
    framework_version,
    error_state,
    problem_preset,
    repeats,
    setup_time,
    warmup_time,
    min_exec_time,
    max_exec_time,
    median_exec_time,
    quartile25_exec_time,
    quartile75_exec_time,
    teardown_time,
    validated
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

# Note the space before N/A is deliberate so that the MAX operator always
# picks the error_state rather than N/A.
_sql_latest_implementation_summary = """
SELECT
    MAX(timestamp) as As_of,
    benchmark,
    problem_preset as problem_size,
    MAX(
        CASE
            WHEN implementation == "numba_dpex_k" THEN error_state
            ELSE " N/A"
        END
    ) as numba_dpex_k,
    MAX(
        CASE
            WHEN implementation == "numba_dpex_p" THEN error_state
            ELSE " N/A"
        END
    ) as numba_dpex_p,
    MAX(
        CASE
            WHEN implementation == "numba_dpex_n" THEN error_state
            ELSE " N/A"
        END
    ) as numba_dpex_n,
    MAX(
        CASE
            WHEN implementation == "dpnp" THEN error_state
            ELSE " N/A"
        END
    ) as dpnp,
    MAX(
        CASE
            WHEN implementation == "numpy" THEN error_state
            ELSE " N/A"
        END
    ) as numpy,
    MAX(
        CASE
            WHEN implementation == "python" THEN error_state
            ELSE " N/A"
        END
    ) as python,
    MAX(
        CASE
            WHEN implementation == "numba_n" THEN error_state
            ELSE " N/A"
        END
    ) as numba_n,
    MAX(
        CASE
            WHEN implementation == "numba_np" THEN error_state
            ELSE " N/A"
        END
    ) as numba_np,
    MAX(
        CASE
            WHEN implementation == "numba_npr" THEN error_state
            ELSE " N/A"
        END
    ) as numba_npr,
    MAX(
        CASE
            WHEN implementation == "sycl" THEN error_state
            ELSE " N/A"
        END
    ) as dpcpp
    FROM results
    GROUP BY benchmark, problem_preset;
"""


def create_connection(db_file) -> sqlite3.Connection:
    """create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(e)

    return conn


def create_results_table(conn):
    """create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(_sql_create_results_table)
    except sqlite3.Error as e:
        print(e)


def store_results(conn, result, run_timestamp):
    data = []

    data.append(run_timestamp)
    data.append(result.benchmark_name)
    data.append(result.benchmark_impl_postfix)
    data.append("TODO")
    data.append(result.framework_name + " " + result.framework_version)

    if result.error_state == -1:
        error_state_str = "Unimplemented"
    elif result.error_state == -2:
        error_state_str = "Framework unavailable"
    elif result.error_state == -3:
        error_state_str = "Failed Execution"
    elif result.error_state == -4:
        error_state_str = "Failed Validation"
    else:
        error_state_str = "Success"

    data.append(error_state_str)
    data.append(result.preset)
    data.append(str(result.num_repeats))
    data.append(result.setup_time)
    data.append(result.warmup_time)
    data.append(result.min_exec_time)
    data.append(result.max_exec_time)
    data.append(result.median_exec_time)
    data.append(result.quartile25_exec_time)
    data.append(result.quartile75_exec_time)
    data.append(result.teardown_time)

    if result.validation_state == 0:
        validation_str = "Success"
    else:
        validation_str = "Fail"
    data.append(validation_str)

    cur = conn.cursor()
    cur.execute(_sql_insert_into_results_table, data)
    conn.commit()


def print_implementation_summary(conn):
    import pandas as pd

    df = pd.read_sql_query(_sql_latest_implementation_summary, conn)
    print(df.to_string())
