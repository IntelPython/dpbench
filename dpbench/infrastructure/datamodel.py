# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache 2.0

import hashlib
import sqlite3


def _generate_primary_key(sval: str):
    return int(
        hashlib.sha224(sval).hexdigest(),
        16,
    )


_sql_create_results_table = """
CREATE TABLE IF NOT EXISTS results (
    id integer PRIMARY KEY,
    timestamp integer NOT NULL,
    benchmark text NOT NULL,
    implementation text NOT NULL,
    platform text NOT NULL,
    framework_version text NOT NULL,
    error_state integer NOT NULL,
    problem_preset text,
    setup_time real,
    warmup_time real,
    repeats integer,
    min_exec_time real,
    max_exec_time real,
    median_exec_time real,
    quartile25_exec_time real,
    quartile75_exec_time real,
    teardown_time real,
    validated integer
);
"""

_sql_insert_into_results_table = """
INSERT INTO results(
    id,
    timestamp,
    benchmark,
    implementation,
    platform,
    framework_version,
    CASE error_state
        WHEN -1 THEN "Unimplemented"
        WHEN -2 THEN "Framework unavailable"
        WHEN -3 THEN "Failed Execution"
        WHEN -4 THEN "Failed Validation"
        ELSE "SUCCESS"
    END status,
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
    CASE validated
        WHEN 0 THEN "PASS"
        ELSE "FAIL"
    END validation_state
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
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


def store_results(conn, result):
    data = []
    data.append(
        _generate_primary_key(
            result.benchmark_name + result.benchmark_impl_postfix
        )
    )
    data.append(result.benchmark_name)
    data.append(result.benchmark_impl_postfix)
    data.append("TODO")
    data.append(result.framework_name + " " + result.framework_version)
    data.append(result.error_state)
    data.append(result.preset)
    data.append(result.num_repeats)
    data.append(result.setup_time)
    data.append(result.warmup_time)
    data.append(result.min_exec_time)
    data.append(result.max_exec_time)
    data.append(result.median_exec_time)
    data.append(result.quartile25_exec_time)
    data.append(result.quartile75_exec_time)
    data.append(result.teardown_time)
    data.append(result.validation_state)

    cur = conn.cursor()
    cur.execute(_sql_insert_into_results_table, data)
    conn.commit()
