# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache 2.0

"""The module generates reports for implementation summary and timing summary
from a specific benchmark run.
"""

from . import datamodel as dm

__all__ = [
    "generate_impl_summary_report",
]


# Note the space before N/A is deliberate so that the MAX operator always
# picks the error_state rather than N/A.
_sql_latest_implementation_summary = """
SELECT
    MAX(timestamp) as As_of,
    benchmark,
    problem_preset,
    CASE
        WHEN MAX(input_size) < 1024 THEN MAX(input_size) || 'B'
        WHEN MAX(input_size) >=  1024 AND MAX(input_size) < (1024 * 1024) THEN (MAX(input_size) / 1024) || 'KB'
        WHEN MAX(input_size) >= (1024 * 1024)  AND MAX(input_size) < (1024 * 1024 * 1024) THEN (MAX(input_size) / (1024 * 1024)) || 'MB'
        WHEN MAX(input_size) >= (1024 * 1024 * 1024) AND MAX(input_size) < (1024 * 1024 * 1024 *1024) THEN (MAX(input_size) / (1024 * 1024 * 1024)) || 'GB'
        WHEN MAX(input_size) >= (1024 * 1024 * 1024 * 1024) THEN (MAX(input_size) / (1024 * 1024 * 1024 * 1024)) || 'TB'
    END AS input_size,
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


def generate_impl_summary_report(results_db):
    import pathlib

    import pandas as pd

    conn = dm.create_connection(db_file=results_db)

    parent_folder = pathlib.Path(__file__).parent.absolute()
    impl_postfix_path = parent_folder.joinpath(
        "..", "configs", "impl_postfix.json"
    )
    legends = pd.read_json(impl_postfix_path)
    df = pd.read_sql_query(_sql_latest_implementation_summary, conn)

    print("Legend")
    print("======")
    print(legends)
    print("")
    print("Summary of current implementation")
    print("=================================")
    print(df.to_string())
