# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import sqlalchemy

import dpbench.config as cfg
import dpbench.infrastructure as dpbi
from dpbench.infrastructure.enums import ErrorCodes
from dpbench.infrastructure.frameworks.framework import Framework


def _format_ns(time_in_ns):
    time = int(time_in_ns)
    assert time >= 0
    suff = [("s", 1000_000_000), ("ms", 1000_000), ("\u03BCs", 1000), ("ns", 0)]
    for s, scale in suff:
        if time >= scale:
            scaled_time = float(time) / scale if scale > 0 else time
            return f"{scaled_time}{s} ({time} ns)"


def _print_results(result: dpbi.BenchmarkResults, framework: Framework):
    print(
        "================ implementation "
        + result.impl_postfix
        + " ========================\n"
        + "implementation:",
        result.impl_postfix,
    )

    if result.error_state == ErrorCodes.SUCCESS:
        print("framework:", framework.fname)
        print("framework version:", framework.version())
        print("setup time:", _format_ns(result.setup_time))
        print("warmup time:", _format_ns(result.warmup_time))
        print("teardown time:", _format_ns(result.teardown_time))
        print("max execution times:", _format_ns(result.max_exec_time))
        print("min execution times:", _format_ns(result.min_exec_time))
        print("median execution times:", _format_ns(result.median_exec_time))
        print("repeats:", result.repeats)
        print("preset:", result.preset)
        print("validated:", result.validation_state)
    else:
        print("error states:", result.error_state)
        print("error msg:", result.error_msg)


def print_report(
    conn: sqlalchemy.Engine,
    run_id: int,
    implementations: set[str],
    comparison_pairs: list[tuple[str, str]] = [],
):
    if not implementations:
        implementations = {impl.postfix for impl in cfg.GLOBAL.implementations}

    implementations = list(implementations)
    implementations.sort()

    dpbi.generate_impl_summary_report(
        conn, run_id=run_id, implementations=implementations
    )

    dpbi.generate_performance_report(
        conn,
        run_id=run_id,
        implementations=implementations,
        headless=True,
    )

    dpbi.generate_comparison_report(
        conn,
        run_id=run_id,
        implementations=implementations,
        comparison_pairs=comparison_pairs,
        headless=True,
    )

    unexpected_failures = dpbi.get_unexpected_failures(conn, run_id=run_id)

    if len(unexpected_failures) > 0:
        raise ValueError(
            f"Unexpected benchmark implementations failed: {unexpected_failures}.",
        )
