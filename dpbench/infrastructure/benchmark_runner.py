# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
"""Provides infrastructure to run benchmarks."""

import inspect
import logging
import multiprocessing as mp
import multiprocessing.connection as mpc
from dataclasses import dataclass
from typing import Any

import sqlalchemy

import dpbench.config as cfg
from dpbench.infrastructure.benchmark import Benchmark
from dpbench.infrastructure.benchmark_results import BenchmarkResults
from dpbench.infrastructure.benchmark_validation import validate_results
from dpbench.infrastructure.datamodel import store_results
from dpbench.infrastructure.enums import ErrorCodes, ValidationStatusCodes
from dpbench.infrastructure.frameworks import Framework
from dpbench.infrastructure.frameworks.fabric import build_framework
from dpbench.infrastructure.timer import timer

"""
Send on process creation:
    Framework configuration: so the process can set it up.

Send on benchmark run request:
    Benchmark configuration

Send on benchmark finish:
    Benchmark execution results
"""


@dataclass
class BaseRunConfig:
    """Run configuration that is required to run benchmark."""

    benchmark: cfg.Benchmark
    implementation: str
    ref_framework: cfg.Framework = None
    preset: str = "S"
    repeat: int = 10
    validate: bool = True
    precision: str = None
    print_results: bool = True

    @classmethod
    def from_instance(cls, instance):
        """Creates BaseRunConfiguration from it's inheritor."""
        return cls(
            **{
                k: v
                for k, v in instance.__dict__.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class RunConfig(BaseRunConfig):
    """Extended run configuration that controls benchmark execution."""

    conn: sqlalchemy.Engine = None
    framework: cfg.Framework = None
    timeout: float = 200.0
    run_id: int = None
    skip_expected_failures: bool = False


class BenchmarkRunner:
    """Benchmark runner that delegates runs between processes.

    It creates process for each framework to avoid conflicts in framework
    imports.
    """

    def __init__(
        self,
        method: str = "spawn",
    ) -> None:
        """Creates BenchmarkRunner. No processes get spawn at this point.

        Args:
            method: method for sub process creations. Currently only spawn is
                supported. We need to get rid of GLOBAL config to get support
                for 'fork' method.
        """
        self._ctx = mp.get_context(method)
        self._framework_processes: dict[
            str, tuple[mp.Process, mpc.Connection]
        ] = {}

    def get_process(
        self, framework: cfg.Framework
    ) -> tuple[mp.Process, mpc.Connection]:
        """Get process with connection to it using caching.

        Caching is working the way that if the process does not exists it will
        be created automatically. Framework comparison is done by
        framework.simple_name key.

        Args:
            framework: framework to which return the process.

        Returns: tuple of process and connection pip to control it.
        """
        p, conn = self._framework_processes.get(
            framework.simple_name, (None, None)
        )
        if not p:
            return self.create_process(framework)
        return p, conn

    def kill_process(self, framework: cfg.Framework) -> None:
        """Kill the process for framework and closes connection.

        Args:
            framework: framework to which kill the process.
        """
        logging.info(f"Killing process for {framework.simple_name}")
        p, conn = self._framework_processes.get(
            framework.simple_name, (None, None)
        )
        if not p:
            return
        p.kill()
        p.join()
        conn.close()
        del self._framework_processes[framework.simple_name]

    def create_process(
        self, framework: cfg.Framework
    ) -> tuple[mp.Process, mpc.Connection]:
        """Create a process and updates cache for it.

        Args:
            framework: framework to which create the process.
        """
        logging.info(f"Creating new process for {framework.simple_name}")
        parent_conn, child_conn = self._ctx.Pipe()
        p = self._ctx.Process(target=BenchmarkRunner.runner, args=(child_conn,))
        self._framework_processes[framework.simple_name] = (p, parent_conn)

        p.start()

        parent_conn.send(logging.root.level)
        parent_conn.send(framework)
        parent_conn.send(cfg.GLOBAL.dtypes)

        return (p, parent_conn)

    def close_connections(self):
        """Closes all opened connections and processes."""
        for framework_name, proc in self._framework_processes.items():
            p, c = proc

            logging.info(f"Closing connection to {framework_name}")
            c.close()
            p.join()

    @staticmethod
    def runner(c: mpc.Connection) -> None:
        """Static method that is the root for new process."""
        # Starting from here it is an equivalent of running this method from
        # clean environment, so we have to set up all the GlOBAL inputs.

        # TODO: get rid of GLOBAL configs. We need to refactor all the
        #   infrastructure, that it avoids global variable usage. Instead we
        #   should pass configs in constructors.
        logging.root.setLevel(c.recv())

        logging.info("Running new process")

        framework_config: cfg.Framework = c.recv()
        cfg.GLOBAL.dtypes = c.recv()
        logging.info(f"Setting up the framework {framework_config.simple_name}")

        framework = build_framework(framework_config)

        cfg.GLOBAL.frameworks = [framework_config]

        while True:
            try:
                rc: BaseRunConfig = c.recv()
            except EOFError:
                logging.info("Exiting process")
                break

            rc.benchmark.implementations = [
                impl
                for impl in rc.benchmark.implementations
                if impl.postfix
                in {
                    rc.implementation,
                    rc.benchmark.reference_implementation_postfix,
                }
            ]

            if rc.ref_framework:
                cfg.GLOBAL.frameworks.append(rc.ref_framework)
            cfg.GLOBAL.benchmarks = [rc.benchmark]

            if logging.root.level <= logging.DEBUG:
                import pprint

                logging.debug(
                    f"Sub process configuration {pprint.pformat(cfg.GLOBAL)}"
                )

            logging.info(
                f"Running benchmark {rc.benchmark.short_name} for {rc.implementation}, {rc.preset}"
            )
            benchmark_results, _ = BenchmarkRunner.run_benchmark(
                rc,
                framework,
            )

            benchmark_results.print(framework.fname, framework.version())

            c.send(benchmark_results)

            if rc.ref_framework:
                cfg.GLOBAL.frameworks = cfg.GLOBAL.frameworks[:-1]

    @staticmethod
    def run_benchmark(
        rc: BaseRunConfig,
        framework: Framework,
    ) -> tuple[BenchmarkResults, dict]:
        """Static method to run benchmark.

        framework: framework that should be used during execution.
        """
        logging.info(
            f"Running {rc.benchmark.module_name} on {framework.fname} ({type(framework)})"
        )
        bench = Benchmark(rc.benchmark)
        bench.initialize_input_data(rc.preset, rc.precision)

        results = BenchmarkResults(rc.repeat, rc.implementation, rc.preset)

        if not framework:
            results.error_state = ErrorCodes.NO_FRAMEWORK
            results.error_msg = "No framework"
            return (results, {})

        results_dict = dict()
        _exec(
            bench,
            framework,
            rc.implementation,
            rc.preset,
            rc.repeat,
            results_dict,
            # copy output if we want to validate results
            rc.validate,
        )

        results.error_state = results_dict.get(
            "error_state", ErrorCodes.FAILED_EXECUTION
        )
        results.error_msg = results_dict.get("error_msg", "Unexpected crash")
        results.input_size = results_dict.get("input_size")
        if results.error_state != ErrorCodes.SUCCESS:
            return (results, {})

        output = results_dict.get("outputs", {})
        results.setup_time = results_dict["setup_time"]
        results.warmup_time = results_dict["warmup_time"]
        results.exec_times = results_dict["exec_times"]
        results.teardown_time = results_dict["teardown_time"]

        if rc.validate and results.error_state == ErrorCodes.SUCCESS:
            ref_framework = build_framework(rc.ref_framework)
            ref_output = _exec_simple(
                bench,
                ref_framework,
                rc.benchmark.reference_implementation_postfix,
                rc.preset,
            )
            if validate_results(ref_output, output):
                results.validation_state = ValidationStatusCodes.SUCCESS
            else:
                results.validation_state = ValidationStatusCodes.FAILURE
                results.error_state = ErrorCodes.FAILED_VALIDATION
                results.error_msg = "Validation failed"

        return (results, output)

    def run_benchmark_in_sub_process(
        self,
        rc: RunConfig,
    ) -> BenchmarkResults:
        """Runs benchmark in sub process.

        Args:
            rc: running configuration.

        The method blocks workflow and waits for the result, but executes it in
        separate process.
        """
        if (
            rc.skip_expected_failures
            and rc.implementation
            in rc.benchmark.expected_failure_implementations
        ):
            results = BenchmarkResults(0, rc.implementation, rc.preset)
            results.error_state = ErrorCodes.FAILED_EXECUTION
            results.error_msg = "Expected failure"

            results.print()

            return results

        if rc.implementation not in {
            b.postfix for b in rc.benchmark.implementations
        }:
            results = BenchmarkResults(0, rc.implementation, rc.preset)
            results.error_state = ErrorCodes.UNIMPLEMENTED
            results.error_msg = "Unimplemented"

            results.print()

            return results

        _, conn = self.get_process(rc.framework)

        brc = BaseRunConfig.from_instance(rc)

        brc.ref_framework: cfg.Framework = [
            f
            for f in cfg.GLOBAL.frameworks
            if rc.benchmark.reference_implementation_postfix
            in {p.postfix for p in f.postfixes}
        ][0]

        conn.send(brc)

        if conn.poll(rc.timeout if rc.timeout else self._default_timeout):
            try:
                results: BenchmarkResults = conn.recv()
            except EOFError:
                results = BenchmarkResults(0, rc.implementation, rc.preset)
                results.error_state = ErrorCodes.FAILED_EXECUTION
                results.error_msg = "Core dump"

                results.print()
                self.kill_process(rc.framework)
        else:
            results = BenchmarkResults(0, rc.implementation, rc.preset)
            results.error_state = ErrorCodes.EXECUTION_TIMEOUT
            results.error_msg = "Execution timed out"

            results.print()
            self.kill_process(rc.framework)

        return results

    def run_benchmark_and_save(
        self,
        rc: RunConfig,
    ):
        """Runs benchmarks and saves the result into database.

        Saves result if connection to database was provided.

        Args:
            rc: runtime configuration.
        """
        results = self.run_benchmark_in_sub_process(rc)

        if rc.conn:
            framework = build_framework(rc.framework)

            store_results(
                rc.conn,
                results.Result(
                    run_id=rc.run_id,
                    benchmark_name=rc.benchmark.module_name,
                    framework_version=framework.fname
                    + " "
                    + framework.version()
                    if framework
                    and results.error_state != ErrorCodes.UNIMPLEMENTED
                    else "n/a",
                ),
            )


def _set_input_args(
    bench: Benchmark, framework: Framework, np_input_data: dict
):
    inputs = dict()

    for arg in bench.info.input_args:
        if arg in bench.info.array_args:
            inputs[arg] = framework.copy_to_func()(np_input_data[arg])
        else:
            inputs[arg] = np_input_data[arg]

    return inputs


def _reset_output_args(
    bench: Benchmark, framework: Framework, inputs: dict, np_input_data: dict
):
    for arg in bench.info.output_args:
        overwritten_data = inputs.get(arg, None)
        if overwritten_data is None or arg not in bench.info.array_args:
            continue
        inputs[arg] = framework.copy_to_func()(np_input_data[arg])


def _array_size(array: Any) -> int:
    try:
        return array.nbytes
    except AttributeError:
        return array.size * array.itemsize


def _exec_simple(
    bench: Benchmark,
    framework: Framework,
    impl_postfix: str,
    preset: str,
):
    np_input_data = bench.get_input_data(preset=preset)
    inputs = _set_input_args(bench, framework, np_input_data)
    impl_fn = bench.get_implementation(impl_postfix)

    try:
        retval = framework.execute(impl_fn, inputs)
        results_dict = {}

        _exec_copy_output(bench, framework, retval, inputs, results_dict)

        return results_dict["outputs"]
    except Exception:
        logging.exception("Benchmark execution failed at the warmup step.")
        return None


def _exec(
    bench: Benchmark,
    framework: Framework,
    impl_postfix: str,
    preset: str,
    repeat: int,
    results_dict: dict,
    copy_output: bool,
):
    """Executes a benchmark for a given implementation.

    A helper function to execute a benchmark. The function is called in a
    separate sub-process by a BenchmarkRunner instance. The ``_exec`` function
    first runs the benchmark implementation function once as a warmup and then
    performs the specified number of repetitions. The output results are reset
    before each repetition and the final output is serialized into a npz
    (compressed NumPy data file) file.

    All timing results and the path to the serialized results are written to
    the results_dict input argument that is managed by the calling process.

    Args:
        bench : A Benchmark object representing the benchmark to be executed.
        framework : A Framework for which the benchmark is to be executed.
        impl_postfix : The identifier for the benchmark implementation.
        preset : A problem size entry defined in the bench_info JSON.
        timeout : Number of seconds after which the execution is killed.
        repeat : Number of repetitions of the benchmark execution.
        precision: The precision to use for benchmark input data.
        args : Input arguments to benchmark implementation function.
        results_dict : A dictionary where timing and other results are stored.
        copy_output : A flag that controls copying output.
    """
    np_input_data = bench.get_input_data(preset=preset)

    with timer() as t:
        inputs = _set_input_args(bench, framework, np_input_data)
    results_dict["setup_time"] = t.get_elapsed_time()

    input_size = 0
    for arg in bench.info.array_args:
        input_size += _array_size(bench.bdata[preset][arg])

    results_dict["input_size"] = input_size

    impl_fn = bench.get_implementation(impl_postfix)

    # Warmup
    with timer() as t:
        try:
            framework.execute(impl_fn, inputs)
        except Exception:
            logging.exception("Benchmark execution failed at the warmup step.")
            results_dict["error_state"] = ErrorCodes.FAILED_EXECUTION
            results_dict["error_msg"] = "Execution failed"
            return

    results_dict["warmup_time"] = t.get_elapsed_time()

    _reset_output_args(bench, framework, inputs, np_input_data)

    exec_times = [0] * repeat

    retval = None
    for i in range(repeat):
        with timer() as t:
            retval = framework.execute(impl_fn, inputs)
        exec_times[i] = t.get_elapsed_time()

        # Do not reset the output from the last repeat
        if i < repeat - 1:
            _reset_output_args(bench, framework, inputs, np_input_data)

    results_dict["exec_times"] = exec_times

    # Get the output data
    results_dict["teardown_time"] = 0.0
    if copy_output:
        _exec_copy_output(bench, framework, retval, inputs, results_dict)

    results_dict["error_state"] = ErrorCodes.SUCCESS
    results_dict["error_msg"] = ""


def _exec_copy_output(
    bench: Benchmark,
    fmwrk: Framework,
    retval,
    inputs: dict,
    results_dict: dict,
):
    output_arrays = dict()
    with timer() as t:
        for out_arg in bench.info.output_args:
            if out_arg in bench.info.array_args:
                output_arrays[out_arg] = fmwrk.copy_from_func()(inputs[out_arg])

    # Special case: if the benchmark implementation returns anything, then
    # add that to the results dict
    if retval is not None:
        output_arrays["return-value"] = convert_to_numpy(retval, fmwrk)

    results_dict["outputs"] = output_arrays
    results_dict["teardown_time"] = t.get_elapsed_time()


def convert_to_numpy(value: any, fmwrk: Framework) -> any:
    """Calls copy_from_func on all array values."""
    if isinstance(value, tuple):
        retval_list = list(value)
        for i, _ in enumerate(retval_list):
            retval_list[i] = fmwrk.copy_from_func()(retval_list[i])
        value = tuple(retval_list)
    else:
        value = fmwrk.copy_from_func()(value)

    return value
