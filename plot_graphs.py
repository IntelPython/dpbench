import os
import sys

import utils as util

import options

try:
    import pandas as pd
except:
    print("Pandas not available\n")


def get_runtime_data(app_name, cmds, platform):
    util.chdir(platform)

    try:
        df = pd.read_csv(
            "runtimes.csv",
            names=["input_size", "runtime"],
            index_col="input_size",
        )
        return df.loc[cmds["ref_input"], "runtime"]
    except:
        print("Runtimes not available for " + app_name + "\n")
        return 0


def get_runtimes(opts, all_plot_data, impl):
    util.chdir(impl)

    numba_dir = os.getcwd()

    for app, cmds in opts.wls.wl_list.items():
        if cmds["execute"] is True:

            plot_data_entry = {}
            if app in all_plot_data:
                plot_data_entry = all_plot_data[app]

            util.chdir(app)
            app_dir = os.getcwd()
            if (
                opts.platform == options.platform.cpu
                or opts.platform == options.platform.all
            ):
                cpu_perf = get_runtime_data(app, cmds, "CPU")
                if cpu_perf is not 0:
                    plot_data_entry[impl + "_cpu"] = cpu_perf
                util.chdir(app_dir)

            if (
                opts.platform == options.platform.gpu
                or opts.platform == options.platform.all
            ):
                gpu_perf = get_runtime_data(app, cmds, "GPU")
                if gpu_perf is not 0:
                    plot_data_entry[impl + "_gpu"] = gpu_perf

            util.chdir(numba_dir)
            all_plot_data[app] = plot_data_entry


def check_envvars_tools(opts):
    if (
        opts.analysis is not options.analysis.all
        and opts.analysis is not options.analysis.perf
    ):
        print(
            "Plotting can be run only with option --analysis(-a) set to all or perf. Exiting"
        )
        sys.exit()

    try:
        import pandas
    except:
        print("Pandas not available. Plotting disabled\n")
        sys.exit()


def plot_efficiency_graph(all_plot_data):
    df = pd.DataFrame.from_dict(all_plot_data, orient="index")
    plot = False

    try:
        df["CPU"] = (df["numba_cpu"] / df["native_cpu"]) * 100.00
        plot = True
        df.drop(columns=["native_cpu", "numba_cpu"], inplace=True)
    except:
        print("CPU Efficiency data not available\n")

    try:
        df["GPU"] = (df["numba_gpu"] / df["native_gpu"]) * 100.00
        plot = True
        df.drop(columns=["native_gpu", "numba_gpu"], inplace=True)
    except:
        print("GPU Efficiency data not available\n")

    if plot:
        # df.drop(columns=['native_cpu', 'native_gpu', 'numba_cpu', 'numba_gpu'], inplace=True)

        bar_chart = df.plot.bar(rot=45, fontsize=10)
        # bar_chart.legend(loc='upper right')
        bar_chart.set_ylabel("Efficiency in percentage", fontsize=10)
        bar_chart.set_xlabel("Benchmark", fontsize=10)
        bar_chart.set_title(
            "Efficiency of Numba execution relative to OpenMP execution on CPU and GPU",
            fontsize=10,
        )
        fig = bar_chart.get_figure()
        fig_filename = "Efficiency_graph.pdf"
        fig.savefig(fig_filename, bbox_inches="tight")
    else:
        print(
            "Insufficient data to generate Efficiency graph. Verify execution times in runtimes.csv\n"
        )


def plot_speedup_graph(all_plot_data):
    df = pd.DataFrame.from_dict(all_plot_data, orient="index")
    plot = False

    try:
        df["OpenMP"] = (df["native_cpu"] / df["native_gpu"]) * 100.00
        plot = True
        df.drop(columns=["native_cpu", "native_gpu"], inplace=True)
    except:
        print("CPU Speedup data not available\n")

    try:
        df["Numba"] = (df["numba_cpu"] / df["numba_gpu"]) * 100.00
        plot = True
        df.drop(columns=["numba_cpu", "numba_gpu"], inplace=True)
    except:
        print("GPU Speedup data not available\n")

    if plot:
        # df.drop(columns=['native_cpu', 'native_gpu', 'numba_cpu', 'numba_gpu'], inplace=True)

        bar_chart = df.plot.bar(rot=45, fontsize=10)
        # bar_chart.legend(loc='upper right')
        bar_chart.set_ylabel("Speedup in percentage", fontsize=10)
        bar_chart.set_xlabel("Benchmark", fontsize=10)
        bar_chart.set_title(
            "Speedup of GPU execution over CPU execution for Numba and OpenMP",
            fontsize=10,
        )
        fig = bar_chart.get_figure()
        fig_filename = "Speedup_graph.pdf"
        fig.savefig(fig_filename, bbox_inches="tight")
    else:
        print(
            "Insufficient data to generate Speedup graph. Verify execution times in runtimes.csv\n"
        )


def run(opts):
    check_envvars_tools(opts)

    ref_cwd = os.getcwd()

    all_plot_data = {}

    if (
        opts.impl == options.implementation.native
        or opts.impl == options.implementation.all
    ):
        get_runtimes(opts, all_plot_data, "native")
        util.chdir(ref_cwd)

    if (
        opts.impl == options.implementation.numba
        or opts.impl == options.implementation.all
    ):
        get_runtimes(opts, all_plot_data, "numba")
        util.chdir(ref_cwd)

    plot_efficiency_graph(all_plot_data)
    plot_speedup_graph(all_plot_data)
