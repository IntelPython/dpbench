import options
import execute_implementations as ei
import plot_graphs as pg

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--run",
        type=options.run,
        required=False,
        default=options.run.all,
        choices=list(options.run),
    )
    parser.add_argument(
        "-i",
        "--impl",
        type=options.implementation,
        required=False,
        default=options.implementation.all,
        choices=list(options.implementation),
    )
    parser.add_argument(
        "-p",
        "--platform",
        type=options.platform,
        required=False,
        default=options.platform.all,
        choices=list(options.platform),
    )
    parser.add_argument(
        "-a",
        "--analysis",
        type=options.analysis,
        required=False,
        default=options.analysis.all,
        choices=list(options.analysis),
    )
    parser.add_argument("-k", "--kernel", required=False, action="store_true")
    parser.add_argument(
        "-c", "--comp_only", required=False, action="store_true"
    )
    parser.add_argument(
        "-ws",
        "--workloads",
        type=str,
        required=False,
        nargs="+",
        default=[],
        choices=[e.value for e in options.all_workloads],
    )
    args = parser.parse_args()

    # ******** Create Options class **********
    opts = options.options()
    opts.run = args.run
    opts.impl = args.impl
    opts.platform = args.platform
    opts.analysis = args.analysis
    opts.kernel = args.kernel
    opts.comp_only = args.comp_only
    opts.wls = options.workloads(args.workloads, opts.kernel, opts.comp_only)

    # print log of execution configuration
    # ****************************************

    # *************** RUN AND GENERATE DATA *********
    if opts.run == options.run.execute or opts.run == options.run.all:
        ei.run(opts)

    # *************** PLOT FROM DATA *********
    if opts.run == options.run.plot or opts.run == options.run.all:
        pg.run(opts)
