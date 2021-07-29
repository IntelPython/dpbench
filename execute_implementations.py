import os
import shutil
import sys

import options
import util


def run_native_optimised_CPU(app_name, cmds, analysis):
    # cd to cpu folder
    # if perf run
    # run perf
    # if vtune run
    # run vtune
    # if advisor run
    # run advisor
    if not util.chdir("CPU"):
        print("Native Optimized CPU version of " + str(app_name) + " not available")
        return

    # compile
    clean_string = ["make", "clean"]
    util.run_command(clean_string, verbose=False)

    build_string = ["make"]
    util.run_command(build_string, verbose=False)

    if analysis == options.analysis.test:
        run_cmd = cmds['NATIVE_OPTIMISED_TEST_CMD']
        util.run_command(run_cmd, verbose=True)

    if analysis == options.analysis.vtune or analysis == options.analysis.all:
        shutil.rmtree('vtune_dir', ignore_errors=True)
        run_cmd = options.VTUNE_THREADING_CMD + cmds['NATIVE_OPTIMISED_VTUNE_CMD']
        util.run_command(run_cmd, verbose=True)

    if analysis == options.analysis.advisor or analysis == options.analysis.all:
        shutil.rmtree('roofline', ignore_errors=True)
        run_cmd = options.ADVISOR_SURVEY_CMD + cmds['NATIVE_OPTIMISED_ADVISOR_CMD']
        util.run_command(run_cmd, verbose=True)

        run_cmd = options.ADVISOR_FLOP_CMD + cmds['NATIVE_OPTIMISED_ADVISOR_CMD']
        util.run_command(run_cmd, verbose=True)

        run_cmd = options.ADVISOR_ROOFLINE_CMD
        util.run_command(run_cmd, verbose=True)

    if analysis == options.analysis.perf or analysis == options.analysis.all:
        run_cmd = cmds['NATIVE_OPTIMISED_PERF_CMD']
        util.run_command(run_cmd, verbose=True)

    clean_string = ["make", "clean"]
    util.run_command(clean_string, verbose=False)


def run_native_optimised_GPU(app_name, cmds, analysis):
    # cd to cpu folder
    # if perf run
    # run perf
    # if vtune run
    # run vtune
    # if advisor run
    # run advisor
    if not util.chdir("GPU"):
        print("Optimized Native GPU version of " + str(app_name) + " not available")
        return

    # compile
    clean_string = ["make", "clean"]
    util.run_command(clean_string, verbose=False)

    build_string = ["make"]
    util.run_command(build_string, verbose=False)

    if analysis == options.analysis.test:
        run_cmd = cmds['NATIVE_OPTIMISED_TEST_CMD']
        util.run_command(run_cmd, verbose=True)

    if analysis == options.analysis.vtune or analysis == options.analysis.all:
        shutil.rmtree('vtune_dir', ignore_errors=True)
        run_cmd = options.VTUNE_GPU_OFFLOAD_CMD + cmds['NATIVE_OPTIMISED_VTUNE_CMD']
        util.run_command(run_cmd, verbose=True)
        shutil.rmtree('vtune_hotspots_dir', ignore_errors=True)
        run_cmd = options.VTUNE_GPU_HOTSPOTS_CMD + cmds['NATIVE_OPTIMISED_VTUNE_CMD']
        util.run_command(run_cmd, verbose=True)

    if analysis == options.analysis.advisor or analysis == options.analysis.all:
        shutil.rmtree('roofline', ignore_errors=True)
        run_cmd = options.ADVISOR_GPU_SURVEY_CMD + cmds['NATIVE_OPTIMISED_ADVISOR_CMD']
        util.run_command(run_cmd, verbose=True)

        run_cmd = options.ADVISOR_GPU_FLOP_CMD + cmds['NATIVE_OPTIMISED_ADVISOR_CMD']
        util.run_command(run_cmd, verbose=True)

        run_cmd = options.ADVISOR_GPU_ROOFLINE_CMD
        util.run_command(run_cmd, verbose=True)

        try:
            run_cmd = options.ADVISOR_GPU_METRICS_CMD
            util.run_command(run_cmd, verbose=True, filename="GPU_Metrics.txt")
        except:
            print("Failed to generate Advisor GPU Metrics")

    if analysis == options.analysis.perf or analysis == options.analysis.all:
        run_cmd = cmds['NATIVE_OPTIMISED_PERF_CMD']
        util.run_command(run_cmd, verbose=True)

    clean_string = ["make", "clean"]
    util.run_command(clean_string, verbose=False)


def run_native_CPU(app_name, cmds, analysis):
    # cd to cpu folder
    # if perf run
    # run perf
    # if vtune run
    # run vtune
    # if advisor run
    # run advisor
    if not util.chdir("CPU"):
        print("Native CPU version of " + str(app_name) + " not available")
        return

    # compile
    clean_string = ["make", "clean"]
    util.run_command(clean_string, verbose=False)

    build_string = ["make"]
    util.run_command(build_string, verbose=False)

    if analysis == options.analysis.test:
        run_cmd = cmds['NATIVE_TEST_CMD']
        util.run_command(run_cmd, verbose=True)

    if analysis == options.analysis.vtune or analysis == options.analysis.all:
        shutil.rmtree('vtune_dir', ignore_errors=True)
        run_cmd = options.VTUNE_THREADING_CMD + cmds['NATIVE_VTUNE_CMD']
        util.run_command(run_cmd, verbose=True)

    if analysis == options.analysis.advisor or analysis == options.analysis.all:
        shutil.rmtree('roofline', ignore_errors=True)
        run_cmd = options.ADVISOR_SURVEY_CMD + cmds['NATIVE_ADVISOR_CMD']
        util.run_command(run_cmd, verbose=True)

        run_cmd = options.ADVISOR_FLOP_CMD + cmds['NATIVE_ADVISOR_CMD']
        util.run_command(run_cmd, verbose=True)

        run_cmd = options.ADVISOR_ROOFLINE_CMD
        util.run_command(run_cmd, verbose=True)

    if analysis == options.analysis.perf or analysis == options.analysis.all:
        run_cmd = cmds['NATIVE_PERF_CMD']
        util.run_command(run_cmd, verbose=True)

    clean_string = ["make", "clean"]
    util.run_command(clean_string, verbose=False)


def run_native_GPU(app_name, cmds, analysis):
    # cd to cpu folder
    # if perf run
    # run perf
    # if vtune run
    # run vtune
    # if advisor run
    # run advisor
    if not util.chdir("GPU"):
        print("Native GPU version of " + str(app_name) + " not available")
        return

    # compile
    clean_string = ["make", "clean"]
    util.run_command(clean_string, verbose=False)

    build_string = ["make"]
    util.run_command(build_string, verbose=False)

    if analysis == options.analysis.test:
        run_cmd = cmds['NATIVE_TEST_CMD']
        util.run_command(run_cmd, verbose=True)

    if analysis == options.analysis.vtune or analysis == options.analysis.all:
        shutil.rmtree('vtune_dir', ignore_errors=True)
        run_cmd = options.VTUNE_GPU_OFFLOAD_CMD + cmds['NATIVE_VTUNE_CMD']
        util.run_command(run_cmd, verbose=True)
        shutil.rmtree('vtune_hotspots_dir', ignore_errors=True)
        run_cmd = options.VTUNE_GPU_HOTSPOTS_CMD + cmds['NATIVE_VTUNE_CMD']
        util.run_command(run_cmd, verbose=True)

    if analysis == options.analysis.advisor or analysis == options.analysis.all:
        shutil.rmtree('roofline', ignore_errors=True)
        run_cmd = options.ADVISOR_GPU_SURVEY_CMD + cmds['NATIVE_ADVISOR_CMD']
        util.run_command(run_cmd, verbose=True)

        run_cmd = options.ADVISOR_GPU_FLOP_CMD + cmds['NATIVE_ADVISOR_CMD']
        util.run_command(run_cmd, verbose=True)

        run_cmd = options.ADVISOR_GPU_ROOFLINE_CMD
        util.run_command(run_cmd, verbose=True)

        try:
            run_cmd = options.ADVISOR_GPU_METRICS_CMD
            util.run_command(run_cmd, verbose=True, filename="GPU_Metrics.txt")
        except:
            print("Failed to generate Advisor GPU Metrics")        

    if analysis == options.analysis.perf or analysis == options.analysis.all:
        run_cmd = cmds['NATIVE_PERF_CMD']
        util.run_command(run_cmd, verbose=True)

    clean_string = ["make", "clean"]
    util.run_command(clean_string, verbose=False)


def run_numba_CPU(app_name, cmds, analysis):
    if not util.chdir("CPU"):
        print("Numba CPU version of " + str(app_name) + " not available")
        return

    if analysis == options.analysis.test:
        run_cmd = cmds['NUMBA_CPU_TEST_CMD']
        util.run_command(run_cmd, verbose=True)

    if analysis == options.analysis.vtune or analysis == options.analysis.all:
        shutil.rmtree('vtune_dir', ignore_errors=True)
        run_cmd = options.VTUNE_THREADING_CMD + cmds['NUMBA_CPU_VTUNE_CMD']
        util.run_command(run_cmd, verbose=True)

    if analysis == options.analysis.advisor or analysis == options.analysis.all:
        shutil.rmtree('roofline', ignore_errors=True)
        run_cmd = options.ADVISOR_SURVEY_CMD + cmds['NUMBA_CPU_ADVISOR_CMD']
        util.run_command(run_cmd, verbose=True)

        run_cmd = options.ADVISOR_FLOP_CMD + cmds['NUMBA_CPU_ADVISOR_CMD']
        util.run_command(run_cmd, verbose=True)

        run_cmd = options.ADVISOR_ROOFLINE_CMD
        util.run_command(run_cmd, verbose=True)

    if analysis == options.analysis.perf or analysis == options.analysis.all:
        run_cmd = cmds['NUMBA_CPU_PERF_CMD']
        util.run_command(run_cmd, verbose=True)

    shutil.rmtree('__pycache__', ignore_errors=True)


def run_numba_GPU(app_name, cmds, analysis):
    if not util.chdir("GPU"):
        print("Numba GPU version of " + str(app_name) + " not available")
        return

    if analysis == options.analysis.test:
        run_cmd = cmds['NUMBA_TEST_CMD']
        util.run_command(run_cmd, verbose=True)

    if analysis == options.analysis.vtune or analysis == options.analysis.all:
        shutil.rmtree('vtune_dir', ignore_errors=True)
        run_cmd = options.VTUNE_GPU_OFFLOAD_CMD + cmds['NUMBA_VTUNE_CMD']
        util.run_command(run_cmd, verbose=True)
        shutil.rmtree('vtune_hotspots_dir', ignore_errors=True)
        run_cmd = options.VTUNE_GPU_HOTSPOTS_CMD + cmds['NUMBA_VTUNE_CMD']
        util.run_command(run_cmd, verbose=True)        

    if analysis == options.analysis.advisor or analysis == options.analysis.all:
        shutil.rmtree('roofline', ignore_errors=True)
        run_cmd = options.ADVISOR_GPU_SURVEY_CMD + cmds['NUMBA_ADVISOR_CMD']
        util.run_command(run_cmd, verbose=True)

        run_cmd = options.ADVISOR_GPU_FLOP_CMD + cmds['NUMBA_ADVISOR_CMD']
        util.run_command(run_cmd, verbose=True)

        run_cmd = options.ADVISOR_GPU_ROOFLINE_CMD
        util.run_command(run_cmd, verbose=True)

        try:
            run_cmd = options.ADVISOR_GPU_METRICS_CMD
            util.run_command(run_cmd, verbose=True, filename="GPU_Metrics.txt")
        except:
            print("Failed to generate Advisor GPU Metrics")            

    if analysis == options.analysis.perf or analysis == options.analysis.all:
        run_cmd = cmds['NUMBA_PERF_CMD']
        util.run_command(run_cmd, verbose=True)

    shutil.rmtree('__pycache__', ignore_errors=True)


def run_scikit_learn_CPU(app_name, cmds, analysis):
    if not util.chdir("CPU"):
        print("SKLearn CPU version of " + str(app_name) + " not available")
        return
    
    if analysis == options.analysis.test:
        run_cmd = cmds['SCIKIT_LEARN_TEST_CMD']
        util.run_command(run_cmd, verbose=True)

    if analysis == options.analysis.vtune or analysis == options.analysis.all:
        shutil.rmtree('vtune_dir', ignore_errors=True)
        run_cmd = options.VTUNE_THREADING_CMD + cmds['SCIKIT_LEARN_VTUNE_CMD']
        util.run_command(run_cmd, verbose=True)

    if analysis == options.analysis.advisor or analysis == options.analysis.all:
        shutil.rmtree('roofline', ignore_errors=True)
        run_cmd = options.ADVISOR_SURVEY_CMD + cmds['SCIKIT_LEARN_ADVISOR_CMD']
        util.run_command(run_cmd, verbose=True)

        run_cmd = options.ADVISOR_FLOP_CMD + cmds['SCIKIT_LEARN_ADVISOR_CMD']
        util.run_command(run_cmd, verbose=True)

        run_cmd = options.ADVISOR_ROOFLINE_CMD
        util.run_command(run_cmd, verbose=True)

    if analysis == options.analysis.perf or analysis == options.analysis.all:
        run_cmd = cmds['SCIKIT_LEARN_PERF_CMD']
        util.run_command(run_cmd, verbose=True)

    shutil.rmtree('__pycache__', ignore_errors=True)


def run_daal4py_CPU(app_name, cmds, analysis):
    if not util.chdir("CPU"):
        print("Daal4py CPU version of " + str(app_name) + " not available")
        return

    if analysis == options.analysis.test:
        run_cmd = cmds['DAAL4PY_TEST_CMD']
        util.run_command(run_cmd, verbose=True)

    if analysis == options.analysis.vtune or analysis == options.analysis.all:
        shutil.rmtree('vtune_dir', ignore_errors=True)
        run_cmd = options.VTUNE_THREADING_CMD + cmds['DAAL4PY_VTUNE_CMD']
        util.run_command(run_cmd, verbose=True)

    if analysis == options.analysis.advisor or analysis == options.analysis.all:
        shutil.rmtree('roofline', ignore_errors=True)
        run_cmd = options.ADVISOR_SURVEY_CMD + cmds['DAAL4PY_ADVISOR_CMD']
        util.run_command(run_cmd, verbose=True)

        run_cmd = options.ADVISOR_FLOP_CMD + cmds['DAAL4PY_ADVISOR_CMD']
        util.run_command(run_cmd, verbose=True)

        run_cmd = options.ADVISOR_ROOFLINE_CMD
        util.run_command(run_cmd, verbose=True)

    if analysis == options.analysis.perf or analysis == options.analysis.all:
        run_cmd = cmds['DAAL4PY_PERF_CMD']
        util.run_command(run_cmd, verbose=True)

    shutil.rmtree('__pycache__', ignore_errors=True)


def run_native_optimised(opts):
    # cd to native_optimised folder
    # loop over list of applications
    # if option is CPU or all
    # run native_optimised CPU
    # if option is GPU or all
    # run native_optimised GPU
    # cd back to root folder
    util.chdir("native_optimised")
        
    native_optimised_dir = os.getcwd()

    for app, cmds in opts.wls.wl_list.items():
        if cmds['execute'] and util.chdir(app):
            app_dir = os.getcwd()
            if opts.platform == options.platform.cpu or opts.platform == options.platform.all:
                run_native_optimised_CPU(app, cmds, opts.analysis)
                util.chdir(app_dir)

            if opts.platform == options.platform.gpu or opts.platform == options.platform.all:
                run_native_optimised_GPU(app, cmds, opts.analysis)

            util.chdir(native_optimised_dir)


def run_native(opts):
    # cd to native folder
    # loop over list of applications
    # if option is CPU or all
    # run native CPU
    # if option is GPU or all
    # run native GPU
    # cd back to root folder
    util.chdir("native")

    native_dir = os.getcwd()

    for app, cmds in opts.wls.wl_list.items():
        if cmds['execute'] and util.chdir(app):
            app_dir = os.getcwd()
            if opts.platform == options.platform.cpu or opts.platform == options.platform.all:
                run_native_CPU(app, cmds, opts.analysis)
                util.chdir(app_dir)

            if opts.platform == options.platform.gpu or opts.platform == options.platform.all:
                run_native_GPU(app, cmds, opts.analysis)

            util.chdir(native_dir)

def run_native_dpcpp(opts):
    # cd to native dpcpp folder
    # loop over list of applications
    # if option is CPU or all
    # run native CPU
    # if option is GPU or all
    # run native GPU
    # cd back to root folder
    util.chdir("native_dpcpp")

    native_dir = os.getcwd()

    for app, cmds in opts.wls.wl_list.items():
        if cmds['execute'] and util.chdir(app):
            app_dir = os.getcwd()
            if opts.platform == options.platform.cpu or opts.platform == options.platform.all:
                run_native_CPU(app, cmds, opts.analysis)
                util.chdir(app_dir)

            if opts.platform == options.platform.gpu or opts.platform == options.platform.all:
                run_native_GPU(app, cmds, opts.analysis)

            util.chdir(native_dir)            

def run_numba(opts):
    # cd to numba folder
    # loop over list of applications
    # if option is CPU or all
    # run native CPU
    # if option is GPU or all
    # run native GPU
    # cd back to root folder
    util.chdir("numba")

    numba_dir = os.getcwd()

    for app, cmds in opts.wls.wl_list.items():
        if cmds['execute'] and util.chdir(app):
            app_dir = os.getcwd()
            if opts.platform == options.platform.cpu or opts.platform == options.platform.all:
                run_numba_CPU(app, cmds, opts.analysis)
                util.chdir(app_dir)

            if opts.platform == options.platform.gpu or opts.platform == options.platform.all:
                run_numba_GPU(app, cmds, opts.analysis)

            util.chdir(numba_dir)

def run_dpnp(opts):
    # cd to dpnp folder
    # loop over list of applications
    # if option is CPU or all
    # run native CPU
    # if option is GPU or all
    # run native GPU
    # cd back to root folder
    util.chdir("dpnp")

    numba_dir = os.getcwd()

    for app, cmds in opts.wls.wl_list.items():
        if cmds['execute'] and util.chdir(app):
            app_dir = os.getcwd()
            if opts.platform == options.platform.cpu or opts.platform == options.platform.all:
                run_numba_CPU(app, cmds, opts.analysis)
                util.chdir(app_dir)

            if opts.platform == options.platform.gpu or opts.platform == options.platform.all:
                run_numba_GPU(app, cmds, opts.analysis)

            util.chdir(numba_dir)
            

def run_scikit_learn(opts):
    # cd to scikit_learn folder
    # loop over list of applications
    # if option is CPU or all
    # run native CPU
    # cd back to root folder
    util.chdir("scikit_learn")

    scikit_learn_dir = os.getcwd()

    for app, cmds in opts.wls.wl_list.items():
        if cmds['execute'] and util.chdir(app):
            app_dir = os.getcwd()
            if opts.platform == options.platform.cpu or opts.platform == options.platform.all:
                run_scikit_learn_CPU(app, cmds, opts.analysis)
                util.chdir(app_dir)

            util.chdir(scikit_learn_dir)


def run_daal4py(opts):
    # cd to daal4py folder
    # loop over list of applications
    # if option is CPU or all
    # run native CPU
    # cd back to root folder
    util.chdir("daal4py")

    daal4py_dir = os.getcwd()

    for app, cmds in opts.wls.wl_list.items():
        if cmds['execute'] and util.chdir(app):
            app_dir = os.getcwd()
            if opts.platform == options.platform.cpu or opts.platform == options.platform.all:
                run_daal4py_CPU(app, cmds, opts.analysis)
                util.chdir(app_dir)

            util.chdir(daal4py_dir)


def check_envvars_tools(opts):
    # if numba check if conda env has numba installed

    # if native check if icc is available

    # if vtune or advisor
    # check if vtune is avaiable check if advisor is available
    # check numba profiling env is set. if not set it.
    # if GPU profiling set GPU profiling env

    # TODO: Fix hang when below code is enabled
    # if  opts.impl == options.implementation.numba or opts.impl == options.implementation.all:
    #     try:
    #         import numba
    #     except:
    #         print("Numba not available. Exiting\n")
    #         sys.exit()
    #     if opts.platform == options.platform.gpu or opts.platform == options.platform.all:
    #         try:
    #             from numba import dppy
    #         except:
    #             print("Numba DPPY not available. Exiting\n")
    #             sys.exit()

    from shutil import which
    if opts.impl == options.implementation.native or opts.impl == options.implementation.all:
        if which("icx") is None:
            print("ICX compiler is required to run native implementations. Exiting\n")
            sys.exit()

        if opts.platform == options.platform.gpu or opts.platform == options.platform.all:
            if os.environ.get('LIBOMPTARGET_DEVICETYPE') is None:
                os.environ['LIBOMPTARGET_DEVICETYPE'] = 'gpu'
            print(os.environ['LIBOMPTARGET_DEVICETYPE'])

    if opts.analysis == options.analysis.vtune or opts.analysis == options.analysis.all:
        if which("vtune") is None:
            print(
                "Intel VTune Profiler not available. Install Intel OpenAPI Base Toolkit and run the setvars.sh script in Intel OpenAPI Base Toolkit.\n")
            sys.exit()

    if opts.analysis == options.analysis.advisor or opts.analysis == options.analysis.all:
        if which("advixe-cl") is None:
            print(
                "Intel Advisor is not available. Install Intel OpenAPI Base Toolkit and run the setvars.sh script in Intel OpenAPI Base Toolkit.\n")
            sys.exit()

    if opts.analysis == options.analysis.advisor or opts.analysis == options.analysis.vtune or opts.analysis == options.analysis.all:
        if os.environ.get('NUMBA_ENABLE_PROFILING') is None:
            os.environ['NUMBA_ENABLE_PROFILING'] = '1'
        print(os.environ['NUMBA_ENABLE_PROFILING'])

        if os.environ.get('ZE_ENABLE_API_TRACING') is None:
            os.environ['ZE_ENABLE_API_TRACING'] = '1'
        print(os.environ['ZE_ENABLE_API_TRACING'])
        
        if opts.platform == options.platform.gpu or opts.platform == options.platform.all:
            if os.environ.get('ADVIXE_EXPERIMENTAL') is None:
                os.environ['ADVIXE_EXPERIMENTAL'] = 'gpu-profiling'
            print(os.environ['ADVIXE_EXPERIMENTAL'])


def run(opts):
    # loop over impls
    # if impl is native_optimised and option selected is native_optimised or all
    # run native_optimised
    # if impl is native and option selected is native or all
    # run native
    # if impl is numba option selected is numba or all
    # run numba

    # check if all required env variables are set
    check_envvars_tools(opts)

    ref_cwd = os.getcwd()

    if opts.impl == options.implementation.native or opts.impl == options.implementation.all:
        run_native(opts)
        util.chdir(ref_cwd)        

    if opts.impl == options.implementation.numba or opts.impl == options.implementation.all:
        run_numba(opts)
        util.chdir(ref_cwd)

    if opts.impl == options.implementation.native_dpcpp:
        run_native_dpcpp(opts)
        util.chdir(ref_cwd)

    if opts.impl == options.implementation.dpnp:
        run_dpnp(opts)
        util.chdir(ref_cwd)        
        
    if opts.impl == options.implementation.scikit_learn:
        run_scikit_learn(opts)
        util.chdir(ref_cwd)

    if opts.impl == options.implementation.daal4py:
        run_daal4py(opts)
        util.chdir(ref_cwd)

    if opts.impl == options.implementation.native_optimised:
        run_native_optimised(opts)
        util.chdir(ref_cwd)        
