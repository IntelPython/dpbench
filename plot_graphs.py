import util
import options
import os, sys

try:
    import pandas as pd
except:
    print("Pandas not available\n")

#TODO: Normalize peak performance of CPU and GPU

def run_native_CPU(app_name, cmds):
    util.chdir("CPU")

    df = pd.read_csv('perf_output.csv',names=['input_size','throughput'], index_col='input_size')

    # this is needed for setting the layout to show complete figure
    #from matplotlib import rcParams
    #rcParams.update({'figure.autolayout': True})
    
    bar_chart = df.plot.bar(legend=False,rot=45,fontsize=10)
    #bar_chart.set(xlabel='Input size', ylabel='Thoughput in input elements Processed per second')
    bar_chart.set_ylabel('Thoughput in input elements processed per second',fontsize=10)
    bar_chart.set_xlabel('Input size',fontsize=10)
    fig = bar_chart.get_figure()
    fig_filename = str(app_name) + "_native_CPU_performance.pdf"
    fig.savefig(fig_filename,bbox_inches="tight")

    #print(df)
    return df.loc[cmds['ref_input'],'throughput']

def plot_native_GPU(app_name, cmds):
    util.chdir("GPU")

    df = pd.read_csv('perf_output.csv',names=['input_size','throughput'], index_col='input_size')

    # this is needed for setting the layout to show complete figure
    #from matplotlib import rcParams
    #rcParams.update({'figure.autolayout': True})
    
    bar_chart = df.plot.bar(legend=False,rot=45,fontsize=10)
    #bar_chart.set(xlabel='Input size', ylabel='Thoughput in input elements Processed per second')
    bar_chart.set_ylabel('Thoughput in input elements processed per second',fontsize=10)
    bar_chart.set_xlabel('Input size',fontsize=10)
    fig = bar_chart.get_figure()
    fig_filename = str(app_name) + "_native_GPU_performance.pdf"
    fig.savefig(fig_filename,bbox_inches="tight")

    #print(df)
    return df.loc[cmds['ref_input'],'throughput']

def plot_numba_CPU(app_name, cmds):
    util.chdir("CPU")

    df = pd.read_csv('perf_output.csv',names=['input_size','throughput'], index_col='input_size')

    # this is needed for setting the layout to show complete figure
    #from matplotlib import rcParams
    #rcParams.update({'figure.autolayout': True})
    
    bar_chart = df.plot.bar(legend=False,rot=45,fontsize=10)
    #bar_chart.set(xlabel='Input size', ylabel='Thoughput in input elements Processed per second')
    bar_chart.set_ylabel('Thoughput in input elements processed per second',fontsize=10)
    bar_chart.set_xlabel('Input size',fontsize=10)
    fig = bar_chart.get_figure()
    fig_filename = str(app_name) + "_numba_CPU_performance.pdf"
    fig.savefig(fig_filename,bbox_inches="tight")

    #print(df)
    return df.loc[cmds['ref_input'],'throughput']
        
def plot_numba_GPU(app_name, cmds):
    util.chdir("GPU")

    df = pd.read_csv('perf_output.csv',names=['input_size','throughput'], index_col='input_size')

    # this is needed for setting the layout to show complete figure
    #from matplotlib import rcParams
    #rcParams.update({'figure.autolayout': True})
    
    bar_chart = df.plot.bar(legend=False,rot=45,fontsize=10)
    #bar_chart.set(xlabel='Input size', ylabel='Thoughput in input elements Processed per second')
    bar_chart.set_ylabel('Thoughput in input elements processed per second',fontsize=10)
    bar_chart.set_xlabel('Input size',fontsize=10)
    fig = bar_chart.get_figure()
    fig_filename = str(app_name) + "_numba_GPU_performance.pdf"
    fig.savefig(fig_filename,bbox_inches="tight")

    #print(df)
    return df.loc[cmds['ref_input'],'throughput']    
    
def plot_native(opts, all_plot_data):
    util.chdir("native")

    native_dir = os.getcwd();
    
    for app, cmds in opts.wls.wl_list.items():
        if cmds['execute'] is True:

            plot_data_entry = {}
            if app in all_plot_data:
                plot_data_entry = all_plot_data[app]                
                
            util.chdir(app)
            app_dir = os.getcwd();
            if opts.platform == options.platform.cpu or opts.platform == options.platform.all:
                cpu_perf = plot_native_CPU(app, cmds)

                plot_data_entry['native_cpu'] = cpu_perf
                util.chdir(app_dir)
                
            if opts.platform == options.platform.gpu or opts.platform == options.platform.all:
                gpu_perf = plot_native_GPU(app, cmds)
                plot_data_entry['native_gpu'] = gpu_perf
                
            util.chdir(numba_dir)
            all_plot_data[app] = plot_data_entry    

def plot_numba(opts, all_plot_data):
    util.chdir("numba")

    numba_dir = os.getcwd();
    
    for app, cmds in opts.wls.wl_list.items():
        if cmds['execute'] is True:

            plot_data_entry = {}
            if app in all_plot_data:
                plot_data_entry = all_plot_data[app]                
                
            util.chdir(app)
            app_dir = os.getcwd();
            if opts.platform == options.platform.cpu or opts.platform == options.platform.all:
                cpu_perf = plot_numba_CPU(app, cmds)

                plot_data_entry['numba_cpu'] = cpu_perf
                util.chdir(app_dir)
                
            if opts.platform == options.platform.gpu or opts.platform == options.platform.all:
                gpu_perf = plot_numba_GPU(app, cmds)
                plot_data_entry['numba_gpu'] = gpu_perf
                
            util.chdir(numba_dir)
            all_plot_data[app] = plot_data_entry

def check_envvars_tools(opts):
    if opts.analysis is not options.analysis.all and opts.analysis is not options.analysis.perf:
        print("Plotting can be run only with option --analysis(-a) set to all or perf. Exiting")
        sys.exit()

    try:
        import pandas
    except:
        print ("Pandas not available. Plotting disabled\n")
        sys.exit()
    
def run(opts):
    check_envvars_tools(opts)
    
    ref_cwd = os.getcwd();
    
    all_plot_data = {}
    
    if opts.impl == options.implementation.native or opts.impl == options.implementation.all:
        plot_native(opts, all_plot_data)
        util.chdir(ref_cwd)
        
    if  opts.impl == options.implementation.numba or opts.impl == options.implementation.all:
        plot_numba(opts, all_plot_data)
        util.chdir(ref_cwd)

    #print (all_plot_data)

    df = pd.DataFrame.from_dict(all_plot_data, orient='index')

    #print(df)

    bar_chart = df.plot.bar(rot=45,fontsize=10)
    bar_chart.set_ylabel('Thoughput in input elements processed per second',fontsize=10)
    bar_chart.set_xlabel('Benchmark',fontsize=10)
    fig = bar_chart.get_figure()
    fig_filename = "benchmarks_performance.pdf"
    fig.savefig(fig_filename,bbox_inches="tight")    
