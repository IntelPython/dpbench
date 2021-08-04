import numpy as np
import sys,json,os
import numpy.random as rnd
from timeit import default_timer as now
import dpctl, dpctl.memory as dpmem

######################################################
# GLOBAL DECLARATIONS THAT WILL BE USED IN ALL FILES #
######################################################

PRINT_DATA = 0

# make xrange available in python 3
try:
    xrange
except NameError:
    xrange = range

LOW = 0
HIGH = 10.0
SEED = 9
HALO = 1
STR_SIZE = 256
DEVICE = 0

LWS = 2**8

###############################################
def get_device_selector (is_gpu = True):
    if is_gpu is True:
        device_selector = "gpu"
    else:
        device_selector = "cpu"

    if os.environ.get('SYCL_DEVICE_FILTER') is None or os.environ.get('SYCL_DEVICE_FILTER') == "opencl":
        return "opencl:" + device_selector

    if os.environ.get('SYCL_DEVICE_FILTER') == "level_zero":
        return "level_zero:" + device_selector

    return os.environ.get('SYCL_DEVICE_FILTER')

def gen_data_np(rows, cols):
    return (
        rnd.randint(LOW, HIGH, (rows, cols)),
        np.empty(cols)
    )

def gen_data_usm(rows, cols):
    data_buf = rnd.randint(LOW, HIGH, (rows, cols), dtype=np.int32)
    result_buf = np.empty(cols, dtype=np.int32)

    with dpctl.device_context(get_device_selector()):
        data_usm = dpmem.MemoryUSMShared(rows*cols*np.dtype('i4').itemsize)
        result_usm = dpmem.MemoryUSMShared(cols*np.dtype('i4').itemsize)

        data_usm.copy_from_host(data_buf.reshape((-1)).view("u1"))
        result_usm.copy_from_host(result_buf.view("u1"))
    
    return (np.ndarray((rows,cols), buffer=data_usm, dtype='i4'),
            np.ndarray((cols), buffer=result_usm, dtype='i4')
    )

##############################################	

def run(name, alg, sizes=5, step=2, rows=2**10, cols=2**6, pyramid_height=20):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', required=False, default=sizes,  help="Number of steps")
    parser.add_argument('--step',  required=False, default=step,   help="Factor for each step")
    parser.add_argument('--rows',  required=False, default=rows,   help="Initial row size")
    parser.add_argument('--cols',  required=False, default=cols,   help="Initial column size")
    parser.add_argument('--pyht',  required=False, default=pyramid_height,   help="Initial pyramid height")
    parser.add_argument('--repeat',required=False, default=1,    help="Iterations inside measured region")
    parser.add_argument('--json',  required=False, default=__file__.replace('py','json'), help="output json data filename")
    parser.add_argument('--usm',   required=False, action='store_true',  help="Use USM Shared or pure numpy")
	
    args = parser.parse_args()
    sizes= int(args.steps)
    step = int(args.step)
    rows = int(args.rows)
    cols= int(args.cols)
    pyramid_height = int(args.pyht)
    repeat=int(args.repeat)
    kwargs={}
 
    output = {}
    output['name']      = name
    output['sizes']     = sizes
    output['step']      = step
    output['repeat']    = repeat
    output['randseed']  = SEED
    output['metrics']   = []

    rnd.seed(SEED)
    f2 = open("runtimes.csv",'w',1)
    
    for i in xrange(sizes):
        if args.usm is True:
            data, result = gen_data_usm(rows, cols)
        else:
            data, result = gen_data_np(rows, cols)

        alg(data, rows, cols, pyramid_height, result)
        
        iterations = xrange(repeat)
        t0 = now()
        for _ in iterations:
            alg(data, rows, cols, pyramid_height, result)
        time = now() - t0
        
        f2.write(str(rows) + "," + str(time) + "\n")
        rows *= step
        mops = 0.
        nopt = 0
        print("ERF: {:15s} | Size: {:10d} | MOPS: {:15.2f} | TIME: {:10.6f}".format(name, nopt, mops,time),flush=True)
        output['metrics'].append((nopt,mops,time))
        repeat -= step
        if repeat < 1:
            repeat = 1
    json.dump(output,open(args.json,'w'),indent=2, sort_keys=True)

    f2.close()
