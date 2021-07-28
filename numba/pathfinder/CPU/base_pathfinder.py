import numpy as np
import sys,json
import numpy.random as rnd
from timeit import default_timer as now

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

LWS = 2**10

###############################################

def gen_data(rows, cols):
    return (
        rnd.randint(LOW, HIGH, (rows, cols)),
        np.empty(cols)
    )

##############################################	

def run(name, alg, sizes=10, step=2, rows=2**10, cols=2**6, pyramid_height=20):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', required=False, default=sizes,  help="Number of steps")
    parser.add_argument('--step',  required=False, default=step,   help="Factor for each step")
    parser.add_argument('--rows',  required=False, default=rows,   help="Initial row size")
    parser.add_argument('--cols',  required=False, default=cols,   help="Initial column size")
    parser.add_argument('--pyht',  required=False, default=pyramid_height,   help="Initial pyramid height")
    parser.add_argument('--repeat',required=False, default=1,    help="Iterations inside measured region")
    parser.add_argument('--json',  required=False, default=__file__.replace('py','json'), help="output json data filename")
	
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
        data, result = gen_data(rows, cols)
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
