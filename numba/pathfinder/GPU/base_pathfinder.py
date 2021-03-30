import numpy as np
from random import seed, uniform
import sys
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

LWS = 2**8

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
	
    args = parser.parse_args()
    sizes= int(args.steps)
    step = int(args.step)
    rows = int(args.rows)
    cols= int(args.cols)
    pyramid_height = int(args.pyht)
    repeat=int(args.repeat)
    kwargs={}

    rnd.seed(SEED)
    f2 = open("runtimes.csv",'w',1)
    
    for i in xrange(sizes):
        data, result = gen_data(rows, cols)
        iterations = xrange(repeat)
        t0 = now()
        for _ in iterations:
            alg(data, rows, cols, pyramid_height, result)
        time = now() - t0
        
        if PRINT_DATA: print("AFTER KERNEL:\n **** data *******\n", data, "\n******* result *******\n",result)
        print("\nInput size:", rows, cols, pyramid_height, "Time:", time)
        f2.write(str(rows) + "," + str(time) + "\n")
        rows *= step
        repeat -= step
        if repeat < 1:
            repeat = 1

    f2.close()
