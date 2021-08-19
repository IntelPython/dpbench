import numpy as np
try:
    import numpy.random_intel as rnd
    numpy_ver="Intel"
except:
    import numpy.random as rnd
    numpy_ver="regular"

SEED = 777777    
S0L = 10.0
S0H = 50.0
XL = 10.0
XH = 50.0
TL = 1.0
TH = 2.0

def dump_binary(price, strike, t):
    with open('price.bin', 'w') as fd:
        price.tofile(fd)

    with open('strike.bin', 'w') as fd:
        strike.tofile(fd)

    with open('t.bin', 'w') as fd:
        t.tofile(fd)

def dump_text(price, strike, t):
    with open('gen_price.txt', 'w') as fd:
        price.tofile(fd, '\n', '%s')

    with open('gen_strike.txt', 'w') as fd:
        strike.tofile(fd, '\n', '%s')

    with open('gen_t.txt', 'w') as fd:
        t.tofile(fd, '\n', '%s')
        
def gen_rand_data(nopt):
    rnd.seed(SEED)
    return (rnd.uniform(S0L, S0H, nopt),
            rnd.uniform(XL, XH, nopt),
            rnd.uniform(TL, TH, nopt))

def gen_data_to_file(nopt = 2**10):
    price, strike, t = gen_rand_data(nopt)
    dump_binary(price, strike, t)
    #dump_text(price, strike, t)
