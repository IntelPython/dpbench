import base_pathfinder
import numba_dppy
import dpctl
from numba import int64 as local_dtype

IN_RANGE = lambda x, min, max: ((x)>=(min) and (x)<=(max))
#CLAMP_RANGE = lambda x, min, max: (x = min if (x<(min)) else (max if (x>(max)) else x))
MIN = lambda a,b: ((a) if (a)<=(b) else (b))

@numba_dppy.func
def in_range(x, min, max):
    return ((x)>=(min) and (x)<=(max))

@numba_dppy.func
def min_dppy(a,b):
    return ((a) if (a)<=(b) else (b))

@numba_dppy.kernel
def pathfinder_kernel(gpuWall, gpuSrc, gpuResult, iteration, borderCols, cols, t):
    
    BLOCK_SIZE = numba_dppy.get_local_size(0);

    prev = numba_dppy.local.array(shape=2**10, dtype=local_dtype)
    result = numba_dppy.local.array(shape=2**10, dtype=local_dtype)
    
    bx = numba_dppy.get_group_id(0)
    tx = numba_dppy.get_local_id(0)

    ## Each block finally computes result for a small block after N iterations.
    ## it is the non-overlapping small blocks that cover all the input data

    ## calculate the small block size.
    small_block_cols = BLOCK_SIZE - (iteration*base_pathfinder.HALO*2)

    ## calculate the boundary for the block according to the boundary of its small block
    blkX = (small_block_cols*bx) - borderCols
    blkXmax = blkX+BLOCK_SIZE-1

    ## calculate the global thread coordination
    xidx = blkX+tx

    ## effective range within this block that falls within the valid range of the input data used to rule out computation outside the boundary.
    validXmin = -blkX if (blkX < 0) else 0
    validXmax = BLOCK_SIZE-1-(blkXmax-cols+1) if (blkXmax > cols-1) else BLOCK_SIZE-1

    W = tx-1
    E = tx+1

    W = validXmin if (W < validXmin) else W
    E = validXmax if (E > validXmax) else E

    isValid = in_range(tx, validXmin, validXmax)

    if (in_range(xidx, 0, cols-1)):
        prev[tx] = gpuSrc[xidx]

    numba_dppy.barrier(numba_dppy.CLK_LOCAL_MEM_FENCE)

    computed = False

    for i in range(iteration):
        computed = False

        if (in_range(tx, i+1, BLOCK_SIZE-i-2) and isValid):
            computed = True
            left = prev[W]
            up = prev[tx]
            right = prev[E]
            shortest = min_dppy(left, up)
            #index = cols*(t+i)+xidx
            result[tx] = shortest + gpuWall[t+i,xidx]

            ## add debugging info to the debug output buffer...
            #if (tx==11 and i==0):
                ## set bufIndex to what value/range of values you want to know.
                #bufIndex = gpuSrc[xidx]
                #outputBuffer[bufIndex] = 1

        numba_dppy.barrier(numba_dppy.CLK_LOCAL_MEM_FENCE)

        if (i == iteration - 1):
            break

        if computed is True:
            prev[tx] = result[tx] ## Assign the computation range

        numba_dppy.barrier(numba_dppy.CLK_LOCAL_MEM_FENCE)

    ## update the global memory
    ## after the last iteration, only threads coordinated within the
    ## small block perform the calculation and switch on "computed"
    if computed is True:
        gpuResult[xidx] = result[tx]

def run_pathfinder(data, rows, cols, pyramid_height, result):

    borderCols = pyramid_height * base_pathfinder.HALO
    
    #create a temp list that hold first row of data as first element and empty numpy array as second element
    gpu_result_list = []
    gpu_result_list.append(data[0]) #first row
    gpu_result_list.append(result)
    src = 1
    final_ret = 0
    for t in range(0, rows-1, pyramid_height):
        temp = src
        src = final_ret
        final_ret = temp

        iteration = MIN(pyramid_height, rows-t-1)

        with dpctl.device_context("opencl:cpu"):
            # invoke kernel with data - all rows except first row
            pathfinder_kernel[rows*cols, base_pathfinder.LWS](data[1:rows,:], gpu_result_list[src], gpu_result_list[final_ret], iteration, borderCols, cols, t)

base_pathfinder.run("Numba pathfinder", run_pathfinder)
