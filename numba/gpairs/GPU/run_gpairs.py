import base_gpairs
from device_selector import get_device_selector
import dpctl
import os

backend = os.getenv("NUMBA_BACKEND", "legacy")

if backend == "legacy":
    from numba_dppy import kernel, atomic, DEFAULT_LOCAL_SIZE
    import numba_dppy
    atomic_add = atomic.add
else:
    from numba_dpcomp.mlir.kernel_impl import kernel, atomic, DEFAULT_LOCAL_SIZE
    import numba_dpcomp.mlir.kernel_impl as numba_dppy # this doesn't work for dppy if no explicit numba_dppy before get_global_id(0)
    atomic_add = atomic.add

@kernel
def count_weighted_pairs_3d_intel(
    x1, y1, z1, w1, x2, y2, z2, w2, rbins_squared, result
):
    """Naively count Npairs(<r), the total number of pairs that are separated
    by a distance less than r, for each r**2 in the input rbins_squared.
    """

    start = numba_dppy.get_global_id(0)
    stride = numba_dppy.get_global_size(0)

    n1 = x1.shape[0]
    n2 = x2.shape[0]
    nbins = rbins_squared.shape[0]

    for i in range(start, n1, stride):
        px = x1[i]
        py = y1[i]
        pz = z1[i]
        pw = w1[i]
        for j in range(n2):
            qx = x2[j]
            qy = y2[j]
            qz = z2[j]
            qw = w2[j]
            dx = px - qx
            dy = py - qy
            dz = pz - qz
            wprod = pw * qw
            dsq = dx * dx + dy * dy + dz * dz

            k = nbins - 1
            while dsq <= rbins_squared[k]:
                atomic_add(result, k - 1, wprod)
                k = k - 1
                if k <= 0:
                    break


@kernel
def count_weighted_pairs_3d_intel_ver2(
    x1, y1, z1, w1, x2, y2, z2, w2, rbins_squared, result
):
    """Naively count Npairs(<r), the total number of pairs that are separated
    by a distance less than r, for each r**2 in the input rbins_squared.
    """

    i = numba_dppy.get_global_id(0)
    nbins = rbins_squared.shape[0]
    n2 = x2.shape[0]

    px = x1[i]
    py = y1[i]
    pz = z1[i]
    pw = w1[i]
    for j in range(n2):
        qx = x2[j]
        qy = y2[j]
        qz = z2[j]
        qw = w2[j]
        dx = px - qx
        dy = py - qy
        dz = pz - qz
        wprod = pw * qw
        dsq = dx * dx + dy * dy + dz * dz

        k = nbins - 1
        while dsq <= rbins_squared[k]:
            # disabled for now since it's not supported currently
            # - could reenable later when it's supported (~April 2020)
            # - could work around this to avoid atomics, which would perform better anyway
            # cuda.atomic.add(result, k-1, wprod)
            atomic_add(result, k - 1, wprod)
            k = k - 1
            if k <= 0:
                break

def run_gpairs(d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2, d_rbins_squared, d_result):
    with dpctl.device_context(get_device_selector()):
        count_weighted_pairs_3d_intel[d_x1.shape[0], DEFAULT_LOCAL_SIZE](
            d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2, d_rbins_squared, d_result
        )

base_gpairs.run("Gpairs Dppy kernel", run_gpairs)
