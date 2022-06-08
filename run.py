import dpbench.infrastructure as dpbi

fw_nb = dpbi.NumbaFramework("numba")
fw_np = dpbi.Framework("numpy")
bench = dpbi.Benchmark(bname="black_scholes")
test = dpbi.Test(bench=bench, frmwrk=fw_nb, npfrmwrk=fw_np)
test.run(preset="S", repeat=1, validate=True, timeout=10.0)
