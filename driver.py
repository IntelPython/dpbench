import pkgutil

import dpbench.benchmarks as dp_bms
import dpbench.benchmarks.npbench.npbench.benchmarks as npb
import dpbench.benchmarks.npbench.npbench.benchmarks.polybench as pb
from dpbench.runner import (
    list_available_benchmarks,
    list_possible_implementations,
    run_benchmark,
)

if __name__ == "__main__":
    bms = list_available_benchmarks()
    print(bms)

    npbms = list_available_benchmarks(npb)
    print(npbms)

    pbbms = list_available_benchmarks(pb)
    print(pbbms)

    impl = list_possible_implementations()
    print(impl)

    run_benchmark("black_scholes")
