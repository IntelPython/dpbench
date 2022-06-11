
set(py_module_name _black_scholes_sycl)
pybind11_add_module(${py_module_name}
    MODULE
    black_scholes_sycl/_black_scholes_sycl.cpp
)
target_include_directories(${py_module_name} PUBLIC ${Dpctl_INCLUDE_DIRS})
install(TARGETS ${py_module_name}
  DESTINATION black_scholes_sycl
)
