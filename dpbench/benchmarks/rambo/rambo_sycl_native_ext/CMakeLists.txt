
set(py_module_name _rambo_sycl)
pybind11_add_module(${py_module_name}
    MODULE
    rambo_sycl/_rambo_sycl.cpp
)
target_include_directories(${py_module_name} PUBLIC ${Dpctl_INCLUDE_DIRS})

file(RELATIVE_PATH py_module_dest ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS ${py_module_name}
  DESTINATION ${py_module_dest}/rambo_sycl
)
