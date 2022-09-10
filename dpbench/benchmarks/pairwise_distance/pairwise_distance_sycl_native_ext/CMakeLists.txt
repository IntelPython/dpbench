
set(py_module_name _pairwise_distance_sycl)
pybind11_add_module(${py_module_name}
    MODULE
    pairwise_distance_sycl/_pairwise_distance_sycl.cpp
)
target_include_directories(${py_module_name} PUBLIC ${Dpctl_INCLUDE_DIRS})

file(RELATIVE_PATH py_module_dest ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS ${py_module_name}
  DESTINATION ${py_module_dest}/pairwise_distance_sycl
)
