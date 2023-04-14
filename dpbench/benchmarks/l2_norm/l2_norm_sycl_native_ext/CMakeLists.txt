# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

set(py_module_name _l2_norm_sycl)
pybind11_add_module(${py_module_name}
    MODULE
    l2_norm_sycl/_l2_norm_sycl.cpp
)
target_include_directories(${py_module_name} PUBLIC ${Dpctl_INCLUDE_DIRS})

file(RELATIVE_PATH py_module_dest ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS ${py_module_name}
  DESTINATION ${py_module_dest}/l2_norm_sycl
)
