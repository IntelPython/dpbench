# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

set(module_name gaussian_sycl)
set(py_module_name _${module_name})
python_add_library(${py_module_name} MODULE ${module_name}/${py_module_name}.cpp)
add_sycl_to_target(TARGET ${py_module_name} SOURCES ${module_name}/${py_module_name}.cpp)
target_include_directories(${py_module_name} PRIVATE ${Dpctl_INCLUDE_DIRS})

file(RELATIVE_PATH py_module_dest ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS ${py_module_name}
  DESTINATION ${py_module_dest}/${module_name}
)
