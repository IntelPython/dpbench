# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.23)

set(py_module_name _deformable_convolution_sycl)
pybind11_add_module(${py_module_name}
    MODULE
    deformable_convolution_sycl/impl.cpp
)

find_package(TBB CONFIG REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/cmake NO_DEFAULT_PATH)
find_package(IntelSYCL CONFIG REQUIRED)

if (DEFINED ENV{CONDA_PREFIX})
  set(MKL_ROOT $ENV{CONDA_PREFIX})
endif()
find_package(MKL CONFIG REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/cmake NO_DEFAULT_PATH)

target_compile_options(${py_module_name} PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(${py_module_name} PUBLIC ${Dpctl_INCLUDE_DIRS} $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(${py_module_name} PUBLIC $<LINK_ONLY:MKL::MKL> ${MKL_SYCL})

file(RELATIVE_PATH py_module_dest ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS ${py_module_name}
  DESTINATION ${py_module_dest}/deformable_convolution_sycl
)
