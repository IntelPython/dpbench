# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22..3.27 FATAL_ERROR)

project(dpbench
    LANGUAGES CXX
    DESCRIPTION
        "Benchmark suite to evaluate Intel Data Parallel Extensions for Python"
)

# Help conda build find path from both host and build env.
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)

find_package(Python 3.9 REQUIRED
  COMPONENTS Interpreter Development.Module)

if(NOT DEFINED DPCTL_LIBRARY_PATH)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import dpctl,os; print(os.path.dirname(dpctl.__file__));"
        OUTPUT_VARIABLE DPCTL_LIBRARY_PATH
        RESULT_VARIABLE RET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(RET EQUAL "1")
        message(FATAL_ERROR "Module \'dpctl\' not found.")
    endif()
endif()

cmake_path(APPEND DPCTL_MODULE_PATH ${DPCTL_LIBRARY_PATH} resources cmake)
list(APPEND CMAKE_MODULE_PATH ${DPCTL_MODULE_PATH})

find_package(pybind11 CONFIG REQUIRED)
find_package(IntelSYCL REQUIRED)
find_package(Dpctl REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(dpbench)

set(ignoreMe "${SKBUILD}")
