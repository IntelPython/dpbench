# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

add_subdirectory(black_scholes)
add_subdirectory(pairwise_distance)
add_subdirectory(l2_norm)
add_subdirectory(rambo)
add_subdirectory(kmeans)
add_subdirectory(knn)
add_subdirectory(gpairs)
add_subdirectory(dbscan)
add_subdirectory(rodinia)

# generate dpcpp version into config
set(FILE ${CMAKE_SOURCE_DIR}/dpbench/configs/framework_info/dpcpp.toml)
set(WRITE_VERSION "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

file(READ ${FILE} FILE_CONTENTS)
string(REGEX REPLACE "dpcpp_version = \"[^\"]*\""
"dpcpp_version = \"${WRITE_VERSION}\"" FILE_CONTENTS "${FILE_CONTENTS}")

file(WRITE ${FILE} "${FILE_CONTENTS}")
