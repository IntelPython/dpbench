add_subdirectory(black_scholes)
add_subdirectory(pairwise_distance)
add_subdirectory(l2_norm)
add_subdirectory(rambo)
add_subdirectory(knn)
add_subdirectory(gpairs)

# generate dpcpp version into json
file(WRITE
    ${CMAKE_SOURCE_DIR}/dpbench/configs/framework_info/dpcpp_version.json
    "{\"dpcpp_version\" :"
    " \""
    ${CMAKE_CXX_COMPILER_ID}
    " "
    ${CMAKE_CXX_COMPILER_VERSION}
    "\"}")
