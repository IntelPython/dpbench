add_subdirectory(black_scholes)
add_subdirectory(pairwise_distance)

#generate dpcpp version into json
file(WRITE
${CMAKE_SOURCE_DIR}/dpbench/configs/framework_info/cxx_version.json
"{\"cxx_version\" :"
" \""
${CMAKE_CXX_COMPILER_ID}
" "
${CMAKE_CXX_COMPILER_VERSION}
"\"}")
