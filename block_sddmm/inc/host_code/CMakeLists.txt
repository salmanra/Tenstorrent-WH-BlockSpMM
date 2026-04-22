cmake_minimum_required(VERSION 3.22...3.30)

set(HOST_CODE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/host_code_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bsr_sddmm_multicore_naive.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bsr_sddmm_multicore_CDA.cpp
    # bsr_sddmm_multicore_SnF_C.cpp excluded — kernel stub not ready yet
)

add_library(sddmm_host_code STATIC ${HOST_CODE_SOURCES})

target_include_directories(sddmm_host_code PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_SOURCE_DIR}/ttnn/cpp
)

target_link_libraries(sddmm_host_code PUBLIC sparse_common tt_metal)
