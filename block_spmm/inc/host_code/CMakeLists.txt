cmake_minimum_required(VERSION 3.22...3.30)

add_library(sc26_spmm_host_code STATIC
    host_code_utils.cpp
    bsr_spmm_multicore_load_balanced_new_DM.cpp
    bsr_spmm_multicore_snf.cpp
    bsr_spmm_multicore_snfin0_cdain1.cpp
)

target_include_directories(sc26_spmm_host_code PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_SOURCE_DIR}/ttnn/cpp
)

target_link_libraries(sc26_spmm_host_code PUBLIC tt_metal)
