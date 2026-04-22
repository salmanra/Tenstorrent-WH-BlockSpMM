cmake_minimum_required(VERSION 3.22...3.30)

add_library(sparse_common STATIC host_code_utils.cpp)

target_include_directories(sparse_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_SOURCE_DIR}/ttnn/cpp
)

target_link_libraries(sparse_common PUBLIC tt_metal)
