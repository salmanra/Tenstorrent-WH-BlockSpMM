# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22...3.30)

# Host-side program factories and operation types for the 8x8 BSR SpMM variants.
# Compiled into a static library so tt-metal's ttnn op wrappers can link them.

add_library(
    tiny_bsr_spmm_host
    STATIC
    src/tiny_bsr_spmm_program_factory.cpp
    src/tiny_bsr_spmm_snf_program_factory.cpp
    src/tiny_bsr_spmm_cda_program_factory.cpp
    src/sparse_8x16_matmul_program_factory.cpp
)

add_library(TinyBsrSpmm::Host ALIAS tiny_bsr_spmm_host)

target_include_directories(
    tiny_bsr_spmm_host
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/ttnn/api
    PRIVATE
        ${CMAKE_SOURCE_DIR}/tt_metal/tt-llk/tt_llk_blackhole/llk_lib
        ${CMAKE_SOURCE_DIR}/tt_metal/tt-llk/tt_llk_blackhole/common/inc
        ${CMAKE_CURRENT_SOURCE_DIR}/../llk_blackhole/llk_lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../llk_blackhole/common/inc
)

target_link_libraries(tiny_bsr_spmm_host PUBLIC tt_metal)

target_compile_options(
    tiny_bsr_spmm_host
    PRIVATE
        $<$<CXX_COMPILER_ID:Clang,AppleClang>:-Wno-c++11-narrowing>
        $<$<CXX_COMPILER_ID:GNU>:-Wno-narrowing>
)

# Install device kernels so the JIT build can find them under the relative path
# used by CreateKernel in the program factories.
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/kernels/
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt_metal/programming_examples/Tenstorrent-WH-BlockSpMM/tiny_bsr_spmm/kernels
    COMPONENT ttnn-runtime
)
