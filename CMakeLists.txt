cmake_minimum_required(VERSION 3.22...3.30)

# ─────────────────────────────────────────────────────────────────────────────
# Tenstorrent-WH-BlockSpMM: block-sparse SpMM + SDDMM kernels for the Wormhole
# N150, companion artifact to the paper "Leveraging Network-on-Chip Architecture
# for Block-Sparse Linear Algebra on Tenstorrent".
#
# Embedded-only submodule: expected to be mounted as a git submodule at
#   $TT_METAL_HOME/tt_metal/programming_examples/Tenstorrent-WH-BlockSpMM/
#
# External dependencies: tt_metal (provided by the host tt-metal project) and
# its CREATE_PGM_EXAMPLES_EXE / PROGRAMMING_EXAMPLES_TEST_TARGETS build glue.
# All other dependencies (sparse_common, Tracy profiler) are self-contained.
#
# Output binaries land at:  build/programming_examples/block_sparse/
# ─────────────────────────────────────────────────────────────────────────────

# sparse_common must come first: block_sddmm/block_spmm host_code libs link it.
add_subdirectory(sparse_common)
add_subdirectory(block_spmm/inc/host_code)
add_subdirectory(block_sddmm)

# Entry-point sources always built
set(BLKSPMM_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/block_spmm/src/run_block_spmm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/block_spmm/src/test_block_spmm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/block_spmm/src/export_block_spmm_to_csv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/block_sddmm/src/run_sddmm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/block_sddmm/src/test_sddmm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/block_sddmm/src/export_sddmm.cpp
)

if(ENABLE_TRACY)
    list(
        APPEND
        BLKSPMM_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/block_spmm/src/profile_block_spmm.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/block_sddmm/src/profile_sddmm.cpp
    )
endif()

CREATE_PGM_EXAMPLES_EXE("${BLKSPMM_SRCS}" "block_sparse")

foreach(target ${PROGRAMMING_EXAMPLES_TEST_TARGETS})
    if(TARGET ${target})
        target_link_libraries(
            ${target}
            PRIVATE
                sddmm_host_code
                block_sparse_spmm_host_code
                sparse_common
        )
    endif()
endforeach()

add_custom_target(block_sparse DEPENDS ${PROGRAMMING_EXAMPLES_TEST_TARGETS})
