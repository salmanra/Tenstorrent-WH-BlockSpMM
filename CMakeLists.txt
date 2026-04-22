cmake_minimum_required(VERSION 3.22...3.30)

# ─────────────────────────────────────────────────────────────────────────────
# SC26 submodule: self-contained block-sparse SpMM + SDDMM kernels for the
# paper "Leveraging Network-on-Chip Architecture for Block-Sparse Linear
# Algebra on Tenstorrent".
#
# Embedded-only: this CMakeLists is expected to be pulled in via
# add_subdirectory() from a host project. The submodule brings its own
# sparse_common target; the only external dependency is tt_metal (linked
# transitively via the host_code libraries) plus the host's
# PROGRAMMING_EXAMPLES_TEST_TARGETS / CREATE_PGM_EXAMPLES_EXE build glue.
#
# The host project is expected to:
#   1. add_subdirectory(SC26_submission)
#   2. Build the sources exposed in SC26_ENTRY_POINT_SRCS (+ SC26_TRACY_SRCS
#      guarded by ENABLE_TRACY) into executables
#   3. target_link_libraries(<target> PRIVATE ${SC26_LINK_LIBS})
# ─────────────────────────────────────────────────────────────────────────────

# sparse_common must come first: block_sddmm/block_spmm host_code libs depend on it.
add_subdirectory(sparse_common)
# Build the static host_code libraries for each subproject
add_subdirectory(block_spmm/inc/host_code)
add_subdirectory(block_sddmm)

# Entry-point sources always built
set(SC26_ENTRY_POINT_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/block_spmm/src/run_block_spmm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/block_spmm/src/test_block_spmm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/block_spmm/src/export_block_spmm_to_csv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/block_sddmm/src/test_sddmm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/block_sddmm/src/run_sddmm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/block_sddmm/src/export_sddmm.cpp
    PARENT_SCOPE
)

# Entry-point sources built only when profiling is enabled
set(SC26_TRACY_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/block_spmm/src/profile_block_spmm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/block_sddmm/src/profile_sddmm.cpp
    PARENT_SCOPE
)

# Link libraries that SC26 executables need.
# sparse_common must be provided by the host project.
set(SC26_LINK_LIBS
    sddmm_host_code
    sc26_spmm_host_code
    sparse_common
    PARENT_SCOPE
)
