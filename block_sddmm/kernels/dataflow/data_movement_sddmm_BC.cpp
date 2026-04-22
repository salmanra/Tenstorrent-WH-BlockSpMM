// SDDMM BC Reader kernel (RISCV_0)
//
// Reads the sparse mask block B[i,j] and streams dense C blocks
// along the K reduction dimension for each assigned output block.

#include <stdint.h>
#include "dataflow_api.h"
#include "tt_metal/programming_examples/rahmy/SC26_submission/block_spmm/kernels/common/spmm_tile_ops.hpp"

// Ablation skip flags (set to 1 via CreateKernel defines to skip that phase)
#ifndef SKIP_SPARSE_DRAM_READ
#define SKIP_SPARSE_DRAM_READ 0
#endif
#ifndef SKIP_C_DRAM_READ
#define SKIP_C_DRAM_READ 0
#endif

void kernel_main() {
    // ── Compile-time args ────────────────────────────────────────────
    constexpr bool sparse_is_dram          = get_compile_time_arg_val(0);
    constexpr bool dense_c_is_dram         = get_compile_time_arg_val(1);
    constexpr uint32_t sparse_tensor_addr  = get_compile_time_arg_val(2);
    constexpr uint32_t sparse_block_num_tiles = get_compile_time_arg_val(3);  // Rt*Ct
    constexpr uint32_t dense_c_tensor_addr = get_compile_time_arg_val(4);
    constexpr uint32_t dense_c_stride_w    = get_compile_time_arg_val(5);     // 1
    constexpr uint32_t dense_c_stride_h    = get_compile_time_arg_val(6);     // Kt
    constexpr uint32_t dense_c_block_h     = get_compile_time_arg_val(7);     // Rt
    constexpr uint32_t dense_c_block_w     = get_compile_time_arg_val(8);     // block_k
    constexpr uint32_t dense_c_block_num_tiles = get_compile_time_arg_val(9); // Rt * block_k
    constexpr uint32_t num_blocks_k        = get_compile_time_arg_val(10);

    // CB IDs (from sddmm_reader_common.hpp)
    constexpr uint32_t cb_sparse = 0;   // B mask block
    constexpr uint32_t cb_dense_c = 1;  // C reduction blocks

    // Tile size
    constexpr uint32_t sparse_tile_size = get_tile_size(cb_sparse);
    constexpr uint32_t dense_c_tile_size = get_tile_size(cb_dense_c);
    constexpr auto sparse_df = get_dataformat(cb_sparse);
    constexpr auto dense_c_df = get_dataformat(cb_dense_c);

    // Address generators
    const InterleavedAddrGenFast<sparse_is_dram> sparse_addr_gen = {
        .bank_base_address = sparse_tensor_addr,
        .page_size = sparse_tile_size,
        .data_format = sparse_df};

    const InterleavedAddrGenFast<dense_c_is_dram> dense_c_addr_gen = {
        .bank_base_address = dense_c_tensor_addr,
        .page_size = dense_c_tile_size,
        .data_format = dense_c_df};

    // ── Runtime args ─────────────────────────────────────────────────
    uint32_t arg_idx = 0;
    const uint32_t num_output_blocks = get_arg_val<uint32_t>(arg_idx++);

    // ── Main loop ────────────────────────────────────────────────────
    for (uint32_t ob = 0; ob < num_output_blocks; ob++) {
        uint32_t block_row_i = get_arg_val<uint32_t>(arg_idx++);
        uint32_t blk_data_idx = get_arg_val<uint32_t>(arg_idx++);

        // ── Read sparse mask block B[i,j] into c_0 ──────────────────
        cb_reserve_back(cb_sparse, sparse_block_num_tiles);
#if SKIP_SPARSE_DRAM_READ == 0
        uint32_t l1_write_addr = get_write_ptr(cb_sparse);
        uint32_t sparse_start_tile = blk_data_idx * sparse_block_num_tiles;
        for (uint32_t t = 0; t < sparse_block_num_tiles; t++) {
            noc_async_read_tile(sparse_start_tile + t, sparse_addr_gen, l1_write_addr);
            l1_write_addr += sparse_tile_size;
        }
        noc_async_read_barrier();
#endif
        cb_push_back(cb_sparse, sparse_block_num_tiles);

        // ── Stream C blocks for reduction ────────────────────────────
        for (uint32_t k = 0; k < num_blocks_k; k++) {
            cb_reserve_back(cb_dense_c, dense_c_block_num_tiles);
#if SKIP_C_DRAM_READ == 0
            uint32_t l1_addr_c = get_write_ptr(cb_dense_c);
            uint32_t c_start_tile = block_row_i * dense_c_block_h * dense_c_stride_h + k * dense_c_block_w;
            spmm::read_block_by_tile(
                c_start_tile, dense_c_addr_gen, l1_addr_c,
                dense_c_tile_size, dense_c_block_h, dense_c_block_w,
                dense_c_stride_h, dense_c_stride_w);
            noc_async_read_barrier();
#endif
            cb_push_back(cb_dense_c, dense_c_block_num_tiles);
        }
    }
}
