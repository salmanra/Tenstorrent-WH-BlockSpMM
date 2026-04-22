// SDDMM D Reader + Output Writer kernel (RISCV_1)
//
// Streams dense D blocks along the K reduction dimension and writes
// the output block A[i,j] back to DRAM for each assigned output block.

#include <stdint.h>
#include "dataflow_api.h"
#include "tt_metal/programming_examples/Tenstorrent-WH-BlockSpMM/block_spmm/kernels/common/spmm_tile_ops.hpp"

// Ablation skip flags (set to 1 via CreateKernel defines to skip that phase)
#ifndef SKIP_D_DRAM_READ
#define SKIP_D_DRAM_READ 0
#endif
#ifndef SKIP_DRAM_WRITE
#define SKIP_DRAM_WRITE 0
#endif

void kernel_main() {
    // ── Compile-time args ────────────────────────────────────────────
    constexpr bool dense_d_is_dram         = get_compile_time_arg_val(0);
    constexpr bool out_is_dram             = get_compile_time_arg_val(1);
    constexpr uint32_t dense_d_tensor_addr = get_compile_time_arg_val(2);
    constexpr uint32_t dense_d_stride_w    = get_compile_time_arg_val(3);     // 1
    constexpr uint32_t dense_d_stride_h    = get_compile_time_arg_val(4);     // Nt
    constexpr uint32_t dense_d_block_h     = get_compile_time_arg_val(5);     // block_k
    constexpr uint32_t dense_d_block_w     = get_compile_time_arg_val(6);     // Ct
    constexpr uint32_t dense_d_block_num_tiles = get_compile_time_arg_val(7); // block_k * Ct
    constexpr uint32_t out_tensor_addr     = get_compile_time_arg_val(8);
    constexpr uint32_t out_block_num_tiles = get_compile_time_arg_val(9);     // Rt * Ct
    constexpr uint32_t out_subblock_w      = get_compile_time_arg_val(10);
    constexpr uint32_t out_subblock_h      = get_compile_time_arg_val(11);
    constexpr uint32_t num_blocks_k        = get_compile_time_arg_val(12);
    constexpr uint32_t Rt                  = get_compile_time_arg_val(13);
    constexpr uint32_t Ct                  = get_compile_time_arg_val(14);

    // CB IDs
    constexpr uint32_t cb_dense_d = 2;   // D reduction blocks
    constexpr uint32_t cb_out = 16;      // Output block

    // Tile sizes
    constexpr uint32_t dense_d_tile_size = get_tile_size(cb_dense_d);
    constexpr uint32_t out_tile_size = get_tile_size(cb_out);
    constexpr auto dense_d_df = get_dataformat(cb_dense_d);
    constexpr auto out_df = get_dataformat(cb_out);

    // Address generators
    const InterleavedAddrGenFast<dense_d_is_dram> dense_d_addr_gen = {
        .bank_base_address = dense_d_tensor_addr,
        .page_size = dense_d_tile_size,
        .data_format = dense_d_df};

    const InterleavedAddrGenFast<out_is_dram> out_addr_gen = {
        .bank_base_address = out_tensor_addr,
        .page_size = out_tile_size,
        .data_format = out_df};

    // Subblock iteration counts
    constexpr uint32_t out_num_subblocks_h = Rt / out_subblock_h;
    constexpr uint32_t out_num_subblocks_w = Ct / out_subblock_w;

    // ── Runtime args ─────────────────────────────────────────────────
    uint32_t arg_idx = 0;
    const uint32_t num_output_blocks = get_arg_val<uint32_t>(arg_idx++);

    // ── Main loop ────────────────────────────────────────────────────
    for (uint32_t ob = 0; ob < num_output_blocks; ob++) {
        uint32_t block_col_j = get_arg_val<uint32_t>(arg_idx++);
        uint32_t blk_data_idx = get_arg_val<uint32_t>(arg_idx++);

        // ── Stream D blocks for reduction ────────────────────────────
        for (uint32_t k = 0; k < num_blocks_k; k++) {
            cb_reserve_back(cb_dense_d, dense_d_block_num_tiles);
#if SKIP_D_DRAM_READ == 0
            uint32_t l1_addr_d = get_write_ptr(cb_dense_d);
            uint32_t d_start_tile = k * dense_d_block_h * dense_d_stride_h + block_col_j * dense_d_block_w;
            spmm::read_block_by_tile(
                d_start_tile, dense_d_addr_gen, l1_addr_d,
                dense_d_tile_size, dense_d_block_h, dense_d_block_w,
                dense_d_stride_h, dense_d_stride_w);
            noc_async_read_barrier();
#endif
            cb_push_back(cb_dense_d, dense_d_block_num_tiles);
        }

        // ── Write output block to DRAM ───────────────────────────────
        cb_wait_front(cb_out, out_block_num_tiles);
#if SKIP_DRAM_WRITE == 0
        uint32_t l1_read_addr = get_read_ptr(cb_out);
        uint32_t out_start_tile = blk_data_idx * out_block_num_tiles;
        uint32_t out_sbh_start = out_start_tile;
        for (uint32_t sbh = 0; sbh < out_num_subblocks_h; sbh++) {
            uint32_t out_sbw_start = out_sbh_start;
            for (uint32_t sbw = 0; sbw < out_num_subblocks_w; sbw++) {
                spmm::write_subblock_by_tile(
                    out_sbw_start, out_addr_gen, l1_read_addr,
                    out_tile_size, out_subblock_h, out_subblock_w,
                    Ct, 1);  // stride_h = Ct within block, stride_w = 1
                out_sbw_start += out_subblock_w;
            }
            out_sbh_start += out_subblock_h * Ct;
        }
        noc_async_write_barrier();
#endif
        cb_pop_front(cb_out, out_block_num_tiles);
    }
}
