#include <stdint.h>
#include <cstdint>
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "hostdevcommon/kernel_structs.h"
#include <tools/profiler/kernel_profiler.hpp>
#include "tt_metal/programming_examples/rahmy/SC26_submission/block_spmm/kernels/common/spmm_reader_common.hpp"
#include "tt_metal/programming_examples/rahmy/SC26_submission/block_spmm/kernels/common/spmm_tile_ops.hpp"
#include "tt_metal/programming_examples/rahmy/SC26_submission/block_spmm/kernels/common/spmm_indexing.hpp"

// Ablation skip flags (set to 1 via CreateKernel defines to skip that phase)
#ifndef SKIP_IN0_DRAM_READ
#define SKIP_IN0_DRAM_READ 0
#endif
#ifndef SKIP_DRAM_WRITE
#define SKIP_DRAM_WRITE 0
#endif

void kernel_main(){
    ///////////////////////////////////////////////////////////////////////
    /// COMPILETIME ARGS //////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    constexpr bool in0_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool in1_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool col_indices_is_dram = get_compile_time_arg_val(2) == 1;
    constexpr bool indptr_is_dram = get_compile_time_arg_val(3) == 1;

    constexpr uint32_t in0_tensor_addr = get_compile_time_arg_val(4);
    constexpr uint32_t in0_tensor_stride_w = get_compile_time_arg_val(5);
    constexpr uint32_t in0_tensor_stride_h = get_compile_time_arg_val(6);

    constexpr uint32_t in0_block_w = get_compile_time_arg_val(7);
    constexpr uint32_t in0_block_h = get_compile_time_arg_val(8);
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(9);

    constexpr uint32_t in1_tensor_addr = get_compile_time_arg_val(10);
    constexpr uint32_t in1_tensor_stride_w = get_compile_time_arg_val(11);
    constexpr uint32_t in1_tensor_stride_h = get_compile_time_arg_val(12);

    constexpr uint32_t in1_block_w = get_compile_time_arg_val(13);
    constexpr uint32_t in1_block_h = get_compile_time_arg_val(14);
    constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(15);

    constexpr uint32_t col_indices_addr = get_compile_time_arg_val(16);
    constexpr uint32_t indptr_addr = get_compile_time_arg_val(17);

    constexpr uint32_t col_indices_num_tiles = get_compile_time_arg_val(18);
    constexpr uint32_t indptr_num_tiles = get_compile_time_arg_val(19);

    constexpr uint32_t is_output_writer = get_compile_time_arg_val(20);

    // writer args (only used when is_output_writer == 1)
    constexpr uint32_t out_tensor_addr = get_compile_time_arg_val(21);
    constexpr uint32_t RtNt = get_compile_time_arg_val(22);
    constexpr uint32_t Nt = get_compile_time_arg_val(23);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(24);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(25);

    ///////////////////////////////////////////////////////////////////////
    /// END COMPILETIME ARGS //////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////
    /// RUNTIME ARGS //////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    uint32_t arg_index = 0;
    const uint32_t num_iters_x = get_arg_val<uint32_t>(arg_index++);
    const uint32_t num_iters_y = get_arg_val<uint32_t>(arg_index++);
    const uint32_t output_idx_x_start = get_arg_val<uint32_t>(arg_index++);
    uint32_t y_coords[num_iters_y];
    uint32_t folded_y_coords[num_iters_y];
    for (uint32_t i = 0; i < num_iters_y; i++){
        y_coords[i] = get_arg_val<uint32_t>(arg_index++);
        if constexpr (is_output_writer) {
            folded_y_coords[i] = get_arg_val<uint32_t>(arg_index++);
        }
    }
    uint32_t out_tensor_start_tile_id = 0;
    if constexpr (is_output_writer) {
        out_tensor_start_tile_id = get_arg_val<uint32_t>(arg_index++);
    }

    ///////////////////////////////////////////////////////////////////////
    /// END RUNTIME ARGS //////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    const auto tile_info = spmm::get_tile_info();
    const uint32_t output_tile_size = get_tile_size(spmm::cb_id_out);
    const DataFormat output_format = get_dataformat(spmm::cb_id_out);

    const InterleavedAddrGenFast<in0_is_dram> s0 = {
        .bank_base_address = in0_tensor_addr, .page_size = tile_info.in0_tile_size, .data_format = tile_info.in0_format};
    const InterleavedAddrGenFast<true> out_s = {
        .bank_base_address = out_tensor_addr, .page_size = output_tile_size, .data_format = output_format};

    // Load sparse indexing data (in0 always owns the indexing load in the naive variant)
    uint32_t* col_indices = spmm::load_indexing_tiled<col_indices_is_dram>(
        spmm::cb_id_col_indices, col_indices_addr,
        tile_info.col_indices_tile_size, tile_info.col_indices_format, col_indices_num_tiles);
    uint32_t* indptr = spmm::load_indexing_tiled<indptr_is_dram>(
        spmm::cb_id_indptr, indptr_addr,
        tile_info.indptr_tile_size, tile_info.indptr_format, indptr_num_tiles);

    // Writer setup (computed unconditionally; values are only used when is_output_writer == 1)
    uint32_t out_num_subblocks_w = in1_block_w / out_subblock_w;
    uint32_t out_num_subblocks_h = in0_block_h / out_subblock_h;
    uint32_t out_tensor_next_subblock_stride_w = out_subblock_w;
    uint32_t out_tensor_next_subblock_stride_h = out_subblock_h * Nt;
    uint32_t out_tensor_stride_w = 1;
    uint32_t out_tensor_stride_h = Nt;

    ///////////////////////////////////////////////////////////////////////
    /// PROGRAM BODY //////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    uint32_t out_tensor_x_coord_offset = 0;
    uint32_t output_idx_y, output_idx_x;
    for (uint32_t iter_y = 0; iter_y < num_iters_y; iter_y++){
        output_idx_y = y_coords[iter_y];
        uint32_t block_row_start = indptr[output_idx_y];
        uint32_t block_row_end = indptr[output_idx_y + 1];

        uint32_t in0_tensor_start_tile_id = block_row_start * in0_block_num_tiles;
        for (uint32_t iter_x = 0; iter_x < num_iters_x; iter_x++){
            output_idx_x = output_idx_x_start + iter_x;
            for (uint32_t reduction_iter = block_row_start; reduction_iter < block_row_end; reduction_iter++){
                cb_reserve_back(spmm::cb_id_in0, in0_block_num_tiles);
                uint32_t l1_write_addr_in0 = get_write_ptr(spmm::cb_id_in0);

#if SKIP_IN0_DRAM_READ == 0
                {
                    DeviceZoneScopedN("SpMM Zone: Reading nonzero block from in0 from DRAM");
                    DPRINT_DATA1(DPRINT << "in0 DRAM read: " << reduction_iter << ENDL());
                    uint32_t num_blocks_in = reduction_iter - block_row_start;
                    spmm::read_block_by_tile(
                        in0_tensor_start_tile_id + num_blocks_in * in0_block_num_tiles,
                        s0, l1_write_addr_in0,
                        tile_info.in0_tile_size, in0_block_h, in0_block_w,
                        in0_tensor_stride_h, in0_tensor_stride_w);
                }
                noc_async_read_barrier();
#endif
                cb_push_back(spmm::cb_id_in0, in0_block_num_tiles);
            }

            if constexpr (is_output_writer) {
                uint32_t out_tensor_y_coord_offset = RtNt * folded_y_coords[iter_y];
                uint32_t out_block_num_tiles = in0_block_h * in1_block_w;
                uint32_t out_tensor_sbh_start_tile_id = out_tensor_start_tile_id + out_tensor_y_coord_offset + out_tensor_x_coord_offset;

                cb_wait_front(spmm::cb_id_out, out_block_num_tiles);

#if SKIP_DRAM_WRITE == 0
                {
                    DeviceZoneScopedN("SpMM Zone: Writing Block back to DRAM");
                    uint32_t l1_read_addr = get_read_ptr(spmm::cb_id_out);

                    for (uint32_t sbh = 0; sbh < out_num_subblocks_h; sbh++) {
                        uint32_t out_tensor_sbw_start_tile_id = out_tensor_sbh_start_tile_id;
                        for (uint32_t sbw = 0; sbw < out_num_subblocks_w; sbw++) {
                            spmm::write_subblock_by_tile(
                                out_tensor_sbw_start_tile_id,
                                out_s, l1_read_addr,
                                output_tile_size, out_subblock_h, out_subblock_w,
                                out_tensor_stride_h, out_tensor_stride_w);
                            out_tensor_sbw_start_tile_id += out_tensor_next_subblock_stride_w;
                        }
                        out_tensor_sbh_start_tile_id += out_tensor_next_subblock_stride_h;
                    }
                }
                noc_async_write_barrier();
#endif
                cb_pop_front(spmm::cb_id_out, out_block_num_tiles);
                out_tensor_x_coord_offset += out_num_subblocks_w * out_tensor_next_subblock_stride_w;
            }
        }
        out_tensor_x_coord_offset = 0;
    }
    cb_pop_front(spmm::cb_id_col_indices, col_indices_num_tiles);
    cb_pop_front(spmm::cb_id_indptr, indptr_num_tiles);

}
