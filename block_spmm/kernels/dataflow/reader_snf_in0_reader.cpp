#include <stdint.h>
#include <cstdint>
#include "dataflow_api.h"
#include "hostdevcommon/kernel_structs.h"
#include "debug/dprint.h"
#include <tools/profiler/kernel_profiler.hpp>
#include "tt_metal/programming_examples/rahmy/SC26_submission/block_spmm/kernels/common/spmm_reader_common.hpp"
#include "tt_metal/programming_examples/rahmy/SC26_submission/block_spmm/kernels/common/spmm_tile_ops.hpp"
#include "tt_metal/programming_examples/rahmy/SC26_submission/block_spmm/kernels/common/spmm_indexing.hpp"
#include "tt_metal/programming_examples/rahmy/SC26_submission/block_spmm/kernels/common/spmm_profiling.hpp"

// Compile-time profiling zone toggles (override to 0 via CreateKernel defines)
#ifndef PROFILE_READ_IN0
#define PROFILE_READ_IN0 1
#endif
#ifndef PROFILE_WAIT_IN0
#define PROFILE_WAIT_IN0 1
#endif
#ifndef PROFILE_WRITE_OUT
#define PROFILE_WRITE_OUT 1
#endif

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

    // store-and-forward args
    uint32_t in0_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(20));
    uint32_t in0_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(21));

    constexpr uint32_t is_injector_core = get_compile_time_arg_val(22);
    constexpr uint32_t is_output_writer = get_compile_time_arg_val(23);

    // writer args
    uint32_t out_tensor_addr = get_compile_time_arg_val(24);
    uint32_t RtNt = get_compile_time_arg_val(25);
    uint32_t Nt = get_compile_time_arg_val(26);
    uint32_t out_subblock_w = get_compile_time_arg_val(27);
    uint32_t out_subblock_h = get_compile_time_arg_val(28);

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
        folded_y_coords[i] = get_arg_val<uint32_t>(arg_index++);
    }

    // writer args
    uint32_t out_tensor_start_tile_id = get_arg_val<uint32_t>(arg_index++);
    const uint32_t num_cores_y = get_arg_val<uint32_t>(arg_index++);

    // SnF args
    const uint32_t in0_dest_noc_x = get_arg_val<uint32_t>(arg_index++);
    const uint32_t in0_dest_noc_y = get_arg_val<uint32_t>(arg_index++);
    const uint32_t in0_sender_noc_x = get_arg_val<uint32_t>(arg_index++);
    const uint32_t in0_sender_noc_y = get_arg_val<uint32_t>(arg_index++);
    const uint32_t is_sink_core = get_arg_val<uint32_t>(arg_index++);

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

    // SnF semaphores
    volatile tt_l1_ptr uint32_t* in0_sender_semaphore_addr_ptr =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_sender_semaphore_addr);
    *(in0_sender_semaphore_addr_ptr) = 0;
    const uint64_t in0_sender_semaphore_noc_addr =
    get_noc_addr(in0_sender_noc_x, in0_sender_noc_y, in0_sender_semaphore_addr);

    volatile tt_l1_ptr uint32_t* in0_receiver_semaphore_addr_ptr =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_receiver_semaphore_addr);
    *(in0_receiver_semaphore_addr_ptr) = 0;
    const uint64_t in0_receiver_semaphore_noc_addr =
    get_noc_addr(in0_dest_noc_x, in0_dest_noc_y, in0_receiver_semaphore_addr);

    // Load or wait for sparse indexing data
    uint32_t* col_indices;
    uint32_t* indptr;
    if constexpr (is_output_writer){
        col_indices = spmm::load_indexing_tiled<col_indices_is_dram>(
            spmm::cb_id_col_indices, col_indices_addr,
            tile_info.col_indices_tile_size, tile_info.col_indices_format, col_indices_num_tiles);
        indptr = spmm::load_indexing_tiled<indptr_is_dram>(
            spmm::cb_id_indptr, indptr_addr,
            tile_info.indptr_tile_size, tile_info.indptr_format, indptr_num_tiles);
    }
    else {
        indptr = spmm::wait_for_indexing(spmm::cb_id_indptr, indptr_num_tiles);
        col_indices = spmm::wait_for_indexing(spmm::cb_id_col_indices, col_indices_num_tiles);
    }


    // Writer setup
    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;
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
        uint32_t out_tensor_y_coord_offset = RtNt * folded_y_coords[iter_y];

        // For now, all blocks are the same size. But soon we will want this line to accomodate unaligned blocks
        uint32_t current_block_bytes = tile_info.in0_tile_size * in0_block_num_tiles;
        // Get y_coord for this iter
        output_idx_y = y_coords[iter_y];
        uint32_t block_row_start = indptr[output_idx_y];
        uint32_t block_row_end = indptr[output_idx_y + 1];

        uint32_t in0_tensor_start_tile_id = block_row_start * in0_block_num_tiles;
        for (uint32_t iter_x = 0; iter_x < num_iters_x; iter_x++){
            output_idx_x = output_idx_x_start + iter_x;
            uint32_t in1_tensor_start_tile_id = in1_block_w * output_idx_x;
            for (uint32_t reduction_iter = block_row_start; reduction_iter < block_row_end; reduction_iter++){
                cb_reserve_back(spmm::cb_id_in0, in0_block_num_tiles);
                uint32_t l1_write_addr_in0 = get_write_ptr(spmm::cb_id_in0);
                uint32_t l1_write_addr_in0_start = l1_write_addr_in0;  // Save start address for forwarding

                if constexpr (is_injector_core){
#if SKIP_IN0_DRAM_READ == 0
#if PROFILE_READ_IN0 == 1
                    DeviceZoneScopedN("SpMM Zone: Reading nonzero block from in0 from DRAM");
#endif
                    // Read in0 block from DRAM
                    uint32_t num_blocks_in = reduction_iter - block_row_start;
                    DPRINT_DATA1(DPRINT << "in0 DRAM read: " << reduction_iter << ENDL());
                    spmm::read_block_by_tile(
                        in0_tensor_start_tile_id + num_blocks_in * in0_block_num_tiles,
                        s0, l1_write_addr_in0,
                        tile_info.in0_tile_size, in0_block_h, in0_block_w,
                        in0_tensor_stride_h, in0_tensor_stride_w);
                    noc_async_read_barrier();
#endif
                }
                else {
#if PROFILE_WAIT_IN0 == 1
                    DeviceZoneScopedN("SpMM Zone: Waiting on nonzero block from in0 from neighbor");
#endif
                    // Read in0 block from neighbor
                    noc_semaphore_set(in0_receiver_semaphore_addr_ptr, 0);
                    noc_semaphore_inc(in0_sender_semaphore_noc_addr, 1);
                    noc_semaphore_wait(in0_receiver_semaphore_addr_ptr, 1);
                }
                
                cb_push_back(spmm::cb_id_in0, in0_block_num_tiles);

                if (!is_sink_core) {
                    noc_semaphore_wait(in0_sender_semaphore_addr_ptr, 1);
                    noc_semaphore_set(in0_sender_semaphore_addr_ptr, 0);

                    uint64_t in0_unicast_data_addr = get_noc_addr(in0_dest_noc_x, in0_dest_noc_y, l1_write_addr_in0_start);
                    noc_async_write(l1_write_addr_in0_start, in0_unicast_data_addr, current_block_bytes);
                    noc_async_write_barrier();
                    noc_semaphore_inc(in0_receiver_semaphore_noc_addr, 1);
                }
            }

            if constexpr (is_output_writer){
                uint32_t out_block_num_tiles = in0_block_h * in1_block_w;
                uint32_t out_tensor_sbh_start_tile_id = out_tensor_start_tile_id + out_tensor_y_coord_offset + out_tensor_x_coord_offset;

                cb_wait_front(spmm::cb_id_out, out_block_num_tiles);
                uint32_t l1_read_addr = get_read_ptr(spmm::cb_id_out);
#if SKIP_DRAM_WRITE == 0
#if PROFILE_WRITE_OUT == 1
                DeviceZoneScopedN("SpMM Zone: Writing Block back to DRAM");
#endif
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
