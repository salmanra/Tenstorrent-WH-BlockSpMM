/*
CK is neutral to order, but it's probably worth naming loop vars
    to reflect the order that RK and WK respec/demand
*/


#include <tools/profiler/kernel_profiler.hpp>
#include "tt_metal/programming_examples/Tenstorrent-WH-BlockSpMM/block_spmm/kernels/common/spmm_profiling.hpp"
#include "tt_metal/programming_examples/Tenstorrent-WH-BlockSpMM/block_spmm/kernels/common/spmm_heartbeat.hpp"

// Compile-time profiling zone toggle (override to 0 via CreateKernel defines)
#ifndef PROFILE_COMPUTE
#define PROFILE_COMPUTE 1
#endif

// Ablation skip flag (set to 1 via CreateKernel defines to skip compute)
#ifndef SKIP_COMPUTE
#define SKIP_COMPUTE 0
#endif

#include <cstdint>
#include "hostdevcommon/kernel_structs.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"


void kernel_main() {
    BSPMM_HB_WP("MMS");
    BSPMM_HB_MATH("bmm_iter start");
    // I'm gonna run this unconditionally... i just want to be sure i capture what i need
    MATH(DeviceZoneScopedN("SpMM Zone: MATH core total runtime"));
    ///////////////////////////////////////////////////////////////////////
    /// COMPILETIME ARGS //////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    constexpr uint32_t in0_block_w = get_compile_time_arg_val(0);              // inner block size in tiles
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(1);        // outer row block size (in inner row blocks)
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(2);      // out_subblock_h*in0_block_w*in0_num_subblocks;
    constexpr uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);   // out_subblock_h*in0_block_w
    constexpr uint32_t in1_num_subblocks = get_compile_time_arg_val(4);        // outer column block size (in inner column blocks)
    constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(5);      // out_subblock_w*in0_block_w* in1_num_subblocks;
    constexpr uint32_t in1_per_core_w = get_compile_time_arg_val(6);           // out_subblock_w*in1_num_subblocks
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(7);           // inner row block size in tiles
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(8);           // inner column block size in tiles
    constexpr uint32_t out_subblock_num_tiles = get_compile_time_arg_val(9);  // out_subblock_h * out_subblock_w;
    // num_iters_x used to be compile-time arg 10; it is now runtime arg 0 because the
    // last core column can own fewer x-iterations than the global count.

    ///////////////////////////////////////////////////////////////////////
    /// END COMPILETIME ARGS //////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////
    /// RUNTIME ARGS //////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    const uint32_t num_iters_x = get_arg_val<uint32_t>(0);
    const uint32_t num_iters_y = get_arg_val<uint32_t>(1);
    uint32_t row_sizes[num_iters_y];
    for (uint32_t i = 0; i < num_iters_y; i++){
        row_sizes[i] = get_arg_val<uint32_t>(i+2);
    }

    ///////////////////////////////////////////////////////////////////////
    /// END RUNTIME ARGS //////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////
    /// PROGRAM BODY //////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    compute_kernel_hw_startup<SrcOrder::Reverse>(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);
    matmul_init(tt::CBIndex::c_0, tt::CBIndex::c_1);


#if SKIP_COMPUTE == 1
    // Drain input CBs and push dummy tiles to output so writer does not stall.
    // c_24 is never touched here since there are no partials to spill.
    constexpr uint32_t out_block_num_tiles =
        out_subblock_num_tiles * in0_num_subblocks * in1_num_subblocks;
    for (uint32_t iter_y = 0; iter_y < num_iters_y; iter_y++){
        uint32_t num_blocks = row_sizes[iter_y];
        BSPMM_HB_MATH("bmm_iter iter_y={} num_blocks={}", iter_y, num_blocks);
        for (uint32_t iter_x = 0; iter_x < num_iters_x; iter_x++){
            for (uint32_t input_block = 0; input_block < num_blocks; input_block++){
                bool last_out = input_block == (num_blocks - 1);
                BSPMM_HB_WP("MMW");
                BSPMM_HB_MATH("bmm_iter wait inputs iter_y={} iter_x={} block={}", iter_y, iter_x, input_block);
                cb_wait_front(tt::CBIndex::c_0, in0_block_num_tiles);
                cb_wait_front(tt::CBIndex::c_1, in1_block_num_tiles);
                BSPMM_HB_WP("MMR");
                BSPMM_HB_MATH("bmm_iter inputs ready iter_y={} iter_x={} block={}", iter_y, iter_x, input_block);
                if (last_out) {
                    cb_reserve_back(tt::CBIndex::c_16, out_block_num_tiles);
                    cb_push_back(tt::CBIndex::c_16, out_block_num_tiles);
                }
                cb_pop_front(tt::CBIndex::c_0, in0_block_num_tiles);
                cb_pop_front(tt::CBIndex::c_1, in1_block_num_tiles);
                BSPMM_HB_WP("MMC");
                BSPMM_HB_MATH("bmm_iter consumed inputs iter_y={} iter_x={} block={}", iter_y, iter_x, input_block);
            }
        }
    }
#else
    for (uint32_t iter_y = 0; iter_y < num_iters_y; iter_y++){
        uint32_t num_blocks = row_sizes[iter_y];
        BSPMM_HB_MATH("bmm_iter iter_y={} num_blocks={}", iter_y, num_blocks);
        for (uint32_t iter_x = 0; iter_x < num_iters_x; iter_x++){
            bool enable_reload = false;
            bool spill = num_blocks > 1;

            uint32_t out_num_tiles_to_wait = out_subblock_num_tiles;
            for (uint32_t input_block = 0; input_block < num_blocks; input_block++){
                bool last_out = input_block == (num_blocks - 1);
                BSPMM_HB_WP("MMW");
                BSPMM_HB_MATH("bmm_iter wait inputs iter_y={} iter_x={} block={}", iter_y, iter_x, input_block);
                cb_wait_front(tt::CBIndex::c_0, in0_block_num_tiles);
                cb_wait_front(tt::CBIndex::c_1, in1_block_num_tiles);
                BSPMM_HB_WP("MMR");
                BSPMM_HB_MATH("bmm_iter inputs ready iter_y={} iter_x={} block={}", iter_y, iter_x, input_block);
#if PROFILE_COMPUTE == 1
                DeviceZoneScopedN("SpMM Zone: CK using input blocks");
#endif
                int in0_index_subblock_offset = 0;
                for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                    int in1_index_subblock_offset = 0;
                    for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                        tile_regs_acquire();

                        if (enable_reload) {
                            copy_tile_to_dst_init_short(tt::CBIndex::c_24);
                            cb_wait_front(tt::CBIndex::c_24, out_subblock_num_tiles);
                            for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                                copy_tile(tt::CBIndex::c_24, i, i);
                            }
                            cb_pop_front(tt::CBIndex::c_24, out_subblock_num_tiles);
                            matmul_init(tt::CBIndex::c_0, tt::CBIndex::c_1);
                        }

                        // Compute output sub-block from in0_subblock x in1_subblock
                        int dst_index = 0;
                        int in0_index_h_offset = 0;
                        for (uint32_t h = 0; h < out_subblock_h; h++) {
                            for (uint32_t w = 0; w < out_subblock_w; w++) {
                                int in1_index_inner_dim_offset = 0;
                                for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                                    int in0_index = in0_index_subblock_offset + in0_index_h_offset + inner_dim;
                                    int in1_index = in1_index_subblock_offset + in1_index_inner_dim_offset + w;

                                    matmul_tiles(
                                        tt::CBIndex::c_0,
                                        tt::CBIndex::c_1,
                                        in0_index,
                                        in1_index,
                                        dst_index);

                                    in1_index_inner_dim_offset += in1_per_core_w;
                                }
                                dst_index++;
                            }
                            in0_index_h_offset += in0_block_w;
                        }

                        tile_regs_commit();
                        tile_regs_wait();

                        if (last_out) {
                            // Pack out to output buffer
                            cb_reserve_back(tt::CBIndex::c_16, out_subblock_num_tiles);
                            for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                                pack_tile(i, tt::CBIndex::c_16);
                            }
                            cb_push_back(tt::CBIndex::c_16, out_subblock_num_tiles);
                        } else {
                            // Wait for tiles in output buffer to be written out since interm and output share memory
                            if (input_block == 0) {
                                cb_reserve_back(tt::CBIndex::c_16, out_num_tiles_to_wait);
                                out_num_tiles_to_wait += out_subblock_num_tiles;
                            }
                            // Move partial result to interm buffer
                            cb_reserve_back(tt::CBIndex::c_24, out_subblock_num_tiles);
                            for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                                pack_tile(i, tt::CBIndex::c_24);
                            }
                            cb_push_back(tt::CBIndex::c_24, out_subblock_num_tiles);

                        }
                        tile_regs_release();

                        in1_index_subblock_offset += out_subblock_w;
                    }
                    in0_index_subblock_offset += in0_subblock_num_tiles;
                }

                if (spill) {
                    enable_reload = true;
                }

                cb_pop_front(tt::CBIndex::c_0, in0_block_num_tiles);
                cb_pop_front(tt::CBIndex::c_1, in1_block_num_tiles);
                BSPMM_HB_WP("MMC");
                BSPMM_HB_MATH("bmm_iter consumed inputs iter_y={} iter_x={} block={}", iter_y, iter_x, input_block);


            }
        }
    }
#endif // SKIP_COMPUTE
    BSPMM_HB_WP("MMD");
    BSPMM_HB_MATH("bmm_iter done");
}
