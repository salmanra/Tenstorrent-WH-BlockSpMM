// SDDMM Compute Kernel
//
// For each assigned output block (i,j):
// 1. Accumulate matmul: sum_k C_block(i,k) × D_block(k,j) via tiled multiply-accumulate
// 2. Element-wise (Hadamard) multiply with the sparse mask block B[i,j]
// 3. Push final result to output CB
//
// The matmul reduction phase is identical to SpMM's bmm_iter, but operates
// on c_1 (dense C) × c_2 (dense D) instead of c_0 × c_1.
// The Hadamard multiply is SDDMM-specific.

#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/eltwise_binary.h"

// Ablation skip flag (set to 1 via CreateKernel defines to skip compute)
#ifndef SKIP_COMPUTE
#define SKIP_COMPUTE 0
#endif

namespace NAMESPACE {
void MAIN {
    // ── Compile-time args ────────────────────────────────────────────
    constexpr uint32_t in0_block_w            = get_compile_time_arg_val(0);   // block_k
    constexpr uint32_t in0_num_subblocks      = get_compile_time_arg_val(1);   // Rt / out_subblock_h
    constexpr uint32_t in0_block_num_tiles    = get_compile_time_arg_val(2);   // Rt * block_k
    constexpr uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);   // out_subblock_h * block_k
    constexpr uint32_t in1_num_subblocks      = get_compile_time_arg_val(4);   // Ct / out_subblock_w
    constexpr uint32_t in1_block_num_tiles    = get_compile_time_arg_val(5);   // Ct * block_k
    constexpr uint32_t in1_per_core_w         = get_compile_time_arg_val(6);   // Ct
    constexpr uint32_t out_subblock_h         = get_compile_time_arg_val(7);
    constexpr uint32_t out_subblock_w         = get_compile_time_arg_val(8);
    constexpr uint32_t out_subblock_num_tiles = get_compile_time_arg_val(9);   // out_subblock_h * out_subblock_w
    constexpr uint32_t num_blocks             = get_compile_time_arg_val(10);  // num_blocks_k

    constexpr uint32_t out_block_num_tiles =
        out_subblock_num_tiles * in0_num_subblocks * in1_num_subblocks;  // Rt * Ct

    // CB IDs
    constexpr uint32_t cb_sparse = tt::CBIndex::c_0;   // B mask block
    constexpr uint32_t cb_dense_c = tt::CBIndex::c_1;  // C reduction blocks
    constexpr uint32_t cb_dense_d = tt::CBIndex::c_2;  // D reduction blocks
    constexpr uint32_t cb_out = tt::CBIndex::c_16;      // Output
    constexpr uint32_t cb_intermed = tt::CBIndex::c_24; // Intermediate

    // ── Runtime args ─────────────────────────────────────────────────
    const uint32_t num_output_blocks = get_arg_val<uint32_t>(0);

    // ── Init ─────────────────────────────────────────────────────────
    mm_init(cb_dense_c, cb_dense_d, cb_out);

#if SKIP_COMPUTE == 1
    // ── Skip compute: drain inputs, push dummy output ────────────────
    for (uint32_t ob = 0; ob < num_output_blocks; ob++) {
        for (uint32_t input_block = 0; input_block < num_blocks; input_block++) {
            cb_wait_front(cb_dense_c, in0_block_num_tiles);
            cb_wait_front(cb_dense_d, in1_block_num_tiles);
            bool last_out = (input_block == (num_blocks - 1));
            if (last_out) {
                cb_reserve_back(cb_out, out_block_num_tiles);
                cb_push_back(cb_out, out_block_num_tiles);
            }
            cb_pop_front(cb_dense_c, in0_block_num_tiles);
            cb_pop_front(cb_dense_d, in1_block_num_tiles);
        }
        cb_wait_front(cb_sparse, out_block_num_tiles);
        cb_pop_front(cb_sparse, out_block_num_tiles);
    }
#else
    // ── Main compute loop ────────────────────────────────────────────
    for (uint32_t ob = 0; ob < num_output_blocks; ob++) {
        bool enable_reload = false;
        bool spill = num_blocks > 1;

        // Wait for sparse mask block (pushed first by BC reader)
        cb_wait_front(cb_sparse, out_block_num_tiles);

        for (uint32_t input_block = 0; input_block < num_blocks; input_block++) {
            bool last_out = (input_block == (num_blocks - 1));

            // Wait for C and D reduction blocks
            cb_wait_front(cb_dense_c, in0_block_num_tiles);
            cb_wait_front(cb_dense_d, in1_block_num_tiles);

            // ── Subblock matmul (same structure as SpMM bmm_iter) ────
            int in0_index_subblock_offset = 0;
            for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                int in1_index_subblock_offset = 0;
                for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                    acquire_dst();

                    if (enable_reload) {
                        copy_tile_to_dst_init_short(cb_intermed);
                        cb_wait_front(cb_intermed, out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            copy_tile(cb_intermed, i, i);
                        }
                        cb_pop_front(cb_intermed, out_subblock_num_tiles);
                        mm_init_short(cb_dense_c, cb_dense_d);
                    }

                    // Tiled multiply-accumulate: C[h,k] × D[k,w]
                    int dst_index = 0;
                    int in0_index_h_offset = 0;
                    for (uint32_t h = 0; h < out_subblock_h; h++) {
                        for (uint32_t w = 0; w < out_subblock_w; w++) {
                            int in1_index_inner_dim_offset = 0;
                            for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                                int in0_index = in0_index_subblock_offset + in0_index_h_offset + inner_dim;
                                int in1_index = in1_index_subblock_offset + in1_index_inner_dim_offset + w;
                                matmul_tiles(
                                    cb_dense_c, cb_dense_d,
                                    in0_index, in1_index,
                                    dst_index,
                                    false /* transpose */);
                                in1_index_inner_dim_offset += in1_per_core_w;
                            }
                            dst_index++;
                        }
                        in0_index_h_offset += in0_block_w;
                    }

                    if (last_out) {
                        // ── SDDMM Hadamard multiply ─────────────────
                        // Step 1: Pack matmul result to intermediate c_24
                        cb_reserve_back(cb_intermed, out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            pack_tile(i, cb_intermed);
                        }
                        cb_push_back(cb_intermed, out_subblock_num_tiles);
                        release_dst();

                        // Step 2: Element-wise multiply: B[i,j] (c_0) * intermediate (c_24)
                        acquire_dst();
                        mul_tiles_init(cb_sparse, cb_intermed);
                        cb_wait_front(cb_intermed, out_subblock_num_tiles);

                        // Compute offset into the full Rt*Ct sparse mask for this subblock
                        uint32_t sparse_subblock_offset =
                            in0_subblock * out_subblock_h * in1_per_core_w +
                            in1_subblock * out_subblock_w;

                        int mul_dst_index = 0;
                        for (uint32_t h = 0; h < out_subblock_h; h++) {
                            for (uint32_t w = 0; w < out_subblock_w; w++) {
                                uint32_t sparse_tile_idx = sparse_subblock_offset + h * in1_per_core_w + w;
                                uint32_t intermed_tile_idx = h * out_subblock_w + w;
                                mul_tiles(
                                    cb_sparse, cb_intermed,
                                    sparse_tile_idx, intermed_tile_idx,
                                    mul_dst_index);
                                mul_dst_index++;
                            }
                        }
                        cb_pop_front(cb_intermed, out_subblock_num_tiles);

                        // Step 3: Pack result to output c_16
                        cb_reserve_back(cb_out, out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            pack_tile(i, cb_out);
                        }
                        cb_push_back(cb_out, out_subblock_num_tiles);
                        release_dst();

                        // Re-init matmul for next output block
                        mm_init_short(cb_dense_c, cb_dense_d);
                    } else {
                        // ── Spill partial result to c_24 ─────────────
                        cb_reserve_back(cb_intermed, out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            pack_tile(i, cb_intermed);
                        }
                        cb_push_back(cb_intermed, out_subblock_num_tiles);
                        release_dst();
                    }

                    in1_index_subblock_offset += out_subblock_w;
                }
                in0_index_subblock_offset += in0_subblock_num_tiles;
            }

            if (spill) {
                enable_reload = true;
            }

            cb_pop_front(cb_dense_c, in0_block_num_tiles);
            cb_pop_front(cb_dense_d, in1_block_num_tiles);
        }

        // Pop sparse mask ONCE after all reduction steps for this output block
        cb_pop_front(cb_sparse, out_block_num_tiles);
    }
#endif // SKIP_COMPUTE
};
}
