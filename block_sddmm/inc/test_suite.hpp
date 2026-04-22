#pragma once

#include <cstdint>
#include <random>
#include <string>
#include "include_me.hpp"
#include "sparse_common/bsr_matrix.hpp"
#include "tt-metalium/bfloat16.hpp"

using namespace tt;

namespace bsr_sddmm_test_suite {

    // SDDMM test cases return: sampling mask (BSR), dense C, dense D, test name.
    // SDDMM: A = B ⊙ (C × D)
    // Verification: compare device output against sampling_mask.sddmm(c, d) on CPU.

    using SDDMMTestReturnType = std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, dense_matrix<bfloat16>, std::string>;

    // Helper to build a name suffix from block size
    template<uint32_t R, uint32_t C>
    std::string block_suffix() {
        return "_R" + std::to_string(R) + "C" + std::to_string(C);
    }

    // Single block: 1×1 block grid, trivial case
    template<uint32_t R = 32, uint32_t C = 32>
    inline SDDMMTestReturnType test_single_block() {
        bsr_matrix<bfloat16> mask(R, C, R, C, 1, FILL_ROW, UNIFORM);
        dense_matrix<bfloat16> c_mat(R, 2 * C, ARANGE);
        dense_matrix<bfloat16> d_mat(2 * C, C, UNIFORM);
        return {mask, c_mat, d_mat, "single_block" + block_suffix<R, C>()};
    }

    // Diagonal mask: only diagonal blocks are sampled
    template<uint32_t R = 32, uint32_t C = 32>
    inline SDDMMTestReturnType test_diagonal_mask() {
        bsr_matrix<bfloat16> mask(4 * R, 4 * C, R, C, 4, FILL_DIAG, UNIFORM);
        dense_matrix<bfloat16> c_mat(4 * R, 2 * C, ARANGE);
        dense_matrix<bfloat16> d_mat(2 * C, 4 * C, UNIFORM);
        return {mask, c_mat, d_mat, "diagonal_mask" + block_suffix<R, C>()};
    }

    // Full mask: all blocks nonzero (degenerates to full GEMM then hadamard)
    template<uint32_t R = 32, uint32_t C = 32>
    inline SDDMMTestReturnType test_full_mask() {
        // 2×2 block grid, all 4 blocks present
        bsr_matrix<bfloat16> mask(2 * R, 2 * C, R, C, 4, FILL_ROW, UNIFORM);
        dense_matrix<bfloat16> c_mat(2 * R, C, ARANGE);
        dense_matrix<bfloat16> d_mat(C, 2 * C, UNIFORM);
        return {mask, c_mat, d_mat, "full_mask" + block_suffix<R, C>()};
    }

    // Identity mask: mask blocks are identity matrices
    // Output should equal corresponding blocks of C×D
    template<uint32_t R = 32, uint32_t C = 32>
    inline SDDMMTestReturnType test_identity_mask() {
        bsr_matrix<bfloat16> mask(4 * R, 4 * C, R, C, 4, FILL_DIAG, ID);
        dense_matrix<bfloat16> c_mat(4 * R, 2 * C, ARANGE);
        dense_matrix<bfloat16> d_mat(2 * C, 4 * C, UNIFORM);
        return {mask, c_mat, d_mat, "identity_mask" + block_suffix<R, C>()};
    }

    // Random sparse mask: general case with random sparsity pattern
    template<uint32_t R = 32, uint32_t C = 32>
    inline SDDMMTestReturnType test_random_sparse_mask() {
        // 8×8 block grid, ~25% density (16 blocks out of 64)
        bsr_matrix<bfloat16> mask(8 * R, 8 * C, R, C, 16, UNIFORM);
        dense_matrix<bfloat16> c_mat(8 * R, 4 * C, ARANGE);
        dense_matrix<bfloat16> d_mat(4 * C, 8 * C, UNIFORM);
        return {mask, c_mat, d_mat, "random_sparse_mask" + block_suffix<R, C>()};
    }

    // Nonsquare: tall mask × wide dense product
    template<uint32_t R = 32, uint32_t C = 32>
    inline SDDMMTestReturnType test_nonsquare() {
        // mask: 8×4 block grid, C: 8R×2C, D: 2C×4C
        bsr_matrix<bfloat16> mask(8 * R, 4 * C, R, C, 8, FILL_ROW, UNIFORM);
        dense_matrix<bfloat16> c_mat(8 * R, 2 * C, ARANGE);
        dense_matrix<bfloat16> d_mat(2 * C, 4 * C, UNIFORM);
        return {mask, c_mat, d_mat, "nonsquare" + block_suffix<R, C>()};
    }

    // Large random: bigger test for stress testing
    template<uint32_t R = 32, uint32_t C = 32>
    inline SDDMMTestReturnType test_large_random() {
        // 16×16 block grid, ~10% density
        bsr_matrix<bfloat16> mask(16 * R, 16 * C, R, C, 26, UNIFORM);
        dense_matrix<bfloat16> c_mat(16 * R, 8 * C, ARANGE);
        dense_matrix<bfloat16> d_mat(8 * C, 16 * C, UNIFORM);
        return {mask, c_mat, d_mat, "large_random" + block_suffix<R, C>()};
    }

    // Diagonal mask with K=C (single reduction step) for large block sizes
    template<uint32_t R = 32, uint32_t C = 32>
    inline SDDMMTestReturnType test_diagonal_mask_short_k() {
        bsr_matrix<bfloat16> mask(4 * R, 4 * C, R, C, 4, FILL_DIAG, UNIFORM);
        dense_matrix<bfloat16> c_mat(4 * R, C, ARANGE);
        dense_matrix<bfloat16> d_mat(C, 4 * C, UNIFORM);
        return {mask, c_mat, d_mat, "diagonal_mask_shortk" + block_suffix<R, C>()};
    }

    // Nonsquare with K=C (single reduction step) for large block sizes
    template<uint32_t R = 32, uint32_t C = 32>
    inline SDDMMTestReturnType test_nonsquare_short_k() {
        bsr_matrix<bfloat16> mask(8 * R, 4 * C, R, C, 8, FILL_ROW, UNIFORM);
        dense_matrix<bfloat16> c_mat(8 * R, C, ARANGE);
        dense_matrix<bfloat16> d_mat(C, 4 * C, UNIFORM);
        return {mask, c_mat, d_mat, "nonsquare_shortk" + block_suffix<R, C>()};
    }

    // Fixed sparsity pattern test: reproduces a deadlock found in ProfileSweepK registry 0
    // M=8192, N=8192, K=512, R=C=256, 256 blocks (25% density)
    inline SDDMMTestReturnType test_deadlock_sweepK_0() {
        constexpr uint32_t R = 256, C = 256;
        constexpr uint32_t M = 8192, N = 8192, K = 512;
        constexpr size_t nblocks = 256;

        std::vector<int> indptr = {
            0, 7, 13, 20, 26, 32, 36, 46, 53, 63, 71, 77, 91, 96, 104, 111, 117,
            127, 142, 147, 155, 164, 174, 182, 191, 201, 208, 214, 225, 233, 240, 246, 256
        };
        std::vector<int> indices = {
            5, 9, 11, 15, 20, 26, 31,
            0, 4, 6, 13, 16, 17,
            8, 9, 11, 15, 23, 24, 31,
            2, 4, 12, 23, 24, 25,
            0, 1, 3, 4, 23, 31,
            15, 17, 22, 23,
            0, 1, 6, 8, 13, 17, 19, 26, 27, 31,
            9, 14, 21, 26, 28, 30, 31,
            4, 8, 20, 22, 23, 24, 25, 28, 29, 31,
            2, 9, 10, 12, 15, 21, 22, 30,
            1, 2, 4, 13, 21, 26,
            1, 3, 4, 7, 9, 11, 12, 18, 19, 23, 24, 26, 29, 31,
            11, 15, 19, 21, 22,
            1, 2, 4, 7, 11, 12, 16, 24,
            4, 6, 9, 12, 13, 22, 24,
            3, 13, 21, 22, 27, 28,
            2, 9, 12, 15, 16, 20, 27, 28, 29, 30,
            1, 2, 3, 7, 11, 12, 13, 18, 21, 25, 26, 27, 28, 30, 31,
            8, 10, 12, 17, 28,
            5, 6, 7, 15, 16, 17, 26, 28,
            1, 8, 10, 15, 16, 22, 27, 28, 29,
            4, 6, 9, 11, 16, 21, 23, 24, 25, 29,
            0, 1, 6, 7, 20, 23, 26, 29,
            1, 3, 7, 9, 14, 17, 19, 28, 29,
            1, 4, 11, 13, 14, 16, 18, 24, 27, 29,
            9, 12, 23, 25, 26, 28, 29,
            14, 15, 16, 20, 25, 30,
            1, 3, 4, 9, 13, 14, 15, 16, 28, 29, 31,
            6, 8, 15, 16, 17, 18, 24, 31,
            3, 12, 18, 19, 21, 22, 26,
            5, 9, 10, 23, 25, 27,
            4, 5, 6, 13, 15, 16, 20, 24, 28, 29,
        };

        // Generate random block data
        size_t block_elems = nblocks * R * C;
        std::vector<bfloat16> block_data(block_elems);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < block_elems; i++) {
            block_data[i] = bfloat16(dist(rng));
        }

        bsr_matrix<bfloat16> mask(std::move(block_data), std::move(indptr), std::move(indices), M, N, R, C, nblocks);
        dense_matrix<bfloat16> c_mat(M, K, RAND);
        dense_matrix<bfloat16> d_mat(K, N, RAND);
        return {mask, c_mat, d_mat, "deadlock_sweepK_0_M8192_N8192_K512_R256_C256_d25"};
    }

    using TestFunctionPtr = SDDMMTestReturnType (*)();

    // Default registry: R=C=32
    static TestFunctionPtr TestRegistry[] = {
        test_single_block<>,           // 0
        test_diagonal_mask<>,          // 1
        test_full_mask<>,              // 2
        test_identity_mask<>,          // 3
        test_random_sparse_mask<>,     // 4
        test_nonsquare<>,              // 5
        test_large_random<>,           // 6
        // R=C=64
        test_single_block<64, 64>,     // 7
        test_diagonal_mask<64, 64>,    // 8
        test_full_mask<64, 64>,        // 9
        test_identity_mask<64, 64>,    // 10
        test_nonsquare<64, 64>,        // 11
        // R=C=128 (full_mask uses K=C → 1 reduction step)
        test_full_mask<128, 128>,      // 12
        // R=C=256 (K=C → 1 reduction step to keep bf16 precision with ARANGE data)
        test_full_mask<256, 256>,              // 13
        test_diagonal_mask_short_k<256, 256>,  // 14
        test_nonsquare_short_k<256, 256>,      // 15
        // Deadlock repro cases
        test_deadlock_sweepK_0,                // 16
    };

} // namespace bsr_sddmm_test_suite
