#pragma once

#include <cstdint>
#include <random>
#include "include_me.hpp"
#include "tt-metalium/bfloat16.hpp"
#include "tt-metalium/bfloat4.hpp"

using namespace tt;

using CoreSpec = std::variant<CoreCoord, CoreRange, CoreRangeSet>;

namespace bsr_test_suite {

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_big_zero_rows_more();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_big_zero_rows();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_big_diag();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_random();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_random_tall_blocks();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_random_wide_blocks();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_random_wide();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_random_tall();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_checkerboard();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_big_random();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_big_dense();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_basic();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_2_blocks();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_2_blocks_diag();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_diag_times_wide();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_diag_first_row_times_wide();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_diag_second_row_times_wide();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_simplified_times_wide();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_simplified_tall_times_wide();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_simplified_tall_times_wide_v2();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_big_block_times_wide();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_2_blocks_fill_row();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_2_blocks_fill_col();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_4_blocks();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_off_diag_first_row();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_diag_first_row();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_nonsquare();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_2_blocks_nonsquare();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_nonsquare_tall();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_2_blocks_nonsquare_tall();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_nonsquare_diag_blocks();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_nonsquare_diag_first_row();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_nonsquare_off_diag_first_row();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_simplified_off_diag_first_row();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_nonsquare_diag_tall();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_many_nonsquare();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_nonsquare_stacked();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_2_blocks_col();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_2_blocks_row_simplified();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_2_blocks_col_simplified();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_dense();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_many_empty_rows();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_dense_2_blocks_per_core();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_huge_diag();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_top_row_many_nonsquare();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_bottom_row_many_nonsquare();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_bottom_row_square();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_bottom_row_square_one_block();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_bottom_row_square_one_block_v2();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_mid_diag();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_big_dense_no_cheat();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_huge_col();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_huge_col_v2();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_huge_row();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_huge_row_v2();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_num_iters_y_2_odd_empty_rows();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_big_dense_large_K();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_random_hang();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_random_hang_v2();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_big_dense_large_Kv2();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_big_dense_large_Kv3();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_many_iters_x();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_big_dense_large_R();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_big_dense_large_Rv2();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_big_dense_large_Rv3();
    template <uint32_t, uint32_t>
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_profile_case_sparse_fill_column();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_1_block_simplified();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_1_block_id();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_1_block_uniform();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_1_block_arange();

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_enormous();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_many_iters_y();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_many_iters_both();

    template<uint32_t, uint32_t, uint32_t>
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string>  profile_case_sparse_fill_random_large();

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_triangular();


    using TestFunctionPtr = std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> (*)();

    static TestFunctionPtr TestRegistry[] = {
        test_basic, // 0
        test_2_blocks, // 1
        test_2_blocks_col, // 2
        test_2_blocks_col_simplified, // 3
        test_2_blocks_row_simplified, // 4
        test_2_blocks_nonsquare, // 5
        test_many_nonsquare, // 6 FAIL
        test_nonsquare_diag_blocks, // 7 FAIL
        test_nonsquare_tall, // 8
        test_2_blocks_nonsquare_tall, // 9
        test_nonsquare, // 10
        test_nonsquare_diag_tall, // 11
        test_nonsquare_stacked, // 12
        test_nonsquare_diag_first_row, // 13
        test_nonsquare_off_diag_first_row, // 14
        test_simplified_off_diag_first_row, // 15
        test_2_blocks_diag, // 16 FAIL Deadlock
        test_off_diag_first_row, // 17
        test_diag_first_row, // 18
        test_2_blocks_fill_row, // 19
        test_2_blocks_fill_col, // 20
        test_4_blocks, // 21
        test_diag_times_wide, // 22
        test_simplified_times_wide, // 23
        test_simplified_tall_times_wide, // 24
        test_simplified_tall_times_wide_v2, // 25
        test_big_block_times_wide, // 26
        test_diag_first_row_times_wide, // 27
        test_diag_second_row_times_wide, // 28
        test_big_diag, // 29
        test_checkerboard, // 30
        test_random, // 31
        test_random_wide, // 32
        test_random_tall, // 33
        test_random_tall_blocks, // 34
        test_random_wide_blocks, // 35
        test_big_zero_rows, // 36
        test_big_zero_rows_more, // 37
        test_dense, // 38
        test_many_empty_rows, // 39
        test_big_random, // 40
        test_big_dense, // 41
        test_dense_2_blocks_per_core, // 42
        test_huge_diag, // 43
        test_top_row_many_nonsquare, // 44
        test_bottom_row_many_nonsquare, // 45
        test_bottom_row_square, // 46
        test_bottom_row_square_one_block, // 47
        test_bottom_row_square_one_block_v2, // 48
        test_mid_diag, // 49
        test_big_dense_no_cheat, // 50
        test_huge_col, // 51
        test_huge_col_v2, // 52
        test_huge_row, // 53
        test_huge_row_v2, // 54
        test_num_iters_y_2_odd_empty_rows, // 55
        test_big_dense_large_K, // 56
        test_random_hang, // 57
        test_random_hang_v2, // 58
        test_big_dense_large_Kv2, // 59
        test_big_dense_large_Kv3, // 60
        test_many_iters_x, // 61
        test_big_dense_large_R, // 62
        test_big_dense_large_Rv2, // 63
        test_big_dense_large_Rv3, // 64
        test_profile_case_sparse_fill_column<32, 32>, // 65
        test_profile_case_sparse_fill_column<64, 64>, // 66
        test_profile_case_sparse_fill_column<128, 128>, // 67
        test_1_block_simplified, // 68
        test_1_block_arange, // 69
        // test_1_block_uniform, // PCC is a failed metric on this degen case, but the output is correct
        test_1_block_id, // 70
        // test_enormous, // 71
        // test_many_iters_y, // 72
        // test_many_iters_both, // 73
        test_triangular, // 71
    };

    static std::uniform_real_distribution<> dis(-1000.0, 1000.0);

    template <uint32_t R = 32, uint32_t C = 32>
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_profile_case_sparse_fill_column() {
        // matmul params setup
        uint32_t M = 1024;
        uint32_t N = 1024;
        uint32_t K = 1024;
        // block params setup
        uint32_t block_matrix_height = M / R;
        uint32_t nblocks = block_matrix_height;

        // nz blocks fill the first column
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_COL, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[50];
        size_t n = sprintf(buf, "test_profile_case_sparse_fill_column_R%i_C%d", R, C);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_huge_row_v2() {

        // matmul params setup
        uint32_t M = 4096;
        uint32_t N = 32;
        uint32_t K = 4096;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 32;
        uint32_t block_matrix_height = M / R;
        uint32_t nblocks = block_matrix_height;

        // nz blocks fill the diagonal
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_huge_row_v2");
    }
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_huge_row() {

        // matmul params setup
        uint32_t M = 2048;
        uint32_t N = 2048;
        uint32_t K = 2048;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t block_matrix_height = M / R;
        uint32_t nblocks = block_matrix_height;

        // nz blocks fill the diagonal
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_huge_row");
    }
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_huge_col_v2() {

        // matmul params setup
        uint32_t M = 4096;
        uint32_t N = 4096;
        uint32_t K = 32;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 32;
        uint32_t block_matrix_height = M / R;
        uint32_t nblocks = block_matrix_height;

        // nz blocks fill the diagonal
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_COL, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_huge_col_v2");
    }
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_huge_col() {

        // matmul params setup
        uint32_t M = 2048;
        uint32_t N = 2048;
        uint32_t K = 2048;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t block_matrix_height = M / R;
        uint32_t nblocks = block_matrix_height;

        // nz blocks fill the diagonal
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_COL, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_huge_col");
    }
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_huge_diag() {

        // matmul params setup
        uint32_t M = 4096;
        uint32_t N = 4096;
        uint32_t K = 4096;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 32;
        uint32_t block_matrix_height = M / R;
        uint32_t nblocks = block_matrix_height;

        // nz blocks fill the diagonal
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_DIAG, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_huge_diag");
    }
        std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_mid_diag() {

        // matmul params setup
        uint32_t M = 1024;
        uint32_t N = 1024;
        uint32_t K = 1024;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 32;
        uint32_t block_matrix_height = M / R;
        uint32_t nblocks = block_matrix_height;

        // nz blocks fill the diagonal
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_DIAG, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_mid_diag");
    }

            std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_iter_2_2() {

        // matmul params setup
        uint32_t M = 512;
        uint32_t N = 512;
        uint32_t K = 512;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 32;
        uint32_t block_matrix_height = M / R;
        uint32_t nblocks = block_matrix_height;

        // nz blocks fill the diagonal
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_DIAG, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_iter_2_2");
    }


    // this case should expose the performance difference between the naive and new versions
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_many_empty_rows() {
        // matmul params setup
        uint32_t M = 8192;
        uint32_t N = 256;
        uint32_t K = 256;
        // block params setup
        uint32_t R = 256;
        uint32_t C = 256;
        uint32_t nblocks = 1;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);



        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_many_empty_rows");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_big_zero_rows_more() {
        // matmul params setup
        uint32_t M = 256;
        uint32_t N = 256;
        uint32_t K = 256;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 8;
        uint32_t block_matrix_height = M / R;
        std::vector<float> data(R*C*nblocks);
        for (int k = 0; k < nblocks; k++){
            for (int i = 0; i < R*C; i++){
                data[k*R*C + i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            }
        }
        std::vector<int> indptr = {0, 4, 4, 8, 8};
        std::vector<int> indices = {0, 1, 2, 3, 0, 1, 2, 3};

        // all nz on one row
        bsr_matrix<float> bsr(data, indptr, indices, M, K, R, C, nblocks);
        dense_matrix<float> dense(K, N, RAND);



        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_big_zero_rows_more");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_big_zero_rows() {
        // matmul params setup
        uint32_t M = 256;
        uint32_t N = 256;
        uint32_t K = 256;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 4;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);



        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_big_zero_rows");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_big_diag() {
        // matmul params setup
        uint32_t M = 256;
        uint32_t N = 256;
        uint32_t K = 256;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 4;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_DIAG, RAND);
        dense_matrix<float> dense(K, N, RAND);



        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_big_diag");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_random() {
        // matmul params setup
        uint32_t M = 256;
        uint32_t N = 256;
        uint32_t K = 256;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 4;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);



        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_random");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_random_tall_blocks() {
        // matmul params setup
        uint32_t M = 256;
        uint32_t N = 256;
        uint32_t K = 256;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 64;
        uint32_t nblocks = 4;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);



        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_random_tall_blocks");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_random_wide_blocks() {
        // matmul params setup
        uint32_t M = 256;
        uint32_t N = 256;
        uint32_t K = 256;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 32;
        uint32_t nblocks = 4;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);



        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_random_wide_blocks");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_random_wide() {
        // matmul params setup
        uint32_t M = 256;
        uint32_t N = 1024;
        uint32_t K = 256;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 8;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);



        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_random_wide");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_random_tall() {
        // matmul params setup
        uint32_t M = 1024;
        uint32_t N = 256;
        uint32_t K = 256;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 8;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);



        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_random_tall");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_checkerboard() {
        // matmul params setup
        uint32_t M = 256;
        uint32_t N = 256;
        uint32_t K = 256;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 8;
        uint32_t block_matrix_height = M / R;

        std::vector<float> data(R*C*nblocks);
        for (int k = 0; k < nblocks; k++){
            for (int i = 0; i < R*C; i++){
                data[k*R*C + i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            }
        }
        std::vector<int> indptr = {0, 2, 4, 6, 8};
        std::vector<int> indices = {0, 2, 1, 3, 0, 2, 1, 3};

        bsr_matrix<float> bsr(data, indptr, indices, M, K, R, C, nblocks);
        dense_matrix<float> dense(K, N, RAND);



        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_checkerboard");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_random_hang() {
        // matmul params setup
        uint32_t M = 1024;
        uint32_t N = 1024;
        uint32_t K = 1024;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 32;
        uint32_t block_matrix_height = M / R;

        std::vector<float> data(R*C*nblocks);
        for (int k = 0; k < nblocks; k++){
            for (int i = 0; i < R*C; i++){
                data[k*R*C + i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            }
        }
        std::vector<int> indptr = {0, 0, 1, 3, 6, 7, 11, 13, 16, 18, 21, 25, 27, 28, 29, 31, 32};
        std::vector<int> indices = {13,6, 13, 0, 8, 11, 5, 4, 8, 10, 12, 6, 14, 4, 14, 15, 4, 5, 2, 7, 8, 4, 8, 11, 12, 3, 10, 0, 3, 0, 4, 6};

        // all nz on one row
        bsr_matrix<float> bsr(data, indptr, indices, M, K, R, C, nblocks);

        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_random_hang");
    }

        std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_random_hang_v2() {
        // matmul params setup
        uint32_t M = 1024;
        uint32_t N = 1024;
        uint32_t K = 1024;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 32;
        uint32_t block_matrix_height = M / R;

        std::vector<float> data(R*C*nblocks);
        for (int k = 0; k < nblocks; k++){
            for (int i = 0; i < R*C; i++){
                data[k*R*C + i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            }
        }
        std::vector<int> indptr = {0, 0, 3, 4, 4, 6, 11, 14, 16, 16, 18, 19, 20, 22, 24, 29, 32};
        std::vector<int> indices = {0, 3, 4, 11, 1, 2, 0, 2, 6, 11, 14, 7, 10, 14, 0, 9, 13, 14, 1, 14, 9, 15, 1, 3, 3, 7, 11, 13, 14, 8, 9, 15};

        // all nz on one row
        bsr_matrix<float> bsr(data, indptr, indices, M, K, R, C, nblocks);

        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_random_hang_v2");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_num_iters_y_2_odd_empty_rows() {
        // matmul params setup
        uint32_t M = 1024;
        uint32_t N = 1024;
        uint32_t K = 1024;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 8;
        uint32_t block_matrix_height = M / R;

        std::vector<float> data(R*C*nblocks);
        for (int k = 0; k < nblocks; k++){
            for (int i = 0; i < R*C; i++){
                data[k*R*C + i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            }
        }
        std::vector<int> indptr = {0, 1, 1, 1, 2, 2, 3, 3, 3, 6, 7, 7, 7, 7, 7, 7, 8};
        std::vector<int> indices = {0, 0, 0, 1, 2, 0, 0, 0};

        // all nz on one row
        bsr_matrix<float> bsr(data, indptr, indices, M, K, R, C, nblocks);

        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_num_iters_y_2_odd_empty_rows");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_big_random() {
        // matmul params setup
        uint32_t M = 1024;
        uint32_t N = 1024;
        uint32_t K = 1024;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 32;
        uint32_t block_matrix_height = M / R;

        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_big_random");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_big_dense() {
        // matmul params setup
        uint32_t M = 4096;
        uint32_t N = 4096;
        uint32_t K = 512;
        // block size become 512x64, with 64 input blocks
        dense_matrix<float> tmp(M, K, RAND);
        bsr_matrix<float> bsr(tmp, N);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_big_dense");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_big_dense_no_cheat() {
        // matmul params setup
        uint32_t M = 4096;
        uint32_t N = 4096;
        uint32_t K = 512;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 512;
        uint32_t block_matrix_height = M / R;

        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_big_dense_no_cheat");
    }
        std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_enormous() {
        // matmul params setup
        uint32_t M = 2 << 14;
        uint32_t N = 2 << 14;
        uint32_t K = 2 << 14;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 32;
        uint32_t block_matrix_height = M / R;
        uint32_t block_matrix_width = K / C;
        uint32_t nblocks = (block_matrix_height * block_matrix_width) / 4;

        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_enormous");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_big_dense_large_R() {
        // matmul params setup
        uint32_t M = 4096;
        uint32_t N = 4096;
        uint32_t K = 512;
        // block params setup
        uint32_t R = 1024;
        uint32_t C = 64;
        uint32_t nblocks = 32;
        uint32_t block_matrix_height = M / R;

        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_big_dense_large_R");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_big_dense_large_Rv2() {
        // matmul params setup
        uint32_t M = 1024;
        uint32_t N = 2048;
        uint32_t K = 512;
        // block params setup
        uint32_t R = 1024;
        uint32_t C = 64;
        uint32_t nblocks = 8;
        uint32_t block_matrix_height = M / R;

        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_big_dense_large_Rv2");
    }

        std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_big_dense_large_Rv3() {
        // matmul params setup
        uint32_t M = 512;
        uint32_t N = 8192;
        uint32_t K = 512;
        // block params setup
        uint32_t R = 512;
        uint32_t C = 64;
        uint32_t nblocks = 8;
        uint32_t block_matrix_height = M / R;

        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_big_dense_large_Rv3");
    }
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_many_iters_y() {
        // matmul params setup
        uint32_t M = 32768;
        uint32_t N = 512;
        uint32_t K = 512;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t block_matrix_height = M / R;
        uint32_t block_matrix_width = K / C;
        uint32_t nblocks = (block_matrix_height * block_matrix_width) / 4;

        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_many_iters_y");
    }
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_many_iters_x() {
        // matmul params setup
        uint32_t M = 512;
        uint32_t N = 32768;
        uint32_t K = 512;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 32;
        uint32_t block_matrix_height = M / R;

        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_many_iters_x");
    }
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_many_iters_both() {
        // TODO: this test case is VERY interesting now: there are enough nz rows to fill HALF
        //       of the alloted runtime args per kernel (124/256 for reader_in0_snf)
        //       this also takes 10 minutes in my sequential CPU version
        //       while taking one second (less?) on the device
        // matmul params setup
        uint32_t M = 32768;
        uint32_t N = 32768;
        uint32_t K = 512;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t block_matrix_height = M / R;
        uint32_t block_matrix_width = K / C;
        uint32_t nblocks = (block_matrix_height * block_matrix_width) / 4;


        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_many_iters_both");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_small_many_iters_x() {
        // matmul params setup
        uint32_t M = 64;
        uint32_t N = 2 << 16;
        uint32_t K = 64;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 128;
        uint32_t block_matrix_height = M / R;

        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_small_many_iters_x");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_big_dense_large_K() {
        // matmul params setup
        uint32_t M = 32;
        uint32_t N = 32;
        uint32_t K = 1024;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 32;
        uint32_t nblocks = 32;
        uint32_t block_matrix_height = M / R;

        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_big_dense_large_K");
    }
        std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_big_dense_large_Kv2() {
        // matmul params setup
        uint32_t M = 32;
        uint32_t N = 32;
        uint32_t K = 2048;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 32;
        uint32_t nblocks = 32;
        uint32_t block_matrix_height = M / R;

        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_big_dense_large_Kv2");
    }

        std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_big_dense_large_Kv3() {
        // matmul params setup
        uint32_t M = 32;
        uint32_t N = 32;
        uint32_t K = 4096;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 32;
        uint32_t nblocks = 32;
        uint32_t block_matrix_height = M / R;

        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_big_dense_large_Kv3");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_dense_2_blocks_per_core() {
        // matmul params setup

        uint32_t num_cores = 64; // given our wormhole arch, this is how many Tensix tiles we have

        uint32_t M = 1024;
        uint32_t N = 128;
        uint32_t K = 128;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 32;
        uint32_t nblocks = 128;
        uint32_t block_matrix_height = M / R;

        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_dense_2_blocks_per_core");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_dense() {
        uint32_t M = 1024;
        uint32_t N = 256;
        uint32_t K = 64;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 32;
        uint32_t nblocks = 64;
        uint32_t block_matrix_height = M / R;

        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_dense");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_basic() {
        // matmul params setup
        uint32_t M = 64;
        uint32_t N = 64;
        uint32_t K = 64;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 1;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);

        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_basic");
    }


    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_triangular() {
        // matmul params setup
        uint32_t M = 8192;
        uint32_t N = 8192;
        uint32_t K = 8192;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 32;
        uint32_t nblocks = 0; // unused in tril constructor
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_TRIL, RAND);

        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_triangular");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_2_blocks() {
        // matmul params setup
        uint32_t M = 64;
        uint32_t N = 64;
        uint32_t K = 128;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 2;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_2_blocks");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_2_blocks_diag() {
        // matmul params setup
        uint32_t M = 128;
        uint32_t N = 64;
        uint32_t K = 128;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 2;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_DIAG, RAND);
        dense_matrix<float> dense(K, N, RAND);



        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_2_blocks_diag");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_diag_times_wide() {
        // matmul params setup
        uint32_t M = 128;
        uint32_t N = 128;
        uint32_t K = 128;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 2;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_DIAG, RAND);
        dense_matrix<float> dense(K, N, RAND);



        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_diag_times_wide");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_diag_first_row_times_wide() {
        // matmul params setup
        uint32_t M = 64;
        uint32_t N = 128;
        uint32_t K = 128;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 1;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);



        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_diag_first_row_times_wide");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_diag_second_row_times_wide() {
        // matmul params setup
        uint32_t M = 64;
        uint32_t N = 128;
        uint32_t K = 128;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 1;
        uint32_t block_matrix_height = M / R;

        std::vector<float> data(R*C*nblocks);
        for (int k = 0; k < nblocks; k++){
            for (int i = 0; i < R*C; i++){
                data[k*R*C + i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            }
        }
        std::vector<int> indptr = {0, 1};
        std::vector<int> indices = {1};

        bsr_matrix<float> bsr(data, indptr, indices, M, K, R, C, nblocks);
        dense_matrix<float> dense(K, N, RAND);



        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_diag_second_row_times_wide");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_simplified_times_wide() {
        // matmul params setup
        uint32_t M = 32;
        uint32_t N = 64;
        uint32_t K = 32;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 32;
        uint32_t nblocks = 1;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);



        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_simplified_times_wide");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_simplified_tall_times_wide() {
        // matmul params setup
        uint32_t M = 64;
        uint32_t N = 64;
        uint32_t K = 32;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 32;
        uint32_t nblocks = 1;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);



        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_simplified_tall_times_wide");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_simplified_tall_times_wide_v2() {
        // matmul params setup
        uint32_t M = 64;
        uint32_t N = 64;
        uint32_t K = 32;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 32;
        uint32_t nblocks = 1;
        uint32_t block_matrix_height = M / R;

        std::vector<float> data(R*C*nblocks);
        for (int k = 0; k < nblocks; k++){
            for (int i = 0; i < R*C; i++){
                data[k*R*C + i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            }
        }
        std::vector<int> indptr = {0, 0, 1};
        std::vector<int> indices = {0};

        // all nz on one row
        bsr_matrix<float> bsr(data, indptr, indices, M, K, R, C, nblocks);

        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_simplified_tall_times_wide_v2");
    }


    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_bottom_row_square() {
        // matmul params setup
        uint32_t M = 64;
        uint32_t N = 32;
        uint32_t K = 64;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 32;
        uint32_t nblocks = 2;
        uint32_t block_matrix_height = M / R;

        std::vector<float> data(R*C*nblocks);
        for (int k = 0; k < nblocks; k++){
            for (int i = 0; i < R*C; i++){
                data[k*R*C + i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            }
        }
        std::vector<int> indptr = {0, 0, 2};
        std::vector<int> indices = {0, 1};

        // all nz on one row
        bsr_matrix<float> bsr(data, indptr, indices, M, K, R, C, nblocks);
        // for (int i = 0; i < data.size(); i+=32) {
        //     for (int j = 0; j < 32; j++){
        //         std::cout << data[i + j] << ' ';
        //     }
        //     std::cout << std::endl;
        // }
        // std::cout << std::endl;
        // std::cout << std::endl;
        // std::cout << std::endl;

        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_bottom_row_square");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_bottom_row_square_one_block() {
        // matmul params setup
        uint32_t M = 64;
        uint32_t N = 64;
        uint32_t K = 64;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 32;
        uint32_t nblocks = 1;
        uint32_t block_matrix_height = M / R;

        std::vector<float> data(R*C*nblocks);
        for (int k = 0; k < nblocks; k++){
            for (int i = 0; i < R*C; i++){
                data[k*R*C + i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            }
        }
        std::vector<int> indptr = {0, 0, 1};
        std::vector<int> indices = {0};

        // all nz on one row
        bsr_matrix<float> bsr(data, indptr, indices, M, K, R, C, nblocks);

        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_bottom_row_square_one_block");
    }

        std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_bottom_row_square_one_block_v2() {
        // matmul params setup
        uint32_t M = 64;
        uint32_t N = 64;
        uint32_t K = 64;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 32;
        uint32_t nblocks = 1;
        uint32_t block_matrix_height = M / R;

        std::vector<float> data(R*C*nblocks);
        for (int k = 0; k < nblocks; k++){
            for (int i = 0; i < R*C; i++){
                data[k*R*C + i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            }
        }
        std::vector<int> indptr = {0, 0, 1};
        std::vector<int> indices = {1};

        // all nz on one row
        bsr_matrix<float> bsr(data, indptr, indices, M, K, R, C, nblocks);

        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_bottom_row_square_one_block_v2");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_big_block_times_wide() {
        // matmul params setup
        uint32_t M = 64;
        uint32_t N = 1024;
        uint32_t K = 64;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 1;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);



        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_big_block_times_wide");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_2_blocks_fill_row() {
        // matmul params setup
        uint32_t M = 128;
        uint32_t N = 64;
        uint32_t K = 128;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 2;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);



        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_2_blocks_fill_row");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_2_blocks_fill_col() {
        // matmul params setup
        uint32_t M = 128;
        uint32_t N = 64;
        uint32_t K = 128;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 2;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_COL, RAND);
        dense_matrix<float> dense(K, N, RAND);



        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_2_blocks_fill_col");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_4_blocks() {
        // matmul params setup
        uint32_t M = 128;
        uint32_t N = 64;
        uint32_t K = 128;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 4;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_COL, RAND);
        dense_matrix<float> dense(K, N, RAND);



        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_4_blocks");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_off_diag_first_row() {
        // matmul params setup
        uint32_t M = 64;
        uint32_t N = 64;
        uint32_t K = 128;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 1;
        uint32_t block_matrix_height = M / R;

        std::vector<float> data(R*C*nblocks);
        for (int k = 0; k < nblocks; k++){
            for (int i = 0; i < R*C; i++){
                data[k*R*C + i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            }
        }
        std::vector<int> indptr = {0, 1};
        std::vector<int> indices = {1};

        bsr_matrix<float> bsr(data, indptr, indices, M, K, R, C, nblocks);

        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_off_diag_first_row");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_diag_first_row() {
        // matmul params setup
        uint32_t M = 64;
        uint32_t N = 64;
        uint32_t K = 128;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 1;
        uint32_t block_matrix_height = M / R;

        std::vector<float> data(R*C*nblocks);
        for (int k = 0; k < nblocks; k++){
            for (int i = 0; i < R*C; i++){
                data[k*R*C + i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            }
        }
        std::vector<int> indptr = {0, 1};
        std::vector<int> indices = {0};

        bsr_matrix<float> bsr(data, indptr, indices, M, K, R, C, nblocks);

        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_diag_first_row");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_nonsquare() {
        // matmul params setup
        uint32_t M = 32;
        uint32_t N = 32;
        uint32_t K = 64;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 64;
        uint32_t nblocks = 1;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);

        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_nonsquare");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_2_blocks_nonsquare() {
        // matmul params setup
        uint32_t M = 32;
        uint32_t N = 32;
        uint32_t K = 128;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 64;
        uint32_t nblocks = 2;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);

        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_2_blocks_nonsquare");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_nonsquare_tall() {
        // matmul params setup
        uint32_t M = 64;
        uint32_t N = 32;
        uint32_t K = 32;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 32;
        uint32_t nblocks = 1;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_nonsquare_tall");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_2_blocks_nonsquare_tall() {
        // matmul params setup
        uint32_t M = 64;
        uint32_t N = 32;
        uint32_t K = 64;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 32;
        uint32_t nblocks = 2;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_2_blocks_nonsquare_tall");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_nonsquare_diag_blocks() {
        // matmul params setup
        uint32_t M = 64;
        uint32_t N = 32;
        uint32_t K = 128;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 64;
        uint32_t nblocks = 2;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_DIAG, RAND);

        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_nonsquare_diag_blocks");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_nonsquare_diag_first_row() {
        // matmul params setup
        uint32_t M = 32;
        uint32_t N = 32;
        uint32_t K = 128;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 64;
        uint32_t nblocks = 1;
        uint32_t block_matrix_height = M / R;

        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_DIAG, RAND);

        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_nonsquare_diag_first_row");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_nonsquare_off_diag_first_row() {
        // matmul params setup
        uint32_t M = 32;
        uint32_t N = 32;
        uint32_t K = 128;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 64;
        uint32_t nblocks = 1;
        uint32_t block_matrix_height = M / R;

        std::vector<float> data(R*C*nblocks);
        for (int k = 0; k < nblocks; k++){
            for (int i = 0; i < R*C; i++){
                data[k*R*C + i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                // data[k*R*C + i] = i;
            }
        }
        std::vector<int> indptr = {0, 1};
        std::vector<int> indices = {1};

        bsr_matrix<float> bsr(data, indptr, indices, M, K, R, C, nblocks);

        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_nonsquare_off_diag_first_row");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_simplified_off_diag_first_row() {
        // matmul params setup
        uint32_t M = 32;
        uint32_t N = 32;
        uint32_t K = 64;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 32;
        uint32_t nblocks = 1;
        uint32_t block_matrix_height = M / R;

        std::vector<float> data(R*C*nblocks);
        for (int k = 0; k < nblocks; k++){
            for (int i = 0; i < R*C; i++){
                data[k*R*C + i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                // data[k*R*C + i] = i;
            }
        }
        std::vector<int> indptr = {0, 1};
        std::vector<int> indices = {1};

        bsr_matrix<float> bsr(data, indptr, indices, M, K, R, C, nblocks);

        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_simplified_off_diag_first_row");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_nonsquare_diag_tall() {
        // matmul params setup
        uint32_t M = 128;
        uint32_t N = 32;
        uint32_t K = 64;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 32;
        uint32_t nblocks = 2;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_DIAG, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_nonsquare_diag_tall");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_many_nonsquare() {
        // matmul params setup
        uint32_t M = 64;
        uint32_t N = 32;
        uint32_t K = 128;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 64;
        uint32_t nblocks = 4;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_many_nonsquare");
    }

        std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_top_row_many_nonsquare() {
        // matmul params setup
        uint32_t M = 64;
        uint32_t N = 32;
        uint32_t K = 128;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 64;
        uint32_t nblocks = 2;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_top_row_many_nonsquare");
    }

        std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_bottom_row_many_nonsquare() {
        // matmul params setup
        uint32_t M = 64;
        uint32_t N = 32;
        uint32_t K = 128;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 64;
        uint32_t nblocks = 2;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        std::vector<int> indptr = {0, 0, 2};
        std::vector<int> indices = {0, 1};
        std::vector<float> data(R*C*nblocks);
        for (int k = 0; k < nblocks; k++){
            for (int i = 0; i < R*C; i++){
                data[k*R*C + i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            }
        }

        bsr_matrix<float> bsr(data, indptr, indices, M, K, R, C, nblocks);

        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_bottom_row_many_nonsquare");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_nonsquare_stacked() {
        // matmul params setup
        uint32_t M = 64;
        uint32_t N = 32;
        uint32_t K = 64;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 64;
        uint32_t nblocks = 2;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_nonsquare_stacked");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_2_blocks_col() {
        // matmul params setup
        uint32_t M = 128;
        uint32_t N = 64;
        uint32_t K = 64;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 2;
        uint32_t block_matrix_height = M / R;

        // all nz on one col
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_COL, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_2_blocks_col");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_2_blocks_row_simplified() {
        // matmul params setup
        uint32_t M = 32;
        uint32_t N = 32;
        uint32_t K = 64;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 32;
        uint32_t nblocks = 2;
        uint32_t block_matrix_height = M / R;

        // all nz on one col
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_2_blocks_row_simplified");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_2_blocks_col_simplified() {
        // matmul params setup
        uint32_t M = 64;
        uint32_t N = 32;
        uint32_t K = 32;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 32;
        uint32_t nblocks = 2;
        uint32_t block_matrix_height = M / R;

        // all nz on one col
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_COL, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_2_blocks_col_simplified");
    }


    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_1_block_simplified() {
        // matmul params setup
        uint32_t M = 32;
        uint32_t N = 32;
        uint32_t K = 32;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 32;
        uint32_t nblocks = 1;
        uint32_t block_matrix_height = M / R;

        // all nz on one col
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_COL, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_1_block_simplified");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_1_block_uniform() {
        // matmul params setup
        uint32_t M = 32;
        uint32_t N = 32;
        uint32_t K = 32;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 32;
        uint32_t nblocks = 1;
        uint32_t block_matrix_height = M / R;

        // all nz on one col
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_COL, UNIFORM);
        dense_matrix<float> dense(K, N, UNIFORM);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_1_block_uniform");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_1_block_arange() {
        // matmul params setup
        uint32_t M = 32;
        uint32_t N = 32;
        uint32_t K = 32;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 32;
        uint32_t nblocks = 1;
        uint32_t block_matrix_height = M / R;

        // all nz on one col
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_COL, ARANGE);
        dense_matrix<float> dense(K, N, UNIFORM);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_1_block_arange");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_1_block_id() {
        // matmul params setup
        uint32_t M = 32;
        uint32_t N = 32;
        uint32_t K = 32;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 32;
        uint32_t nblocks = 1;
        uint32_t block_matrix_height = M / R;

        // all nz on one col
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_COL, ID);
        dense_matrix<float> dense(K, N, ID);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_1_block_id");
    }

    template <uint32_t R = 32, uint32_t C = 32, uint32_t DensityPercent = 25>
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string>  profile_case_sparse_fill_random_large() {
        // matmul params setup
        uint32_t M = 8192;
        uint32_t N = 8192;
        uint32_t K = 8192;
        // block params setup
        uint32_t block_matrix_height = M / R;
        uint32_t block_matrix_width = K / C;

        constexpr float density = DensityPercent / 100.0f;
        uint32_t divisor = uint32_t(std::round(1.0 / density));
        uint32_t nblocks = (block_matrix_height * block_matrix_width) / divisor;

        // nz blocks placed randomly
        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[50];
        size_t n = sprintf(buf, "profile_case_sparse_fill_random_large_R%i_C%d", R, C);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }



} // namespace bsr_test_suite

namespace dense_test_suite {

    std::tuple<dense_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> dense_test_0();
    std::tuple<dense_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> dense_test_large();

    using TestDenseFunctionPtr = std::tuple<dense_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> (*)();

    static TestDenseFunctionPtr DenseTestRegistry[] = {
        dense_test_0, // 0
        dense_test_large, // 1
    };

    std::tuple<dense_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> dense_test_0() {
        // matmul params setup
        uint32_t M = 2048;
        uint32_t N = 2048;
        uint32_t K = 512;

        // all nz on one row
        dense_matrix<float> src0(M, K, RAND);
        dense_matrix<float> src1(K, N, RAND);

        dense_matrix<bfloat16> src0_bfoat16 = src0.bfloat16_cast();
        dense_matrix<bfloat16> src1_bfloat16 = src1.bfloat16_cast();
        return std::make_tuple(src0_bfoat16, src1_bfloat16, "dense_test_0");
    }

        std::tuple<dense_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> dense_test_large() {
        // matmul params setup
        uint32_t M = 4096;
        uint32_t N = 4096;
        uint32_t K = 512;

        // all nz on one row
        dense_matrix<float> src0(M, K, RAND);
        dense_matrix<float> src1(K, N, RAND);

        dense_matrix<bfloat16> src0_bfoat16 = src0.bfloat16_cast();
        dense_matrix<bfloat16> src1_bfloat16 = src1.bfloat16_cast();
        return std::make_tuple(src0_bfoat16, src1_bfloat16, "dense_test_large");
    }

} // namespace dense_test_suite
