#pragma once

#include <sys/types.h>
#include <algorithm>
#include <cstdint>

#include "include_me.hpp"
#include "host_code_utils.hpp"
#include "tt-metalium/assert.hpp"
#include "tt-metalium/bfloat16.hpp"
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/host_api.hpp"

// profiler includes
#include <system_error>
#include <tracy/Tracy.hpp>
#include <common/TracyColor.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include "hostdevcommon/profiler_common.h"

using namespace tt;
using namespace tt::tt_metal;

using CoreSpec = std::variant<CoreCoord, CoreRange, CoreRangeSet>;

namespace bsr_host_code {

// Re-export shared buffer helpers so SpMM code can call them unqualified
using sparse_common::MakeBuffer;
using sparse_common::MakeCircularBuffer;
using sparse_common::MakeCircularBufferFP32;

// ── Algorithm declarations (3 algorithms for SC26 submission) ──

// Paper name: "Naive"
template<bool verbose = false, bool is_profiling = false, bool use_optimal_noc = true>
void bsr_spmm_multicore_load_balanced_new_DM(
    bsr_matrix<bfloat16>& a,
    dense_matrix<bfloat16>& b,
    dense_matrix<bfloat16>& output,
    bool bcast_batch,
    uint32_t nnz_blocks,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t R,
    uint32_t C,
    uint32_t B,
    IDevice* device);

// Paper name: "SnF in0 naive in1"
template<bool verbose = false, bool is_profiling = false, bool use_optimal_noc = true>
void bsr_spmm_multicore_snf(
    bsr_matrix<bfloat16>& a,
    dense_matrix<bfloat16>& b,
    dense_matrix<bfloat16>& output,
    bool bcast_batch,
    uint32_t nnz_blocks,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t R,
    uint32_t C,
    uint32_t B,
    IDevice* device);

// Paper name: "SnF in0 CDA in1"
template<bool verbose = false, bool is_profiling = false, bool use_optimal_noc = true,
         bool in0_left_to_right = true, bool in1_bottom_to_top = true>
void bsr_spmm_multicore_snfin0_cdain1(
    bsr_matrix<bfloat16>& a,
    dense_matrix<bfloat16>& b,
    dense_matrix<bfloat16>& output,
    bool bcast_batch,
    uint32_t nnz_blocks,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t R,
    uint32_t C,
    uint32_t B,
    IDevice* device);

// ── Ablation skip wrapper declarations ──

#define DECLARE_ABLATION_WRAPPERS(func_name) \
template<bool verbose = false, bool is_profiling = false, bool use_optimal_noc = true> \
void func_name##_no_a_read( \
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output, \
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K, \
    uint32_t R, uint32_t C, uint32_t B, IDevice* device); \
template<bool verbose = false, bool is_profiling = false, bool use_optimal_noc = true> \
void func_name##_no_b_read( \
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output, \
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K, \
    uint32_t R, uint32_t C, uint32_t B, IDevice* device); \
template<bool verbose = false, bool is_profiling = false, bool use_optimal_noc = true> \
void func_name##_no_compute( \
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output, \
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K, \
    uint32_t R, uint32_t C, uint32_t B, IDevice* device); \
template<bool verbose = false, bool is_profiling = false, bool use_optimal_noc = true> \
void func_name##_no_write( \
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output, \
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K, \
    uint32_t R, uint32_t C, uint32_t B, IDevice* device);

DECLARE_ABLATION_WRAPPERS(bsr_spmm_multicore_load_balanced_new_DM)
DECLARE_ABLATION_WRAPPERS(bsr_spmm_multicore_snf)
#undef DECLARE_ABLATION_WRAPPERS

// No-load-balance variants of Naive and SnF
template<bool verbose = false, bool is_profiling = false, bool use_optimal_noc = true>
void bsr_spmm_multicore_load_balanced_new_DM_no_lb(
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output,
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K,
    uint32_t R, uint32_t C, uint32_t B, IDevice* device);

template<bool verbose = false, bool is_profiling = false, bool use_optimal_noc = true>
void bsr_spmm_multicore_snf_no_lb(
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output,
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K,
    uint32_t R, uint32_t C, uint32_t B, IDevice* device);

// 5-param version for snfin0_cdain1 (adds in0_left_to_right, in1_bottom_to_top)
#define DECLARE_ABLATION_WRAPPERS_5(func_name) \
template<bool verbose = false, bool is_profiling = false, bool use_optimal_noc = true, \
         bool in0_left_to_right = true, bool in1_bottom_to_top = true> \
void func_name##_no_a_read( \
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output, \
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K, \
    uint32_t R, uint32_t C, uint32_t B, IDevice* device); \
template<bool verbose = false, bool is_profiling = false, bool use_optimal_noc = true, \
         bool in0_left_to_right = true, bool in1_bottom_to_top = true> \
void func_name##_no_b_read( \
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output, \
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K, \
    uint32_t R, uint32_t C, uint32_t B, IDevice* device); \
template<bool verbose = false, bool is_profiling = false, bool use_optimal_noc = true, \
         bool in0_left_to_right = true, bool in1_bottom_to_top = true> \
void func_name##_no_compute( \
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output, \
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K, \
    uint32_t R, uint32_t C, uint32_t B, IDevice* device); \
template<bool verbose = false, bool is_profiling = false, bool use_optimal_noc = true, \
         bool in0_left_to_right = true, bool in1_bottom_to_top = true> \
void func_name##_no_write( \
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output, \
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K, \
    uint32_t R, uint32_t C, uint32_t B, IDevice* device);

DECLARE_ABLATION_WRAPPERS_5(bsr_spmm_multicore_snfin0_cdain1)
#undef DECLARE_ABLATION_WRAPPERS_5

// No-load-balance variant of CDA
template<bool verbose = false, bool is_profiling = false, bool use_optimal_noc = true,
         bool in0_left_to_right = true, bool in1_bottom_to_top = true>
void bsr_spmm_multicore_snfin0_cdain1_no_lb(
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output,
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K,
    uint32_t R, uint32_t C, uint32_t B, IDevice* device);

// ── Function pointer type ──

using HostCodeFunctionPtr = void (*)(
    bsr_matrix<bfloat16>& a,
    dense_matrix<bfloat16>& b,
    dense_matrix<bfloat16>& output,
    bool bcast_batch,
    uint32_t nnz_blocks,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t R,
    uint32_t C,
    uint32_t B,
    IDevice* device);

// ── Verbose Host Code Registry (for testing, verbose=true, profiling=false) ──
// Indices: 0=Naive, 1=SnF, 2=CDA

static std::pair<HostCodeFunctionPtr, std::string> HostCodeRegistryVerbose[] = {
    {bsr_spmm_multicore_load_balanced_new_DM<true, false>, "bsr_spmm_multicore_load_balanced_new_DM"},       // 0
    {bsr_spmm_multicore_snf<true, false>, "bsr_spmm_multicore_snf"},                                         // 1
    {bsr_spmm_multicore_snfin0_cdain1<true, false>, "bsr_spmm_multicore_snfin0_cdain1"},                     // 2
    {bsr_spmm_multicore_snfin0_cdain1_no_lb<true, false>, "bsr_spmm_multicore_snfin0_cdain1_no_lb"},         // 3
    {bsr_spmm_multicore_load_balanced_new_DM_no_lb<true, false>, "bsr_spmm_multicore_load_balanced_new_DM_no_lb"}, // 4
    {bsr_spmm_multicore_snf_no_lb<true, false>, "bsr_spmm_multicore_snf_no_lb"},                             // 5
};

// ── Host Code Registry (15 entries: 3 algos x 5 variants) ──
// Indices: [0-2] full, [3-5] no_a_read, [6-8] no_b_read, [9-11] no_compute, [12-14] no_write
// Within each group: 0=Naive, 1=SnF, 2=CDA

static std::pair<HostCodeFunctionPtr, std::string> HostCodeRegistryProfiling[] = {
    // [0-2] Full algorithms
    {bsr_spmm_multicore_load_balanced_new_DM<false, true>, "bsr_spmm_multicore_load_balanced_new_DM"},
    {bsr_spmm_multicore_snf<false, true>, "bsr_spmm_multicore_snf"},
    {bsr_spmm_multicore_snfin0_cdain1<false, true>, "bsr_spmm_multicore_snfin0_cdain1"},
    // [3-5] no_a_read ablations (SKIP_IN0_DRAM_READ=1)
    {bsr_spmm_multicore_load_balanced_new_DM_no_a_read<false, true>, "bsr_spmm_multicore_load_balanced_new_DM_no_a_read"},
    {bsr_spmm_multicore_snf_no_a_read<false, true>, "bsr_spmm_multicore_snf_no_a_read"},
    {bsr_spmm_multicore_snfin0_cdain1_no_a_read<false, true>, "bsr_spmm_multicore_snfin0_cdain1_no_a_read"},
    // [6-8] no_b_read ablations (SKIP_IN1_DRAM_READ=1)
    {bsr_spmm_multicore_load_balanced_new_DM_no_b_read<false, true>, "bsr_spmm_multicore_load_balanced_new_DM_no_b_read"},
    {bsr_spmm_multicore_snf_no_b_read<false, true>, "bsr_spmm_multicore_snf_no_b_read"},
    {bsr_spmm_multicore_snfin0_cdain1_no_b_read<false, true>, "bsr_spmm_multicore_snfin0_cdain1_no_b_read"},
    // [9-11] no_compute ablations (SKIP_COMPUTE=1)
    {bsr_spmm_multicore_load_balanced_new_DM_no_compute<false, true>, "bsr_spmm_multicore_load_balanced_new_DM_no_compute"},
    {bsr_spmm_multicore_snf_no_compute<false, true>, "bsr_spmm_multicore_snf_no_compute"},
    {bsr_spmm_multicore_snfin0_cdain1_no_compute<false, true>, "bsr_spmm_multicore_snfin0_cdain1_no_compute"},
    // [12-14] no_write ablations (SKIP_DRAM_WRITE=1)
    {bsr_spmm_multicore_load_balanced_new_DM_no_write<false, true>, "bsr_spmm_multicore_load_balanced_new_DM_no_write"},
    {bsr_spmm_multicore_snf_no_write<false, true>, "bsr_spmm_multicore_snf_no_write"},
    {bsr_spmm_multicore_snfin0_cdain1_no_write<false, true>, "bsr_spmm_multicore_snfin0_cdain1_no_write"},
    // [15] CDA without load balancing (SKIP_LOAD_BALANCE=1)
    {bsr_spmm_multicore_snfin0_cdain1_no_lb<false, true>, "bsr_spmm_multicore_snfin0_cdain1_no_lb"},
    // [16] Naive without load balancing
    {bsr_spmm_multicore_load_balanced_new_DM_no_lb<false, true>, "bsr_spmm_multicore_load_balanced_new_DM_no_lb"},
    // [17] SnF without load balancing
    {bsr_spmm_multicore_snf_no_lb<false, true>, "bsr_spmm_multicore_snf_no_lb"},
};

// Sorting permutation utility (used by load-balanced algorithms)
template<class Vals>
void sortingPermutation(const Vals& values, std::vector<int>& v){
    int size = values.size();
    v.clear(); v.reserve(size);
    for(int i=0; i < size; ++i)
        v.push_back(i);

    std::sort(v.begin(), v.end(), [&values](int a, int b) -> bool {
        return values[a] > values[b];
    });
}

// Utility declarations used by host code implementations
CoreCoord clamped_prev(const std::vector<CoreCoord>& order, uint32_t index);
CoreCoord clamped_next(const std::vector<CoreCoord>& order, uint32_t index);
uint32_t _get_maximum_block_dim_with_NoC_args(int32_t block_dim, int32_t in0_block_w, int32_t num_tiles_in_NoC_args);
uint32_t get_Npc_from_BSR_block_size(uint32_t Nt, uint32_t Mpc, uint32_t in0_block_w, uint32_t num_cores_x, uint32_t num_cores_y, uint32_t num_tiles_for_indexing, uint32_t nnz_rows);

} // namespace bsr_host_code
