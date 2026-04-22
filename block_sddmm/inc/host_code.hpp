#pragma once

#include <sys/types.h>
#include <algorithm>
#include <cstdint>

#include "include_me.hpp"
#include "sparse_common/host_code_utils.hpp"
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

namespace bsr_sddmm_host_code {

using sparse_common::MakeBuffer;
using sparse_common::MakeCircularBuffer;

// SDDMM host code function declarations
// Signature: A = B ⊙ (C × D)
//   B: sparse BSR sampling mask (M×N)
//   C: dense (M×K)
//   D: dense (K×N)
//   A: sparse BSR output (M×N, same sparsity pattern as B)
//
// Key properties:
//   - Output sparsity pattern is known a priori (same as B)
//   - Only compute entries of C×D where B is nonzero
//   - Has both N and K params (C is M×K, D is K×N)

template<bool verbose = false, bool is_profiling = false>
void bsr_sddmm_multicore_naive(
    bsr_matrix<bfloat16>& sampling_mask,
    dense_matrix<bfloat16>& c,
    dense_matrix<bfloat16>& d,
    bsr_matrix<bfloat16>& output,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t R,
    uint32_t C_block,
    uint32_t B,
    IDevice* device);

template<bool verbose = false, bool is_profiling = false>
void bsr_sddmm_multicore_CDA(
    bsr_matrix<bfloat16>& sampling_mask,
    dense_matrix<bfloat16>& c,
    dense_matrix<bfloat16>& d,
    bsr_matrix<bfloat16>& output,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t R,
    uint32_t C_block,
    uint32_t B,
    IDevice* device);

// Ablation skip wrapper declarations (no_b_read, no_c_read, no_d_read, no_compute, no_write)
#define DECLARE_SDDMM_ABLATION_WRAPPERS(func_name) \
template<bool verbose = false, bool is_profiling = false> \
void func_name##_no_b_read( \
    bsr_matrix<bfloat16>& sampling_mask, dense_matrix<bfloat16>& c, dense_matrix<bfloat16>& d, \
    bsr_matrix<bfloat16>& output, uint32_t M, uint32_t N, uint32_t K, \
    uint32_t R, uint32_t C_block, uint32_t B, IDevice* device); \
template<bool verbose = false, bool is_profiling = false> \
void func_name##_no_c_read( \
    bsr_matrix<bfloat16>& sampling_mask, dense_matrix<bfloat16>& c, dense_matrix<bfloat16>& d, \
    bsr_matrix<bfloat16>& output, uint32_t M, uint32_t N, uint32_t K, \
    uint32_t R, uint32_t C_block, uint32_t B, IDevice* device); \
template<bool verbose = false, bool is_profiling = false> \
void func_name##_no_d_read( \
    bsr_matrix<bfloat16>& sampling_mask, dense_matrix<bfloat16>& c, dense_matrix<bfloat16>& d, \
    bsr_matrix<bfloat16>& output, uint32_t M, uint32_t N, uint32_t K, \
    uint32_t R, uint32_t C_block, uint32_t B, IDevice* device); \
template<bool verbose = false, bool is_profiling = false> \
void func_name##_no_compute( \
    bsr_matrix<bfloat16>& sampling_mask, dense_matrix<bfloat16>& c, dense_matrix<bfloat16>& d, \
    bsr_matrix<bfloat16>& output, uint32_t M, uint32_t N, uint32_t K, \
    uint32_t R, uint32_t C_block, uint32_t B, IDevice* device); \
template<bool verbose = false, bool is_profiling = false> \
void func_name##_no_write( \
    bsr_matrix<bfloat16>& sampling_mask, dense_matrix<bfloat16>& c, dense_matrix<bfloat16>& d, \
    bsr_matrix<bfloat16>& output, uint32_t M, uint32_t N, uint32_t K, \
    uint32_t R, uint32_t C_block, uint32_t B, IDevice* device);

DECLARE_SDDMM_ABLATION_WRAPPERS(bsr_sddmm_multicore_naive)
DECLARE_SDDMM_ABLATION_WRAPPERS(bsr_sddmm_multicore_CDA)
#undef DECLARE_SDDMM_ABLATION_WRAPPERS

// Function pointer type for SDDMM host code
using SDDMMHostCodeFunctionPtr = void (*)(
    bsr_matrix<bfloat16>& sampling_mask,
    dense_matrix<bfloat16>& c,
    dense_matrix<bfloat16>& d,
    bsr_matrix<bfloat16>& output,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t R,
    uint32_t C_block,
    uint32_t B,
    IDevice* device);

static std::pair<SDDMMHostCodeFunctionPtr, std::string> HostCodeRegistry[] = {
    {bsr_sddmm_multicore_naive<false, false>, "bsr_sddmm_multicore_naive"},
    {bsr_sddmm_multicore_CDA<false, false>, "bsr_sddmm_multicore_CDA"},
};

static std::pair<SDDMMHostCodeFunctionPtr, std::string> HostCodeRegistryVerbose[] = {
    {bsr_sddmm_multicore_naive<true, false>, "bsr_sddmm_multicore_naive"},
    {bsr_sddmm_multicore_CDA<true, false>, "bsr_sddmm_multicore_CDA"},
};

// Profiling registry: 2 full + 10 ablation variants = 12 entries
//   [0-1]   full (naive, CDA)
//   [2-3]   no_b_read  — skip sparse B mask reads
//   [4-5]   no_c_read  — skip dense C reads (+ CDA sharing)
//   [6-7]   no_d_read  — skip dense D reads (+ CDA sharing)
//   [8-9]   no_compute — skip matmul + Hadamard
//   [10-11] no_write   — skip output DRAM writes
static std::pair<SDDMMHostCodeFunctionPtr, std::string> HostCodeRegistryProfiling[] = {
    // Full algorithms
    {bsr_sddmm_multicore_naive<false, true>, "bsr_sddmm_multicore_naive"},
    {bsr_sddmm_multicore_CDA<false, true>, "bsr_sddmm_multicore_CDA"},
    // no_b_read
    {bsr_sddmm_multicore_naive_no_b_read<false, true>, "bsr_sddmm_multicore_naive_no_b_read"},
    {bsr_sddmm_multicore_CDA_no_b_read<false, true>, "bsr_sddmm_multicore_CDA_no_b_read"},
    // no_c_read
    {bsr_sddmm_multicore_naive_no_c_read<false, true>, "bsr_sddmm_multicore_naive_no_c_read"},
    {bsr_sddmm_multicore_CDA_no_c_read<false, true>, "bsr_sddmm_multicore_CDA_no_c_read"},
    // no_d_read
    {bsr_sddmm_multicore_naive_no_d_read<false, true>, "bsr_sddmm_multicore_naive_no_d_read"},
    {bsr_sddmm_multicore_CDA_no_d_read<false, true>, "bsr_sddmm_multicore_CDA_no_d_read"},
    // no_compute
    {bsr_sddmm_multicore_naive_no_compute<false, true>, "bsr_sddmm_multicore_naive_no_compute"},
    {bsr_sddmm_multicore_CDA_no_compute<false, true>, "bsr_sddmm_multicore_CDA_no_compute"},
    // no_write
    {bsr_sddmm_multicore_naive_no_write<false, true>, "bsr_sddmm_multicore_naive_no_write"},
    {bsr_sddmm_multicore_CDA_no_write<false, true>, "bsr_sddmm_multicore_CDA_no_write"},
};

} // namespace bsr_sddmm_host_code
