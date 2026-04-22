#pragma once

#include <cstdint>
#include <cmath>
#include <random>
#include "include_me.hpp"
#include "sparse_common/bsr_matrix.hpp"
#include "tt-metalium/bfloat16.hpp"

using namespace tt;

namespace sddmm_profiling_suite {

    using ProfileCaseReturnType = std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, dense_matrix<bfloat16>, std::string>;

    // Fully parametric SDDMM cases for sweeps.
    // Template params: M, N, K, R, C, DensityPercent (for the mask)
    // C and D are always dense.
    template <uint32_t M, uint32_t N, uint32_t K, uint32_t R, uint32_t Cb, uint32_t Density>
    ProfileCaseReturnType profile_case_parametric_random() {
        uint32_t blocked_h = M / R;
        uint32_t blocked_w = N / Cb;
        uint32_t nblocks = std::max(1u, (uint32_t)std::round(blocked_h * blocked_w * Density / 100.0));

        bsr_matrix<bfloat16> mask(M, N, R, Cb, nblocks, RAND);
        dense_matrix<bfloat16> c(M, K, RAND);
        dense_matrix<bfloat16> d(K, N, RAND);

        std::string name = "parametric_M" + std::to_string(M) + "_N" + std::to_string(N) +
                           "_K" + std::to_string(K) + "_R" + std::to_string(R) +
                           "_C" + std::to_string(Cb) + "_d" + std::to_string(Density);
        return {mask, c, d, name};
    }

    using ProfileCaseFunctionPtr = ProfileCaseReturnType (*)();

    // Minimal initial registry
    static ProfileCaseFunctionPtr ProfileCaseRegistry[] = {
        profile_case_parametric_random<1024, 1024, 512, 32, 32, 25>,
        profile_case_parametric_random<2048, 2048, 512, 64, 64, 25>,
        profile_case_parametric_random<1024, 1024, 256, 32, 32, 10>,
        profile_case_parametric_random<1024, 1024, 256, 32, 32, 50>,
    };

    // Registry 1: Sweep N (dense output width) — holds M=8192,K=8192,R=C=64,density=25%
    static ProfileCaseFunctionPtr ProfileSweepNRegistry[] = {
        profile_case_parametric_random<8192, 512,  8192, 256, 256, 25>,  // N= 512
        profile_case_parametric_random<8192, 1024, 8192, 256, 256, 25>,  // N=1024
        profile_case_parametric_random<8192, 2048, 8192, 256, 256, 25>,  // N=2048
        profile_case_parametric_random<8192, 4096, 8192, 256, 256, 25>,  // N=4096
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 25>,  // N=8192
    };

    // Registry 2: Sweep density — holds M=N=K=8192,R=C=64, vary density
    static ProfileCaseFunctionPtr ProfileSweepDensityRegistry[] = {
        profile_case_parametric_random<8192, 8192, 8192, 256, 256,  5>,  //  5%
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 10>,  // 10%
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 25>,  // 25%
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 50>,  // 50%
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 75>,  // 75%
    };

    // Registry 3: Sweep K (reduction dimension) — holds M=N=8192,R=C=64,density=25%
    static ProfileCaseFunctionPtr ProfileSweepKRegistry[] = {
        profile_case_parametric_random<8192, 8192,  512, 256, 256, 25>,  // K= 512
        profile_case_parametric_random<8192, 8192, 1024, 256, 256, 25>,  // K=1024
        profile_case_parametric_random<8192, 8192, 2048, 256, 256, 25>,  // K=2048
        profile_case_parametric_random<8192, 8192, 4096, 256, 256, 25>,  // K=4096
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 25>,  // K=8192
    };

    // Registry 4: Sweep block size — holds M=N=K=8192,density=25%
    static ProfileCaseFunctionPtr ProfileSweepBlockSizeRegistry[] = {
        profile_case_parametric_random<8192, 8192, 8192,  32,  32, 25>,  // R=C= 32
        profile_case_parametric_random<8192, 8192, 8192,  64,  64, 25>,  // R=C= 64
        profile_case_parametric_random<8192, 8192, 8192, 128, 128, 25>,  // R=C=128
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 25>,  // R=C=256
    };

} // namespace sddmm_profiling_suite
