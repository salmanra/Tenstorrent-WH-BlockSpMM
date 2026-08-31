// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct TinyBsrSpmmCdaParams {
    std::uint32_t M = 8;
    std::uint32_t K = 16;
    std::uint32_t N = 16;
    std::uint64_t nnz = 0;
    std::uint32_t max_kt_dim = 1;
    std::optional<tt::tt_metal::MemoryConfig> output_mem_config;
};

struct TinyBsrSpmmCdaInputs {
    Tensor a_faces;         // [num_tiles * max_kt_dim, 8, 16] TILE bf16
    Tensor unique_b_tiles;  // [num_unique_pairs * num_output_cols, 16, 16] TILE bf16
    Tensor pair_indices;    // [num_tiles * max_kt_dim] uint32 (row-major)
};

}  // namespace ttnn::experimental::prim
