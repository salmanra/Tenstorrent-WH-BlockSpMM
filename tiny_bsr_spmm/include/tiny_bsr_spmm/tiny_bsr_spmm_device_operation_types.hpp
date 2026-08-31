// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct TinyBsrSpmmParams {
    std::uint32_t M = 8;
    std::uint32_t K = 16;
    std::uint32_t N = 16;
    std::uint64_t nnz = 0;
    std::uint32_t max_kt_dim = 1;
    bool use_snf = false;
    std::optional<tt::tt_metal::MemoryConfig> output_mem_config;
};

struct TinyBsrSpmmInputs {
    Tensor a_faces;  // [num_tiles * max_kt_dim, 8, 16] TILE bf16
    Tensor b_tiles;  // [num_tiles * max_kt_dim, 16, 16] TILE bf16
};

}  // namespace ttnn::experimental::prim
