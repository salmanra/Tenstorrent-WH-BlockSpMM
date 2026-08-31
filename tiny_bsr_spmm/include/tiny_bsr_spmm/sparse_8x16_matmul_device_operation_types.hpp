// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct Sparse8x16MatmulParams {
    std::uint32_t kt_dim = 1;
    std::optional<tt::tt_metal::MemoryConfig> output_mem_config;
};

struct Sparse8x16MatmulInputs {
    Tensor a;  // [8, kt_dim*16] TILE bf16, kt_dim 8x16 tiles
    Tensor b;  // [kt_dim*16, 16] TILE bf16, kt_dim 16x16 tiles
};

}  // namespace ttnn::experimental::prim
