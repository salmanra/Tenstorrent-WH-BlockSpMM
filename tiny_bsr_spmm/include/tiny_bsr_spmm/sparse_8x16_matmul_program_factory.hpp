// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tiny_bsr_spmm/sparse_8x16_matmul_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct Sparse8x16MatmulProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_id{};
        tt::tt_metal::KernelHandle compute_id{};
        tt::tt_metal::KernelHandle writer_id{};
        CoreCoord core;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const Sparse8x16MatmulParams& operation_attributes,
        const Sparse8x16MatmulInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const Sparse8x16MatmulParams& operation_attributes,
        const Sparse8x16MatmulInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
