// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "tiny_bsr_spmm/tiny_bsr_spmm_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct TinyBsrSpmmSnfProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle in0_id{};
        tt::tt_metal::KernelHandle in1_id{};
        tt::tt_metal::KernelHandle compute_id{};
        std::uint32_t sem_sender{};
        std::uint32_t sem_receiver{};
        std::uint32_t num_cores_x{};
        std::uint32_t num_cores_y{};
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const TinyBsrSpmmParams& operation_attributes,
        const TinyBsrSpmmInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const TinyBsrSpmmParams& operation_attributes,
        const TinyBsrSpmmInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
