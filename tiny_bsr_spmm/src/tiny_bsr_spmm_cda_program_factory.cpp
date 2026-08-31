// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tiny_bsr_spmm/tiny_bsr_spmm_cda_program_factory.hpp"

#include <filesystem>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <cstdint>
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

namespace {

std::uint32_t count_tiles_for_core(
    std::uint32_t core_x,
    std::uint32_t core_y,
    std::uint32_t num_output_rows,
    std::uint32_t num_output_cols,
    std::uint32_t num_cores_x,
    std::uint32_t num_cores_y) {
    std::uint32_t count = 0;
    for (std::uint32_t rb = core_y; rb < num_output_rows; rb += num_cores_y) {
        for (std::uint32_t nt = core_x; nt < num_output_cols; nt += num_cores_x) {
            ++count;
        }
    }
    return count;
}

std::uint32_t num_iters_y_for_core_row(std::uint32_t core_y, std::uint32_t num_output_rows, std::uint32_t num_cores_y) {
    std::uint32_t count = 0;
    for (std::uint32_t rb = core_y; rb < num_output_rows; rb += num_cores_y) {
        ++count;
    }
    return count;
}

std::vector<std::uint32_t> read_pair_indices_to_host(const Tensor& pair_indices) {
    std::vector<std::uint32_t> host_data;
    tt::tt_metal::detail::ReadFromBuffer(*pair_indices.buffer(), host_data);
    return host_data;
}

}  // namespace

TinyBsrSpmmCdaProgramFactory::cached_program_t TinyBsrSpmmCdaProgramFactory::create(
    const TinyBsrSpmmCdaParams& operation_attributes, const TinyBsrSpmmCdaInputs& tensor_args, Tensor& output) {
    const auto& a_faces = tensor_args.a_faces;
    const auto& unique_b_tiles = tensor_args.unique_b_tiles;
    const auto& pair_indices = tensor_args.pair_indices;
    const auto M = operation_attributes.M;
    const auto N = operation_attributes.N;
    const auto max_kt_dim = operation_attributes.max_kt_dim;

    auto program = tt::tt_metal::CreateProgram();

    const std::uint32_t num_output_rows = M / 8;
    const std::uint32_t num_output_cols = N / 16;

    const auto device = a_faces.device();
    const auto grid_size = device->compute_with_storage_grid_size();
    const std::uint32_t num_cores_x = std::min<std::uint32_t>(num_output_cols, grid_size.x);
    const std::uint32_t num_cores_y = std::min<std::uint32_t>(num_output_rows, grid_size.y);

    const CoreRange all_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});
    const CoreRangeSet cores(all_cores);

    TT_FATAL(M % 8 == 0, "tiny_bsr_spmm_cda: M must be a multiple of 8, got {}", M);
    TT_FATAL(N % 16 == 0, "tiny_bsr_spmm_cda: N must be a multiple of 16, got {}", N);
    TT_FATAL(max_kt_dim >= 1, "tiny_bsr_spmm_cda: max_kt_dim must be >= 1");
    TT_FATAL(
        a_faces.logical_shape() ==
            ttnn::Shape({static_cast<int>(num_output_rows * num_output_cols * max_kt_dim), 8, 16}),
        "tiny_bsr_spmm_cda: a_faces shape mismatch");
    TT_FATAL(
        pair_indices.logical_shape() == ttnn::Shape({static_cast<int>(num_output_rows * num_output_cols * max_kt_dim)}),
        "tiny_bsr_spmm_cda: pair_indices shape mismatch");

    constexpr std::uint32_t cb_id_a = tt::CBIndex::c_0;
    constexpr std::uint32_t cb_id_b = tt::CBIndex::c_1;
    constexpr std::uint32_t cb_id_out = tt::CBIndex::c_2;

    constexpr std::uint32_t tile_size_a = 256;    // 8 * 16 * 2 bytes
    constexpr std::uint32_t tile_size_b = 512;    // 16 * 16 * 2 bytes
    constexpr std::uint32_t tile_size_out = 256;  // 8 * 16 * 2 bytes

    const auto a_faces_buffer = a_faces.buffer();
    const auto unique_b_tiles_buffer = unique_b_tiles.buffer();
    const auto out_buffer = output.buffer();

    const auto a_cb_config = CircularBufferConfig(max_kt_dim * tile_size_a, {{cb_id_a, tt::DataFormat::Float16_b}})
                                 .set_page_size(cb_id_a, tile_size_a);
    const auto b_cb_config = CircularBufferConfig(2 * tile_size_b, {{cb_id_b, tt::DataFormat::Float16_b}})
                                 .set_page_size(cb_id_b, tile_size_b);
    const auto out_cb_config = CircularBufferConfig(1 * tile_size_out, {{cb_id_out, tt::DataFormat::Float16_b}})
                                   .set_page_size(cb_id_out, tile_size_out);

    CreateCircularBuffer(program, cores, a_cb_config);
    CreateCircularBuffer(program, cores, b_cb_config);
    CreateCircularBuffer(program, cores, out_cb_config);

    const bool a_is_dram = a_faces_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const bool b_is_dram = unique_b_tiles_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const bool out_is_dram = out_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    const std::vector<std::uint32_t> in0_compile_args = {
        a_is_dram ? 1U : 0U,
        out_is_dram ? 1U : 0U,
        max_kt_dim,
        num_output_rows,
        num_output_cols,
        num_cores_x,
        num_cores_y};
    const std::vector<std::uint32_t> in1_compile_args = {b_is_dram ? 1U : 0U, max_kt_dim, num_output_cols, num_cores_y};
    const std::vector<std::uint32_t> compute_compile_args = {max_kt_dim};

    const std::string kernel_dir = "tt_metal/programming_examples/Tenstorrent-WH-BlockSpMM/tiny_bsr_spmm/kernels/tiny_bsr_spmm/";

    const auto in0_id = CreateKernel(
        program,
        kernel_dir + "dataflow/writer_tiny_bsr_spmm_snf.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = in0_compile_args});

    const auto in1_id = CreateKernel(
        program,
        kernel_dir + "dataflow/reader_tiny_bsr_spmm_cda.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = in1_compile_args});

    const std::vector<std::filesystem::path> compute_includes = {
        "tt_metal/programming_examples/Tenstorrent-WH-BlockSpMM/llk_blackhole/llk_lib",
        "tt_metal/tt-llk/tt_llk_blackhole/llk_lib",
        "tt_metal/tt-llk/tt_llk_blackhole/common/inc"};

    const auto compute_id = CreateKernel(
        program,
        kernel_dir + "compute/tiny_bsr_spmm_snf.cpp",
        cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .compile_args = compute_compile_args,
            .defines = {{"MATH_FIDELITY", "MathFidelity::HiFi4"}},
            .compiler_include_paths = compute_includes});

    const std::uint32_t in0_sem_sender = CreateSemaphore(program, cores, 0);
    const std::uint32_t in0_sem_receiver = CreateSemaphore(program, cores, 0);
    const std::uint32_t in1_sem_sender = CreateSemaphore(program, cores, 0);
    const std::uint32_t in1_sem_receiver = CreateSemaphore(program, cores, 0);
    const std::uint32_t in1_sem_barrier = CreateSemaphore(program, cores, 0);
    const std::uint32_t in1_sem_release = CreateSemaphore(program, cores, 0);

    const std::vector<std::uint32_t> host_pair_indices = read_pair_indices_to_host(pair_indices);

    for (std::uint32_t cy = 0; cy < num_cores_y; ++cy) {
        for (std::uint32_t cx = 0; cx < num_cores_x; ++cx) {
            CoreCoord core(cx, cy);
            const bool is_injector = cx == 0;
            const bool is_sink = cx == num_cores_x - 1;

            const auto next_core = CoreCoord(std::min(cx + 1, num_cores_x - 1), cy);
            const auto prev_core = CoreCoord(cx == 0 ? 0 : cx - 1, cy);
            const auto next_core_physical = device->worker_core_from_logical_core(next_core);
            const auto prev_core_physical = device->worker_core_from_logical_core(prev_core);

            const std::uint32_t num_tiles_this_core =
                count_tiles_for_core(cx, cy, num_output_rows, num_output_cols, num_cores_x, num_cores_y);
            const std::uint32_t num_iters_x = (num_output_cols + num_cores_x - 1 - cx) / num_cores_x;
            const std::uint32_t num_iters_y = (num_output_rows + num_cores_y - 1 - cy) / num_cores_y;
            const std::uint32_t output_idx_x_start = cx;

            const std::vector<std::uint32_t> in0_runtime_args = {
                static_cast<std::uint32_t>(a_faces_buffer->address()),
                static_cast<std::uint32_t>(out_buffer->address()),
                cx,
                cy,
                in0_sem_sender,
                in0_sem_receiver,
                static_cast<std::uint32_t>(next_core_physical.x),
                static_cast<std::uint32_t>(next_core_physical.y),
                static_cast<std::uint32_t>(prev_core_physical.x),
                static_cast<std::uint32_t>(prev_core_physical.y),
                is_injector ? 1U : 0U,
                is_sink ? 1U : 0U};
            const std::vector<std::uint32_t> compute_runtime_args = {num_tiles_this_core};

            // Column-wide CDA schedule for this core's column (cx).
            std::vector<std::uint32_t> in1_runtime_args;
            in1_runtime_args.reserve(64 + num_cores_y * num_output_rows * max_kt_dim);
            in1_runtime_args.push_back(static_cast<std::uint32_t>(unique_b_tiles_buffer->address()));
            in1_runtime_args.push_back(cx);
            in1_runtime_args.push_back(cy);
            in1_runtime_args.push_back(num_iters_x);
            in1_runtime_args.push_back(num_iters_y);
            in1_runtime_args.push_back(output_idx_x_start);
            in1_runtime_args.push_back(in1_sem_sender);
            in1_runtime_args.push_back(in1_sem_receiver);
            in1_runtime_args.push_back(in1_sem_barrier);
            in1_runtime_args.push_back(in1_sem_release);

            const auto column_phys = device->worker_core_from_logical_core(CoreCoord(cx, 0));
            in1_runtime_args.push_back(static_cast<std::uint32_t>(column_phys.x));
            for (std::uint32_t r = 0; r < num_cores_y; ++r) {
                const auto phys = device->worker_core_from_logical_core(CoreCoord(cx, r));
                in1_runtime_args.push_back(static_cast<std::uint32_t>(phys.y));
            }

            for (std::uint32_t r = 0; r < num_cores_y; ++r) {
                const std::uint32_t iters_y = num_iters_y_for_core_row(r, num_output_rows, num_cores_y);
                in1_runtime_args.push_back(iters_y);
            }

            for (std::uint32_t r = 0; r < num_cores_y; ++r) {
                const std::uint32_t iters_y = num_iters_y_for_core_row(r, num_output_rows, num_cores_y);
                for (std::uint32_t iy = 0; iy < iters_y; ++iy) {
                    const std::uint32_t rb = r + iy * num_cores_y;
                    for (std::uint32_t k = 0; k < max_kt_dim; ++k) {
                        const std::uint32_t tile_idx = rb * num_output_cols;
                        const std::uint32_t idx = tile_idx * max_kt_dim + k;
                        in1_runtime_args.push_back(host_pair_indices[idx]);
                    }
                }
            }

            SetRuntimeArgs(program, in0_id, core, in0_runtime_args);
            SetRuntimeArgs(program, in1_id, core, in1_runtime_args);
            SetRuntimeArgs(program, compute_id, core, compute_runtime_args);
        }
    }

    return {
        std::move(program),
        {in0_id,
         in1_id,
         compute_id,
         in0_sem_sender,
         in0_sem_receiver,
         in1_sem_sender,
         in1_sem_receiver,
         in1_sem_barrier,
         in1_sem_release,
         num_cores_x,
         num_cores_y}};
}

void TinyBsrSpmmCdaProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const TinyBsrSpmmCdaParams& operation_attributes,
    const TinyBsrSpmmCdaInputs& tensor_args,
    Tensor& output) {
    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;
    const auto device = tensor_args.a_faces.device();

    const std::uint32_t num_output_rows = output.logical_shape()[-2] / 8;
    const std::uint32_t num_output_cols = output.logical_shape()[-1] / 16;

    const std::vector<std::uint32_t> host_pair_indices = read_pair_indices_to_host(tensor_args.pair_indices);

    for (std::uint32_t cy = 0; cy < shared.num_cores_y; ++cy) {
        for (std::uint32_t cx = 0; cx < shared.num_cores_x; ++cx) {
            CoreCoord core(cx, cy);
            const bool is_injector = cx == 0;
            const bool is_sink = cx == shared.num_cores_x - 1;

            const auto next_core = CoreCoord(std::min(cx + 1, shared.num_cores_x - 1), cy);
            const auto prev_core = CoreCoord(cx == 0 ? 0 : cx - 1, cy);
            const auto next_core_physical = device->worker_core_from_logical_core(next_core);
            const auto prev_core_physical = device->worker_core_from_logical_core(prev_core);

            const std::uint32_t num_tiles_this_core =
                count_tiles_for_core(cx, cy, num_output_rows, num_output_cols, shared.num_cores_x, shared.num_cores_y);
            const std::uint32_t num_iters_x = (num_output_cols + shared.num_cores_x - 1 - cx) / shared.num_cores_x;
            const std::uint32_t num_iters_y = (num_output_rows + shared.num_cores_y - 1 - cy) / shared.num_cores_y;
            const std::uint32_t output_idx_x_start = cx;

            const std::vector<std::uint32_t> in0_runtime_args = {
                static_cast<std::uint32_t>(tensor_args.a_faces.buffer()->address()),
                static_cast<std::uint32_t>(output.buffer()->address()),
                cx,
                cy,
                shared.in0_sem_sender,
                shared.in0_sem_receiver,
                static_cast<std::uint32_t>(next_core_physical.x),
                static_cast<std::uint32_t>(next_core_physical.y),
                static_cast<std::uint32_t>(prev_core_physical.x),
                static_cast<std::uint32_t>(prev_core_physical.y),
                is_injector ? 1U : 0U,
                is_sink ? 1U : 0U};
            const std::vector<std::uint32_t> compute_runtime_args = {num_tiles_this_core};

            std::vector<std::uint32_t> in1_runtime_args;
            in1_runtime_args.reserve(64 + shared.num_cores_y * num_output_rows * operation_attributes.max_kt_dim);
            in1_runtime_args.push_back(static_cast<std::uint32_t>(tensor_args.unique_b_tiles.buffer()->address()));
            in1_runtime_args.push_back(cx);
            in1_runtime_args.push_back(cy);
            in1_runtime_args.push_back(num_iters_x);
            in1_runtime_args.push_back(num_iters_y);
            in1_runtime_args.push_back(output_idx_x_start);
            in1_runtime_args.push_back(shared.in1_sem_sender);
            in1_runtime_args.push_back(shared.in1_sem_receiver);
            in1_runtime_args.push_back(shared.in1_sem_barrier);
            in1_runtime_args.push_back(shared.in1_sem_release);

            const auto column_phys = device->worker_core_from_logical_core(CoreCoord(cx, 0));
            in1_runtime_args.push_back(static_cast<std::uint32_t>(column_phys.x));
            for (std::uint32_t r = 0; r < shared.num_cores_y; ++r) {
                const auto phys = device->worker_core_from_logical_core(CoreCoord(cx, r));
                in1_runtime_args.push_back(static_cast<std::uint32_t>(phys.y));
            }

            for (std::uint32_t r = 0; r < shared.num_cores_y; ++r) {
                const std::uint32_t iters_y = num_iters_y_for_core_row(r, num_output_rows, shared.num_cores_y);
                in1_runtime_args.push_back(iters_y);
            }

            for (std::uint32_t r = 0; r < shared.num_cores_y; ++r) {
                const std::uint32_t iters_y = num_iters_y_for_core_row(r, num_output_rows, shared.num_cores_y);
                for (std::uint32_t iy = 0; iy < iters_y; ++iy) {
                    const std::uint32_t rb = r + iy * shared.num_cores_y;
                    for (std::uint32_t k = 0; k < operation_attributes.max_kt_dim; ++k) {
                        const std::uint32_t tile_idx = rb * num_output_cols;
                        const std::uint32_t idx = tile_idx * operation_attributes.max_kt_dim + k;
                        in1_runtime_args.push_back(host_pair_indices[idx]);
                    }
                }
            }

            SetRuntimeArgs(program, shared.in0_id, core, in0_runtime_args);
            SetRuntimeArgs(program, shared.in1_id, core, in1_runtime_args);
            SetRuntimeArgs(program, shared.compute_id, core, compute_runtime_args);
        }
    }
}

}  // namespace ttnn::experimental::prim
