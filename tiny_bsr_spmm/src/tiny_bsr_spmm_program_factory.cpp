// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tiny_bsr_spmm/tiny_bsr_spmm_program_factory.hpp"

#include <filesystem>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <cstdint>

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

}  // namespace

TinyBsrSpmmProgramFactory::cached_program_t TinyBsrSpmmProgramFactory::create(
    const TinyBsrSpmmParams& operation_attributes, const TinyBsrSpmmInputs& tensor_args, Tensor& output) {
    const auto& a_faces = tensor_args.a_faces;
    const auto& b_tiles = tensor_args.b_tiles;
    const auto M = operation_attributes.M;
    const auto N = operation_attributes.N;
    const auto max_kt_dim = operation_attributes.max_kt_dim;

    auto program = tt::tt_metal::CreateProgram();

    const std::uint32_t num_output_rows = M / 8;
    const std::uint32_t num_output_cols = N / 16;

    const auto grid_size = a_faces.device()->compute_with_storage_grid_size();
    const std::uint32_t num_cores_x = std::min<std::uint32_t>(num_output_cols, grid_size.x);
    const std::uint32_t num_cores_y = std::min<std::uint32_t>(num_output_rows, grid_size.y);

    const CoreRange all_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});
    const CoreRangeSet cores(all_cores);

    TT_FATAL(M % 8 == 0, "tiny_bsr_spmm: M must be a multiple of 8, got {}", M);
    TT_FATAL(N % 16 == 0, "tiny_bsr_spmm: N must be a multiple of 16, got {}", N);
    TT_FATAL(max_kt_dim >= 1, "tiny_bsr_spmm: max_kt_dim must be >= 1");
    TT_FATAL(
        a_faces.logical_shape() ==
            ttnn::Shape({static_cast<int>(num_output_rows * num_output_cols * max_kt_dim), 8, 16}),
        "tiny_bsr_spmm: a_faces shape mismatch");
    TT_FATAL(
        b_tiles.logical_shape() ==
            ttnn::Shape({static_cast<int>(num_output_rows * num_output_cols * max_kt_dim), 16, 16}),
        "tiny_bsr_spmm: b_tiles shape mismatch");

    constexpr std::uint32_t cb_id_a = tt::CBIndex::c_0;
    constexpr std::uint32_t cb_id_b = tt::CBIndex::c_1;
    constexpr std::uint32_t cb_id_out = tt::CBIndex::c_2;

    constexpr std::uint32_t tile_size_a = 256;    // 8 * 16 * 2 bytes
    constexpr std::uint32_t tile_size_b = 512;    // 16 * 16 * 2 bytes
    constexpr std::uint32_t tile_size_out = 256;  // 8 * 16 * 2 bytes

    const auto a_faces_buffer = a_faces.buffer();
    const auto b_tiles_buffer = b_tiles.buffer();
    const auto out_buffer = output.buffer();

    const auto a_cb_config = CircularBufferConfig(max_kt_dim * tile_size_a, {{cb_id_a, tt::DataFormat::Float16_b}})
                                 .set_page_size(cb_id_a, tile_size_a);
    const auto b_cb_config = CircularBufferConfig(max_kt_dim * tile_size_b, {{cb_id_b, tt::DataFormat::Float16_b}})
                                 .set_page_size(cb_id_b, tile_size_b);
    const auto out_cb_config = CircularBufferConfig(1 * tile_size_out, {{cb_id_out, tt::DataFormat::Float16_b}})
                                   .set_page_size(cb_id_out, tile_size_out);

    CreateCircularBuffer(program, cores, a_cb_config);
    CreateCircularBuffer(program, cores, b_cb_config);
    CreateCircularBuffer(program, cores, out_cb_config);

    const bool a_is_dram = a_faces_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const bool b_is_dram = b_tiles_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const bool out_is_dram = out_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    const std::vector<std::uint32_t> reader_compile_args = {
        a_is_dram ? 1U : 0U,
        b_is_dram ? 1U : 0U,
        max_kt_dim,
        num_output_rows,
        num_output_cols,
        num_cores_x,
        num_cores_y};
    const std::vector<std::uint32_t> writer_compile_args = {
        out_is_dram ? 1U : 0U, num_output_rows, num_output_cols, num_cores_x, num_cores_y};
    const std::vector<std::uint32_t> compute_compile_args = {max_kt_dim};

    const std::string kernel_dir = "tt_metal/programming_examples/Tenstorrent-WH-BlockSpMM/tiny_bsr_spmm/kernels/tiny_bsr_spmm/";

    const auto reader_id = CreateKernel(
        program,
        kernel_dir + "dataflow/reader_tiny_bsr_spmm.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_args});

    const auto writer_id = CreateKernel(
        program,
        kernel_dir + "dataflow/writer_tiny_bsr_spmm.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_args});

    const std::vector<std::filesystem::path> compute_includes = {
        "tt_metal/programming_examples/Tenstorrent-WH-BlockSpMM/llk_blackhole/llk_lib",
        "tt_metal/tt-llk/tt_llk_blackhole/llk_lib",
        "tt_metal/tt-llk/tt_llk_blackhole/common/inc"};

    const auto compute_id = CreateKernel(
        program,
        kernel_dir + "compute/tiny_bsr_spmm.cpp",
        cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .compile_args = compute_compile_args,
            .defines = {{"MATH_FIDELITY", "MathFidelity::HiFi4"}},
            .compiler_include_paths = compute_includes});

    const std::uint32_t sem_writer_reader = CreateSemaphore(program, cores, 1);

    for (std::uint32_t cy = 0; cy < num_cores_y; ++cy) {
        for (std::uint32_t cx = 0; cx < num_cores_x; ++cx) {
            CoreCoord core(cx, cy);
            const std::uint32_t num_tiles_this_core =
                count_tiles_for_core(cx, cy, num_output_rows, num_output_cols, num_cores_x, num_cores_y);

            const std::vector<std::uint32_t> reader_runtime_args = {
                static_cast<std::uint32_t>(a_faces_buffer->address()),
                static_cast<std::uint32_t>(b_tiles_buffer->address()),
                cx,
                cy,
                sem_writer_reader};
            const std::vector<std::uint32_t> writer_runtime_args = {
                static_cast<std::uint32_t>(out_buffer->address()), cx, cy, sem_writer_reader};
            const std::vector<std::uint32_t> compute_runtime_args = {num_tiles_this_core};

            SetRuntimeArgs(program, reader_id, core, reader_runtime_args);
            SetRuntimeArgs(program, writer_id, core, writer_runtime_args);
            SetRuntimeArgs(program, compute_id, core, compute_runtime_args);
        }
    }

    return {std::move(program), {reader_id, compute_id, writer_id, sem_writer_reader, num_cores_x, num_cores_y}};
}

void TinyBsrSpmmProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const TinyBsrSpmmParams& /*operation_attributes*/,
    const TinyBsrSpmmInputs& tensor_args,
    Tensor& output) {
    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;

    const std::uint32_t num_output_rows = output.logical_shape()[-2] / 8;
    const std::uint32_t num_output_cols = output.logical_shape()[-1] / 16;

    for (std::uint32_t cy = 0; cy < shared.num_cores_y; ++cy) {
        for (std::uint32_t cx = 0; cx < shared.num_cores_x; ++cx) {
            CoreCoord core(cx, cy);
            const std::uint32_t num_tiles_this_core =
                count_tiles_for_core(cx, cy, num_output_rows, num_output_cols, shared.num_cores_x, shared.num_cores_y);

            const std::vector<std::uint32_t> reader_runtime_args = {
                static_cast<std::uint32_t>(tensor_args.a_faces.buffer()->address()),
                static_cast<std::uint32_t>(tensor_args.b_tiles.buffer()->address()),
                cx,
                cy,
                shared.sem_writer_reader};
            const std::vector<std::uint32_t> writer_runtime_args = {
                static_cast<std::uint32_t>(output.buffer()->address()), cx, cy, shared.sem_writer_reader};
            const std::vector<std::uint32_t> compute_runtime_args = {num_tiles_this_core};

            SetRuntimeArgs(program, shared.reader_id, core, reader_runtime_args);
            SetRuntimeArgs(program, shared.writer_id, core, writer_runtime_args);
            SetRuntimeArgs(program, shared.compute_id, core, compute_runtime_args);
        }
    }
}

}  // namespace ttnn::experimental::prim
