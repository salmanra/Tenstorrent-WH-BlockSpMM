// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tiny_bsr_spmm/sparse_8x16_matmul_program_factory.hpp"

#include <filesystem>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <cstdint>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

Sparse8x16MatmulProgramFactory::cached_program_t Sparse8x16MatmulProgramFactory::create(
    const Sparse8x16MatmulParams& operation_attributes, const Sparse8x16MatmulInputs& tensor_args, Tensor& output) {
    const auto& a = tensor_args.a;
    const auto& b = tensor_args.b;
    const auto kt_dim = operation_attributes.kt_dim;

    auto program = tt::tt_metal::CreateProgram();

    const auto core = CoreCoord(0, 0);
    const CoreRangeSet cores(CoreRange(core, core));

    const auto a_shape = a.logical_shape();
    const auto b_shape = b.logical_shape();

    const std::uint32_t M = a_shape[-2];
    const std::uint32_t K = a_shape[-1];
    const std::uint32_t N = b_shape[-1];

    TT_FATAL(M == 8, "sparse_8x16_matmul: M must be 8");
    TT_FATAL(K == kt_dim * 16, "sparse_8x16_matmul: K must be kt_dim*16, got {} vs {}", K, kt_dim * 16);
    TT_FATAL(N == 16, "sparse_8x16_matmul: N must be 16");
    TT_FATAL(kt_dim >= 1, "sparse_8x16_matmul: kt_dim must be >= 1");

    constexpr std::uint32_t cb_id_a = tt::CBIndex::c_0;
    constexpr std::uint32_t cb_id_b = tt::CBIndex::c_1;
    constexpr std::uint32_t cb_id_out = tt::CBIndex::c_2;

    constexpr std::uint32_t tile_size_a = 256;    // 8 * 16 * 2 bytes
    constexpr std::uint32_t tile_size_b = 512;    // 16 * 16 * 2 bytes
    constexpr std::uint32_t tile_size_out = 256;  // 8 * 16 * 2 bytes

    const auto a_buffer = a.buffer();
    const auto b_buffer = b.buffer();
    const auto out_buffer = output.buffer();

    const std::uint32_t a_num_tiles = kt_dim;
    const std::uint32_t b_num_tiles = kt_dim;
    const std::uint32_t out_num_tiles = 1;

    // Circular buffers.
    const auto a_cb_config = CircularBufferConfig(a_num_tiles * tile_size_a, {{cb_id_a, tt::DataFormat::Float16_b}})
                                 .set_page_size(cb_id_a, tile_size_a);
    const auto b_cb_config = CircularBufferConfig(b_num_tiles * tile_size_b, {{cb_id_b, tt::DataFormat::Float16_b}})
                                 .set_page_size(cb_id_b, tile_size_b);
    const auto out_cb_config =
        CircularBufferConfig(out_num_tiles * tile_size_out, {{cb_id_out, tt::DataFormat::Float16_b}})
            .set_page_size(cb_id_out, tile_size_out);

    CreateCircularBuffer(program, cores, a_cb_config);
    CreateCircularBuffer(program, cores, b_cb_config);
    CreateCircularBuffer(program, cores, out_cb_config);

    const bool a_is_dram = a_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const bool b_is_dram = b_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const bool out_is_dram = out_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    const std::vector<std::uint32_t> reader_compile_args = {a_is_dram ? 1U : 0U, b_is_dram ? 1U : 0U, kt_dim};

    const std::string kernel_dir = "tt_metal/programming_examples/Tenstorrent-WH-BlockSpMM/tiny_bsr_spmm/kernels/sparse_8x16_matmul/";

    const auto reader_id = CreateKernel(
        program,
        kernel_dir + "dataflow/reader_sparse_8x16_matmul.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_args});

    const auto writer_id = CreateKernel(
        program,
        kernel_dir + "dataflow/writer_sparse_8x16_matmul.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {out_is_dram ? 1U : 0U}});

    // Compute kernel runs the 8x16x16 LLK unit. Hardcode HiFi4 for v1.
    const std::vector<std::uint32_t> compute_compile_args = {kt_dim};
    const std::vector<std::filesystem::path> compute_includes = {
        "tt_metal/programming_examples/Tenstorrent-WH-BlockSpMM/llk_blackhole/llk_lib",
        "tt_metal/tt-llk/tt_llk_blackhole/llk_lib",
        "tt_metal/tt-llk/tt_llk_blackhole/common/inc"};

    const auto compute_id = CreateKernel(
        program,
        kernel_dir + "compute/sparse_8x16_matmul.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .compile_args = compute_compile_args,
            .defines = {{"MATH_FIDELITY", "MathFidelity::HiFi4"}},
            .compiler_include_paths = compute_includes});

    const std::vector<std::uint32_t> reader_runtime_args = {
        static_cast<std::uint32_t>(a_buffer->address()), static_cast<std::uint32_t>(b_buffer->address())};
    const std::vector<std::uint32_t> writer_runtime_args = {static_cast<std::uint32_t>(out_buffer->address())};

    SetRuntimeArgs(program, reader_id, core, reader_runtime_args);
    SetRuntimeArgs(program, writer_id, core, writer_runtime_args);

    return {std::move(program), {reader_id, compute_id, writer_id, core}};
}

void Sparse8x16MatmulProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const Sparse8x16MatmulParams& /*operation_attributes*/,
    const Sparse8x16MatmulInputs& tensor_args,
    Tensor& output) {
    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;

    const std::vector<std::uint32_t> reader_runtime_args = {
        static_cast<std::uint32_t>(tensor_args.a.buffer()->address()),
        static_cast<std::uint32_t>(tensor_args.b.buffer()->address())};
    const std::vector<std::uint32_t> writer_runtime_args = {static_cast<std::uint32_t>(output.buffer()->address())};

    SetRuntimeArgs(program, shared.reader_id, shared.core, reader_runtime_args);
    SetRuntimeArgs(program, shared.writer_id, shared.core, writer_runtime_args);
}

}  // namespace ttnn::experimental::prim
