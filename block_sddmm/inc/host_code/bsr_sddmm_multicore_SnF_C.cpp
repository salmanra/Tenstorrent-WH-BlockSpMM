#include "../host_code.hpp"
#include "sddmm_zone_config.hpp"

namespace bsr_sddmm_host_code {

template<bool verbose, bool is_profiling>
void bsr_sddmm_multicore_naive_impl(
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
    IDevice* device,
    const std::map<std::string, std::string>& extra_defines = {}) {

    // ── Setup ────────────────────────────────────────────────────────────
    CommandQueue& cq = device->command_queue();
    Program program{};

    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    uint32_t single_tile_size = detail::TileSize(cb_data_format);

    // ── Tiling ───────────────────────────────────────────────────────────
    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t Nt = N / TILE_WIDTH;
    uint32_t Rt = R / TILE_HEIGHT;
    uint32_t Ct = C_block / TILE_WIDTH;

    uint32_t nnz_blocks = sampling_mask.nblocks;

    if constexpr (verbose) {
        log_info(tt::LogVerif, "SDDMM SnF C: M={} N={} K={} R={} C={}", M, N, K, R, C_block);
        log_info(tt::LogVerif, "  Mt={} Nt={} Kt={} Rt={} Ct={} nnz_blocks={}", Mt, Nt, Kt, Rt, Ct, nnz_blocks);
        log_info(tt::LogVerif, "  C: {}x{}, D: {}x{}", c.H, c.W, d.H, d.W);
    }

    // ── Block K selection ────────────────────────────────────────────────
    // Choose block_k: start with Ct, must divide Kt
    uint32_t block_k = Ct;
    while (Kt % block_k != 0 && block_k > 1) block_k--;
    uint32_t num_blocks_k = Kt / block_k;

    if constexpr (verbose) {
        log_info(tt::LogVerif, "  block_k={} num_blocks_k={}", block_k, num_blocks_k);
    }

    // ── Core grid ────────────────────────────────────────────────────────
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    uint32_t num_cores_used = std::min(nnz_blocks, num_cores_total);

    CoreRangeSet all_cores(
        tt::tt_metal::num_cores_to_corerangeset(num_cores_used, compute_with_storage_grid_size, true));

    if constexpr (verbose) {
        log_info(tt::LogVerif, "  num_cores_used={} (of {} total)", num_cores_used, num_cores_total);
    }

    // ── Subblock selection ───────────────────────────────────────────────
    uint32_t out_subblock_h = 0, out_subblock_w = 0;
    for (auto& subblock_hw : bmm_op_utils::SUBBLOCK_HW_CHOICES) {
        out_subblock_h = std::get<0>(subblock_hw);
        out_subblock_w = std::get<1>(subblock_hw);
        if (Rt % out_subblock_h == 0 && Ct % out_subblock_w == 0) {
            break;
        }
    }

    // ── Derived block sizes ──────────────────────────────────────────────
    uint32_t in0_num_subblocks = Rt / out_subblock_h;
    uint32_t in0_block_num_tiles = Rt * block_k;
    uint32_t in0_subblock_num_tiles = out_subblock_h * block_k;

    uint32_t in1_num_subblocks = Ct / out_subblock_w;
    uint32_t in1_block_num_tiles = Ct * block_k;
    uint32_t in1_per_core_w = Ct;

    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;
    uint32_t out_block_num_tiles = Rt * Ct;
    uint32_t sparse_block_num_tiles = Rt * Ct;

    if constexpr (verbose) {
        log_info(tt::LogVerif, "  out_subblock_h={} out_subblock_w={}", out_subblock_h, out_subblock_w);
        log_info(tt::LogVerif, "  in0_block_num_tiles={} in1_block_num_tiles={} out_block_num_tiles={}",
                 in0_block_num_tiles, in1_block_num_tiles, out_block_num_tiles);
    }

    // ── CB Sizing ────────────────────────────────────────────────────────
    // c_0: sparse mask block (single buffer)
    uint32_t sparse_CB_size = sparse_block_num_tiles * single_tile_size;
    // c_1: dense C reduction block (double buffer)
    uint32_t dense_c_CB_size = Rt * block_k * 2 * single_tile_size;
    // c_2: dense D reduction block (double buffer)
    uint32_t dense_d_CB_size = block_k * Ct * 2 * single_tile_size;
    // c_16: output (separate from c_24 because Hadamard multiply needs both simultaneously)
    uint32_t out_CB_size = out_block_num_tiles * single_tile_size;
    // c_24: intermediate (for matmul spill AND Hadamard intermediate)
    uint32_t intermed_CB_size = out_block_num_tiles * single_tile_size;

    // ── Create Circular Buffers ──────────────────────────────────────────
    // c_0: sparse mask
    CircularBufferConfig cb_sparse_config = CircularBufferConfig(sparse_CB_size, {{CBIndex::c_0, cb_data_format}})
        .set_page_size(CBIndex::c_0, single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_sparse_config);

    // c_1: dense C
    CircularBufferConfig cb_c_config = CircularBufferConfig(dense_c_CB_size, {{CBIndex::c_1, cb_data_format}})
        .set_page_size(CBIndex::c_1, single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_c_config);

    // c_2: dense D
    CircularBufferConfig cb_d_config = CircularBufferConfig(dense_d_CB_size, {{CBIndex::c_2, cb_data_format}})
        .set_page_size(CBIndex::c_2, single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_d_config);

    // c_16: output
    CircularBufferConfig cb_out_config = CircularBufferConfig(out_CB_size, {{CBIndex::c_16, cb_data_format}})
        .set_page_size(CBIndex::c_16, single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_out_config);

    // c_24: intermediate (separate from c_16 for Hadamard multiply)
    CircularBufferConfig cb_intermed_config = CircularBufferConfig(intermed_CB_size, {{CBIndex::c_24, cb_data_format}})
        .set_page_size(CBIndex::c_24, single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_intermed_config);

    // ── DRAM Buffer Allocation ───────────────────────────────────────────
    uint32_t sparse_dram_size = single_tile_size * sparse_block_num_tiles * nnz_blocks;
    auto sparse_dram_buffer = MakeBuffer(device, sparse_dram_size, single_tile_size);

    uint32_t dense_c_dram_size = single_tile_size * Mt * Kt;
    auto dense_c_dram_buffer = MakeBuffer(device, dense_c_dram_size, single_tile_size);

    uint32_t dense_d_dram_size = single_tile_size * Kt * Nt;
    auto dense_d_dram_buffer = MakeBuffer(device, dense_d_dram_size, single_tile_size);

    uint32_t out_dram_size = single_tile_size * sparse_block_num_tiles * nnz_blocks;
    auto out_dram_buffer = MakeBuffer(device, out_dram_size, single_tile_size);

    if constexpr (verbose) {
        log_info(tt::LogVerif, " -- DRAM sizes: sparse={} dense_c={} dense_d={} out={}",
                 sparse_dram_size, dense_c_dram_size, dense_d_dram_size, out_dram_size);
    }

    // ── Build block position list ────────────────────────────────────────
    struct BlockPos {
        uint32_t row_i;
        uint32_t col_j;
        uint32_t data_idx;
    };
    std::vector<BlockPos> block_positions;
    block_positions.reserve(nnz_blocks);
    for (uint32_t i = 0; i < sampling_mask.indptr.size() - 1; i++) {
        for (int blk_idx = sampling_mask.indptr[i]; blk_idx < sampling_mask.indptr[i + 1]; blk_idx++) {
            block_positions.push_back({i, (uint32_t)sampling_mask.indices[blk_idx], (uint32_t)blk_idx});
        }
    }
    TT_ASSERT(block_positions.size() == nnz_blocks);

    // ── Compile-time args ────────────────────────────────────────────────
    bool sparse_is_dram = sparse_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    bool dense_c_is_dram = dense_c_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    bool dense_d_is_dram = dense_d_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    bool out_is_dram = out_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM;

    // TODO: add args for SnF
    std::vector<uint32_t> bc_reader_ct_args = {
        (uint32_t)sparse_is_dram,                       // [0]
        (uint32_t)dense_c_is_dram,                      // [1]
        (uint32_t)sparse_dram_buffer->address(),        // [2]
        sparse_block_num_tiles,                          // [3]  Rt*Ct
        (uint32_t)dense_c_dram_buffer->address(),       // [4]
        1u,                                              // [5]  dense_c_stride_w
        Kt,                                              // [6]  dense_c_stride_h
        Rt,                                              // [7]  dense_c_block_h
        block_k,                                         // [8]  dense_c_block_w
        Rt * block_k,                                    // [9]  dense_c_block_num_tiles
        num_blocks_k,                                    // [10]
    };

    std::vector<uint32_t> d_writer_ct_args = {
        (uint32_t)dense_d_is_dram,                      // [0]
        (uint32_t)out_is_dram,                           // [1]
        (uint32_t)dense_d_dram_buffer->address(),       // [2]
        1u,                                              // [3]  dense_d_stride_w
        Nt,                                              // [4]  dense_d_stride_h
        block_k,                                         // [5]  dense_d_block_h
        Ct,                                              // [6]  dense_d_block_w
        block_k * Ct,                                    // [7]  dense_d_block_num_tiles
        (uint32_t)out_dram_buffer->address(),           // [8]
        out_block_num_tiles,                             // [9]  Rt*Ct
        out_subblock_w,                                  // [10]
        out_subblock_h,                                  // [11]
        num_blocks_k,                                    // [12]
        Rt,                                              // [13]
        Ct,                                              // [14]
    };

    std::vector<uint32_t> compute_ct_args = {
        block_k,                    // [0]  in0_block_w
        in0_num_subblocks,          // [1]
        in0_block_num_tiles,        // [2]
        in0_subblock_num_tiles,     // [3]
        in1_num_subblocks,          // [4]
        in1_block_num_tiles,        // [5]
        in1_per_core_w,             // [6]
        out_subblock_h,             // [7]
        out_subblock_w,             // [8]
        out_subblock_num_tiles,     // [9]
        num_blocks_k,               // [10] num_blocks
    };

    if constexpr (verbose) {
        auto print_args = [](const std::string& name, const std::vector<uint32_t>& args) {
            std::cout << "==== " << name << " ====" << std::endl;
            for (size_t i = 0; i < args.size(); ++i) {
                std::cout << "  [" << i << "] = " << args[i] << std::endl;
            }
            std::cout << std::endl;
        };
        print_args("bc_reader_ct_args", bc_reader_ct_args);
        print_args("d_writer_ct_args", d_writer_ct_args);
        print_args("compute_ct_args", compute_ct_args);
    }

    // ── Create Kernels ───────────────────────────────────────────────────
    auto zone_defines = sddmm_zone_config::get_zone_defines();
    zone_defines.insert(extra_defines.begin(), extra_defines.end());

    auto bc_reader_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/Tenstorrent-WH-BlockSpMM/block_sddmm/kernels/dataflow/data_movement_sddmm_BC_SnF.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = bc_reader_ct_args,
            .defines = zone_defines});

    auto d_writer_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/Tenstorrent-WH-BlockSpMM/block_sddmm/kernels/dataflow/data_movement_sddmm_D_out.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = d_writer_ct_args,
            .defines = zone_defines});

    auto compute_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/Tenstorrent-WH-BlockSpMM/block_sddmm/kernels/compute/sddmm_block_multiply.cpp",
        all_cores,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .compile_args = compute_ct_args,
            .defines = zone_defines});

    // ── Runtime args per core ────────────────────────────────────────────
    auto core_coords_vec = corerange_to_cores(all_cores);
    uint32_t blocks_per_core_base = nnz_blocks / num_cores_used;
    uint32_t remainder = nnz_blocks % num_cores_used;
    uint32_t block_offset = 0;

    for (uint32_t ci = 0; ci < core_coords_vec.size(); ci++) {
        auto& core = core_coords_vec[ci];
        uint32_t num_blocks_this_core = blocks_per_core_base + (ci < remainder ? 1 : 0);

        // BC reader runtime args: num_output_blocks, then pairs of (block_row_i, blk_data_idx)
        // TODO: add args for SnF
        std::vector<uint32_t> bc_rt_args;
        bc_rt_args.push_back(num_blocks_this_core);
        for (uint32_t b = 0; b < num_blocks_this_core; b++) {
            auto& bp = block_positions[block_offset + b];
            bc_rt_args.push_back(bp.row_i);
            bc_rt_args.push_back(bp.data_idx);
        }

        // D writer runtime args: num_output_blocks, then pairs of (block_col_j, blk_data_idx)
        std::vector<uint32_t> d_rt_args;
        d_rt_args.push_back(num_blocks_this_core);
        for (uint32_t b = 0; b < num_blocks_this_core; b++) {
            auto& bp = block_positions[block_offset + b];
            d_rt_args.push_back(bp.col_j);
            d_rt_args.push_back(bp.data_idx);
        }

        // Compute runtime args: just num_output_blocks
        std::vector<uint32_t> compute_rt_args;
        compute_rt_args.push_back(num_blocks_this_core);

        tt_metal::SetRuntimeArgs(program, bc_reader_id, core, bc_rt_args);
        tt_metal::SetRuntimeArgs(program, d_writer_id, core, d_rt_args);
        tt_metal::SetRuntimeArgs(program, compute_id, core, compute_rt_args);

        if constexpr (verbose) {
            if (ci == 0) {
                log_info(tt::LogVerif, "Core ({},{}) gets {} blocks starting at offset {}",
                         core.x, core.y, num_blocks_this_core, block_offset);
            }
        }

        block_offset += num_blocks_this_core;
    }

    // ── Data transfer to device ──────────────────────────────────────────
    if constexpr (verbose)
        log_info(tt::LogVerif, " -- H2D transfers --");
    EnqueueWriteBuffer(cq, sparse_dram_buffer, sampling_mask.data.data(), false);
    EnqueueWriteBuffer(cq, dense_c_dram_buffer, c.data.data(), false);
    EnqueueWriteBuffer(cq, dense_d_dram_buffer, d.data.data(), false);

    // ── Execute ──────────────────────────────────────────────────────────
    if constexpr (is_profiling) {
        int num_iters = 10;
        EnqueueProgram(cq, program, true);
        ZoneScopedNC("Device program Loop", tracy::Color::Aquamarine);
        for (int i = 0; i < num_iters; i++) {
            EnqueueProgram(cq, program, true);
        }
    } else {
        if constexpr (verbose)
            log_info(tt::LogVerif, " -- Enqueueing program --");
        EnqueueProgram(cq, program, false);
    }

    // ── Read output ──────────────────────────────────────────────────────
    // Construct output BSR with same structure as mask
    std::vector<bfloat16> out_data(nnz_blocks * R * C_block, bfloat16(0.0f));
    std::vector<int> out_indptr(sampling_mask.indptr);
    std::vector<int> out_indices(sampling_mask.indices);

    EnqueueReadBuffer(cq, out_dram_buffer, out_data.data(), true);
    Finish(cq);

    output = bsr_matrix<bfloat16>(
        std::move(out_data),
        std::move(out_indptr),
        std::move(out_indices),
        M, N, R, C_block, nnz_blocks);

    if constexpr (verbose) {
        log_info(tt::LogVerif, " -- SDDMM complete: output {}x{} with {} blocks --",
                 output.H, output.W, output.nblocks);
    }
}

// Public wrapper
template<bool verbose, bool is_profiling>
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
    IDevice* device) {
    bsr_sddmm_multicore_naive_impl<verbose, is_profiling>(
        sampling_mask, c, d, output, M, N, K, R, C_block, B, device, {});
}

// Explicit template instantiations
template void bsr_sddmm_multicore_naive<false, false>(
    bsr_matrix<bfloat16>&, dense_matrix<bfloat16>&, dense_matrix<bfloat16>&,
    bsr_matrix<bfloat16>&,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, IDevice*);
template void bsr_sddmm_multicore_naive<true, false>(
    bsr_matrix<bfloat16>&, dense_matrix<bfloat16>&, dense_matrix<bfloat16>&,
    bsr_matrix<bfloat16>&,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, IDevice*);
template void bsr_sddmm_multicore_naive<false, true>(
    bsr_matrix<bfloat16>&, dense_matrix<bfloat16>&, dense_matrix<bfloat16>&,
    bsr_matrix<bfloat16>&,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, IDevice*);
template void bsr_sddmm_multicore_naive<true, true>(
    bsr_matrix<bfloat16>&, dense_matrix<bfloat16>&, dense_matrix<bfloat16>&,
    bsr_matrix<bfloat16>&,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, IDevice*);

} // namespace bsr_sddmm_host_code
