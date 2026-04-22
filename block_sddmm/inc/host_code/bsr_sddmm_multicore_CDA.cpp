#include "../host_code.hpp"
#include "sddmm_zone_config.hpp"

namespace bsr_sddmm_host_code {

// ── Packing algorithm types ──────────────────────────────────────────

constexpr uint32_t SENTINEL = UINT32_MAX;

struct PackedBlock {
    uint32_t row_i;
    uint32_t col_j;
    uint32_t data_idx;
};

struct Chunk {
    uint32_t row_i;
    std::vector<PackedBlock> blocks;
};

struct Group {
    std::vector<PackedBlock> blocks;
    uint32_t size() const { return blocks.size(); }
};

// ── Host code implementation ─────────────────────────────────────────

template<bool verbose, bool is_profiling>
void bsr_sddmm_multicore_CDA_impl(
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

    // ── Setup ────────────────────────────────────────────────────────
    CommandQueue& cq = device->command_queue();
    Program program{};

    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    uint32_t single_tile_size = detail::TileSize(cb_data_format);

    // ── Tiling ───────────────────────────────────────────────────────
    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t Nt = N / TILE_WIDTH;
    uint32_t Rt = R / TILE_HEIGHT;
    uint32_t Ct = C_block / TILE_WIDTH;

    uint32_t nnz_blocks = sampling_mask.nblocks;

    if constexpr (verbose) {
        log_info(tt::LogVerif, "SDDMM CDA: M={} N={} K={} R={} C={}", M, N, K, R, C_block);
        log_info(tt::LogVerif, "  Mt={} Nt={} Kt={} Rt={} Ct={} nnz_blocks={}", Mt, Nt, Kt, Rt, Ct, nnz_blocks);
    }

    // ── Block K selection ────────────────────────────────────────────
    uint32_t block_k = Ct;
    while (Kt % block_k != 0 && block_k > 1) block_k--;
    uint32_t num_blocks_k = Kt / block_k;

    if constexpr (verbose) {
        log_info(tt::LogVerif, "  block_k={} num_blocks_k={}", block_k, num_blocks_k);
    }

    // ── Core grid ────────────────────────────────────────────────────
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // ── Build block position list ────────────────────────────────────
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

    // ══════════════════════════════════════════════════════════════════
    // ══ Packing algorithm: compact, cut, pack ════════════════════════
    // ══════════════════════════════════════════════════════════════════

    // Step 1: Group blocks by block_row_i
    std::map<uint32_t, std::vector<PackedBlock>> row_groups;
    for (auto& bp : block_positions) {
        row_groups[bp.row_i].push_back({bp.row_i, bp.col_j, bp.data_idx});
    }

    // Step 2: Cut rows with > num_cores_x blocks into chunks
    std::vector<Chunk> chunks;
    for (auto& [row_i, blocks] : row_groups) {
        for (size_t i = 0; i < blocks.size(); i += num_cores_x) {
            Chunk chunk;
            chunk.row_i = row_i;
            size_t end = std::min(i + (size_t)num_cores_x, blocks.size());
            for (size_t j = i; j < end; j++) {
                chunk.blocks.push_back(blocks[j]);
            }
            chunks.push_back(std::move(chunk));
        }
    }

    // Step 3: First-fit decreasing bin packing into groups of <= num_cores_x
    std::sort(chunks.begin(), chunks.end(), [](const Chunk& a, const Chunk& b) {
        return a.blocks.size() > b.blocks.size();
    });

    std::vector<Group> groups;
    for (auto& chunk : chunks) {
        bool placed = false;
        for (auto& group : groups) {
            if (group.size() + chunk.blocks.size() <= num_cores_x) {
                for (auto& bp : chunk.blocks) {
                    group.blocks.push_back(bp);
                }
                placed = true;
                break;
            }
        }
        if (!placed) {
            Group new_group;
            new_group.blocks = chunk.blocks;
            groups.push_back(std::move(new_group));
        }
    }

    uint32_t num_groups = groups.size();
    if (num_groups == 0) num_groups = 1;  // edge case: no blocks

    // Step 4: Assign groups to core rows (round-robin)
    uint32_t num_cores_y_used = std::min(num_groups, num_cores_y);
    if (num_cores_y_used == 0) num_cores_y_used = 1;
    uint32_t num_cores_x_used = num_cores_x;

    std::vector<std::vector<uint32_t>> core_row_schedule(num_cores_y_used);
    for (uint32_t g = 0; g < groups.size(); g++) {
        uint32_t cy = g % num_cores_y_used;
        core_row_schedule[cy].push_back(g);
    }

    // Step 5: Build the 2D grid with sentinel padding
    // global_max_output_blocks: max output slots across ALL core rows
    uint32_t global_max_output_blocks = 0;
    for (uint32_t cy = 0; cy < num_cores_y_used; cy++) {
        uint32_t slots = core_row_schedule[cy].size();
        if (slots > global_max_output_blocks) global_max_output_blocks = slots;
    }
    if (global_max_output_blocks == 0) global_max_output_blocks = 1;

    // grid[cy][cx][slot] = {row_i, col_j, data_idx} or SENTINEL
    std::vector<std::vector<std::vector<PackedBlock>>> grid(num_cores_y_used,
        std::vector<std::vector<PackedBlock>>(num_cores_x_used,
            std::vector<PackedBlock>(global_max_output_blocks, {SENTINEL, SENTINEL, SENTINEL})));

    for (uint32_t cy = 0; cy < num_cores_y_used; cy++) {
        for (uint32_t s_local = 0; s_local < core_row_schedule[cy].size(); s_local++) {
            uint32_t g_idx = core_row_schedule[cy][s_local];
            auto& group = groups[g_idx];
            for (uint32_t x = 0; x < group.size(); x++) {
                grid[cy][x][s_local] = group.blocks[x];
            }
        }
    }

    // Count real output blocks per core (for compute kernel)
    auto count_real_blocks = [&](uint32_t cy, uint32_t cx) -> uint32_t {
        uint32_t count = 0;
        for (uint32_t s = 0; s < global_max_output_blocks; s++) {
            if (grid[cy][cx][s].row_i != SENTINEL) count++;
        }
        return count;
    };

    if constexpr (verbose) {
        log_info(tt::LogVerif, "  Packing: {} groups -> {}x{} core grid, {} max slots",
                 (uint32_t)groups.size(), num_cores_x_used, num_cores_y_used, global_max_output_blocks);
        for (uint32_t cy = 0; cy < num_cores_y_used; cy++) {
            for (uint32_t cx = 0; cx < num_cores_x_used; cx++) {
                uint32_t n = count_real_blocks(cy, cx);
                if (n > 0) {
                    log_info(tt::LogVerif, "    Core ({},{}) -> {} real blocks", cx, cy, n);
                }
            }
        }
    }

    // ── Subblock selection ───────────────────────────────────────────
    uint32_t out_subblock_h = 0, out_subblock_w = 0;
    for (auto& subblock_hw : bmm_op_utils::SUBBLOCK_HW_CHOICES) {
        out_subblock_h = std::get<0>(subblock_hw);
        out_subblock_w = std::get<1>(subblock_hw);
        if (Rt % out_subblock_h == 0 && Ct % out_subblock_w == 0) {
            break;
        }
    }

    // ── Derived block sizes ──────────────────────────────────────────
    uint32_t in0_num_subblocks = Rt / out_subblock_h;
    uint32_t in0_block_num_tiles = Rt * block_k;
    uint32_t in0_subblock_num_tiles = out_subblock_h * block_k;

    uint32_t in1_num_subblocks = Ct / out_subblock_w;
    uint32_t in1_block_num_tiles = Ct * block_k;
    uint32_t in1_per_core_w = Ct;

    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;
    uint32_t out_block_num_tiles = Rt * Ct;
    uint32_t sparse_block_num_tiles = Rt * Ct;

    // ── CB Sizing ────────────────────────────────────────────────────
    uint32_t sparse_CB_size = sparse_block_num_tiles * single_tile_size;
    uint32_t dense_c_CB_size = Rt * block_k * 2 * single_tile_size;       // double buffer
    uint32_t dense_d_CB_size = block_k * Ct * 2 * single_tile_size;       // double buffer
    uint32_t out_CB_size = out_block_num_tiles * single_tile_size;
    uint32_t intermed_CB_size = out_block_num_tiles * single_tile_size;

    // ── Create CoreRangeSet ──────────────────────────────────────────
    CoreRangeSet all_cores(CoreRange({0, 0}, {num_cores_x_used - 1, num_cores_y_used - 1}));

    // ── Create Circular Buffers ──────────────────────────────────────
    CircularBufferConfig cb_sparse_config = CircularBufferConfig(sparse_CB_size, {{CBIndex::c_0, cb_data_format}})
        .set_page_size(CBIndex::c_0, single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_sparse_config);

    CircularBufferConfig cb_c_config = CircularBufferConfig(dense_c_CB_size, {{CBIndex::c_1, cb_data_format}})
        .set_page_size(CBIndex::c_1, single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_c_config);

    CircularBufferConfig cb_d_config = CircularBufferConfig(dense_d_CB_size, {{CBIndex::c_2, cb_data_format}})
        .set_page_size(CBIndex::c_2, single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_d_config);

    CircularBufferConfig cb_out_config = CircularBufferConfig(out_CB_size, {{CBIndex::c_16, cb_data_format}})
        .set_page_size(CBIndex::c_16, single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_out_config);

    CircularBufferConfig cb_intermed_config = CircularBufferConfig(intermed_CB_size, {{CBIndex::c_24, cb_data_format}})
        .set_page_size(CBIndex::c_24, single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_intermed_config);

    // ── DRAM Buffer Allocation ───────────────────────────────────────
    uint32_t sparse_dram_size = single_tile_size * sparse_block_num_tiles * nnz_blocks;
    auto sparse_dram_buffer = MakeBuffer(device, sparse_dram_size, single_tile_size);

    uint32_t dense_c_dram_size = single_tile_size * Mt * Kt;
    auto dense_c_dram_buffer = MakeBuffer(device, dense_c_dram_size, single_tile_size);

    uint32_t dense_d_dram_size = single_tile_size * Kt * Nt;
    auto dense_d_dram_buffer = MakeBuffer(device, dense_d_dram_size, single_tile_size);

    uint32_t out_dram_size = single_tile_size * sparse_block_num_tiles * nnz_blocks;
    auto out_dram_buffer = MakeBuffer(device, out_dram_size, single_tile_size);

    // ── Semaphore Allocation (8 total: 4 per kernel) ─────────────────
    auto bc_sender_sem_id   = CreateSemaphore(program, all_cores, INVALID);
    auto bc_receiver_sem_id = CreateSemaphore(program, all_cores, INVALID);
    auto bc_barrier_sem_id  = CreateSemaphore(program, all_cores, INVALID);
    auto bc_release_sem_id  = CreateSemaphore(program, all_cores, INVALID);

    auto d_sender_sem_id   = CreateSemaphore(program, all_cores, INVALID);
    auto d_receiver_sem_id = CreateSemaphore(program, all_cores, INVALID);
    auto d_barrier_sem_id  = CreateSemaphore(program, all_cores, INVALID);
    auto d_release_sem_id  = CreateSemaphore(program, all_cores, INVALID);

    // ── Compile-time args ────────────────────────────────────────────
    bool sparse_is_dram = sparse_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    bool dense_c_is_dram = dense_c_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    bool dense_d_is_dram = dense_d_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    bool out_is_dram = out_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM;

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
        bc_sender_sem_id,                                // [11] CDA semaphores
        bc_receiver_sem_id,                              // [12]
        bc_barrier_sem_id,                               // [13]
        bc_release_sem_id,                               // [14]
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
        d_sender_sem_id,                                 // [15] CDA semaphores
        d_receiver_sem_id,                               // [16]
        d_barrier_sem_id,                                // [17]
        d_release_sem_id,                                // [18]
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

    // ── Create Kernels ───────────────────────────────────────────────
    auto zone_defines = sddmm_zone_config::get_zone_defines();
    zone_defines.insert(extra_defines.begin(), extra_defines.end());

    // Yes, transpose the NoC!
    auto bc_reader_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/Tenstorrent-WH-BlockSpMM/block_sddmm/kernels/dataflow/data_movement_sddmm_BC_CDA.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_1_default,
            .compile_args = bc_reader_ct_args,
            .defines = zone_defines});

    auto d_writer_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/Tenstorrent-WH-BlockSpMM/block_sddmm/kernels/dataflow/data_movement_sddmm_D_CDA_out.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_0_default,
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

    // ── Runtime args per core ────────────────────────────────────────
    uint32_t M_slots = global_max_output_blocks;

    for (uint32_t cy = 0; cy < num_cores_y_used; cy++) {
        for (uint32_t cx = 0; cx < num_cores_x_used; cx++) {
            CoreCoord core(cx, cy);
            uint32_t num_real_blocks = count_real_blocks(cy, cx);

            // ── BC reader runtime args ───────────────────────────────
            std::vector<uint32_t> bc_rt_args;
            bc_rt_args.push_back(M_slots);       // max_output_blocks
            bc_rt_args.push_back(cx);             // my_core_idx_x
            bc_rt_args.push_back(num_cores_x_used);  // num_cores_in_row
            {
                auto row_phys = device->worker_core_from_logical_core(CoreCoord(0, cy));
                bc_rt_args.push_back((uint32_t)row_phys.y);  // noc_y_for_row
            }
            // noc_x_table for all cores in this row
            for (uint32_t c = 0; c < num_cores_x_used; c++) {
                auto phys = device->worker_core_from_logical_core(CoreCoord(c, cy));
                bc_rt_args.push_back((uint32_t)phys.x);
            }
            // This core's per-slot data_idx (SENTINEL for no-work)
            for (uint32_t s = 0; s < M_slots; s++) {
                bc_rt_args.push_back(grid[cy][cx][s].data_idx);
            }
            // All cores' block_row_i values [core][slot] (row-major)
            for (uint32_t c = 0; c < num_cores_x_used; c++) {
                for (uint32_t s = 0; s < M_slots; s++) {
                    bc_rt_args.push_back(grid[cy][c][s].row_i);
                }
            }

            // ── D writer runtime args ────────────────────────────────
            std::vector<uint32_t> d_rt_args;
            d_rt_args.push_back(M_slots);        // max_output_blocks
            d_rt_args.push_back(cy);              // my_core_idx_y
            d_rt_args.push_back(num_cores_y_used);   // num_cores_in_column
            {
                auto col_phys = device->worker_core_from_logical_core(CoreCoord(cx, 0));
                d_rt_args.push_back((uint32_t)col_phys.x);  // noc_x_for_column
            }
            // noc_y_table for all cores in this column
            for (uint32_t r = 0; r < num_cores_y_used; r++) {
                auto phys = device->worker_core_from_logical_core(CoreCoord(cx, r));
                d_rt_args.push_back((uint32_t)phys.y);
            }
            // This core's per-slot data_idx (SENTINEL for no-work)
            for (uint32_t s = 0; s < M_slots; s++) {
                d_rt_args.push_back(grid[cy][cx][s].data_idx);
            }
            // All cores' block_col_j values [core][slot] (row-major)
            for (uint32_t r = 0; r < num_cores_y_used; r++) {
                for (uint32_t s = 0; s < M_slots; s++) {
                    d_rt_args.push_back(grid[r][cx][s].col_j);
                }
            }

            // ── Compute runtime args ─────────────────────────────────
            std::vector<uint32_t> compute_rt_args = { num_real_blocks };

            tt_metal::SetRuntimeArgs(program, bc_reader_id, core, bc_rt_args);
            tt_metal::SetRuntimeArgs(program, d_writer_id, core, d_rt_args);
            tt_metal::SetRuntimeArgs(program, compute_id, core, compute_rt_args);

            if constexpr (verbose) {
                if (num_real_blocks > 0 && cx == 0 && cy == 0) {
                    log_info(tt::LogVerif, "Core (0,0): {} real blocks, BC rt_args={}, D rt_args={}",
                             num_real_blocks, bc_rt_args.size(), d_rt_args.size());
                }
            }
        }
    }

    // ── Data transfer to device ──────────────────────────────────────
    if constexpr (verbose)
        log_info(tt::LogVerif, " -- H2D transfers --");
    EnqueueWriteBuffer(cq, sparse_dram_buffer, sampling_mask.data.data(), false);
    EnqueueWriteBuffer(cq, dense_c_dram_buffer, c.data.data(), false);
    EnqueueWriteBuffer(cq, dense_d_dram_buffer, d.data.data(), false);

    // ── Execute ──────────────────────────────────────────────────────
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

    // ── Read output ──────────────────────────────────────────────────
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
        log_info(tt::LogVerif, " -- SDDMM CDA complete: output {}x{} with {} blocks --",
                 output.H, output.W, output.nblocks);
    }
}

// ── Public wrapper ───────────────────────────────────────────────────

template<bool verbose, bool is_profiling>
void bsr_sddmm_multicore_CDA(
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
    bsr_sddmm_multicore_CDA_impl<verbose, is_profiling>(
        sampling_mask, c, d, output, M, N, K, R, C_block, B, device, {});
}

// ── Ablation skip wrappers ────────────────────────────────────────────

#define SDDMM_ABLATION_WRAPPER(func_name, skip_flag) \
template<bool verbose, bool is_profiling> \
void func_name( \
    bsr_matrix<bfloat16>& sampling_mask, \
    dense_matrix<bfloat16>& c, \
    dense_matrix<bfloat16>& d, \
    bsr_matrix<bfloat16>& output, \
    uint32_t M, uint32_t N, uint32_t K, \
    uint32_t R, uint32_t C_block, uint32_t B, \
    IDevice* device) { \
    bsr_sddmm_multicore_CDA_impl<verbose, is_profiling>( \
        sampling_mask, c, d, output, M, N, K, R, C_block, B, device, {{skip_flag, "1"}}); \
}

SDDMM_ABLATION_WRAPPER(bsr_sddmm_multicore_CDA_no_b_read, "SKIP_SPARSE_DRAM_READ")
SDDMM_ABLATION_WRAPPER(bsr_sddmm_multicore_CDA_no_c_read, "SKIP_C_DRAM_READ")
SDDMM_ABLATION_WRAPPER(bsr_sddmm_multicore_CDA_no_d_read, "SKIP_D_DRAM_READ")
SDDMM_ABLATION_WRAPPER(bsr_sddmm_multicore_CDA_no_compute, "SKIP_COMPUTE")
SDDMM_ABLATION_WRAPPER(bsr_sddmm_multicore_CDA_no_write, "SKIP_DRAM_WRITE")

#undef SDDMM_ABLATION_WRAPPER

// Explicit template instantiations
#define INSTANTIATE_SDDMM(func) \
template void func<false, false>( \
    bsr_matrix<bfloat16>&, dense_matrix<bfloat16>&, dense_matrix<bfloat16>&, \
    bsr_matrix<bfloat16>&, \
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, IDevice*); \
template void func<true, false>( \
    bsr_matrix<bfloat16>&, dense_matrix<bfloat16>&, dense_matrix<bfloat16>&, \
    bsr_matrix<bfloat16>&, \
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, IDevice*); \
template void func<false, true>( \
    bsr_matrix<bfloat16>&, dense_matrix<bfloat16>&, dense_matrix<bfloat16>&, \
    bsr_matrix<bfloat16>&, \
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, IDevice*); \
template void func<true, true>( \
    bsr_matrix<bfloat16>&, dense_matrix<bfloat16>&, dense_matrix<bfloat16>&, \
    bsr_matrix<bfloat16>&, \
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, IDevice*);

INSTANTIATE_SDDMM(bsr_sddmm_multicore_CDA)
INSTANTIATE_SDDMM(bsr_sddmm_multicore_CDA_no_b_read)
INSTANTIATE_SDDMM(bsr_sddmm_multicore_CDA_no_c_read)
INSTANTIATE_SDDMM(bsr_sddmm_multicore_CDA_no_d_read)
INSTANTIATE_SDDMM(bsr_sddmm_multicore_CDA_no_compute)
INSTANTIATE_SDDMM(bsr_sddmm_multicore_CDA_no_write)

#undef INSTANTIATE_SDDMM

} // namespace bsr_sddmm_host_code
