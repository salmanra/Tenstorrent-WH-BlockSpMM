#include "../host_code.hpp"
#include "spmm_zone_config.hpp"

namespace bsr_host_code {

template<bool verbose, bool is_profiling, bool use_optimal_noc = true>
void bsr_spmm_multicore_load_balanced_new_DM_impl(
    bsr_matrix<bfloat16>& a,
    dense_matrix<bfloat16>& b,
    dense_matrix<bfloat16>& output,
    bool bcast_batch,
    uint32_t nnz_blocks,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t R,
    uint32_t C,
    uint32_t B,
    IDevice* device,
    const std::map<std::string, std::string>& extra_defines = {}){

    // TT-Metal CommandQueue and Program setup
    CommandQueue& cq = device->command_queue();
    Program program{};

    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    uint32_t single_tile_size = detail::TileSize(cb_data_format);

    tt::DataFormat indexing_data_format = tt::DataFormat::Int32;
    uint32_t indexing_data_single_tile_size = detail::TileSize(indexing_data_format);
    uint32_t num_tiles_for_col_indices = (indexing_data_single_tile_size - 1 + sizeof(int) * nnz_blocks) / indexing_data_single_tile_size;
    uint32_t num_tiles_for_indptr = (indexing_data_single_tile_size - 1 + sizeof(int) * (M / R + 1)) / indexing_data_single_tile_size;
    uint32_t num_tiles_indexing = num_tiles_for_col_indices + num_tiles_for_indptr;

    // Core Grid detection
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;

    // Per-core tiling and blocking args
    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t Nt = N / TILE_WIDTH;

    uint32_t Rt = R / TILE_HEIGHT;
    uint32_t Ct = C / TILE_WIDTH;

    // Core grid assignment — deque with sentinel so folded_bsr_matrix_indices[nnz_rows] is safe
    std::deque<uint32_t> folded_bsr_matrix_indices;
    uint32_t nnz_rows = 0;
    uint32_t folded_index = 0;
    for (uint32_t i = 0; i < a.indptr.size() - 1; i++) {
        if (a.indptr[i+1] - a.indptr[i] > 0){
            folded_bsr_matrix_indices.push_back(folded_index);
            nnz_rows++;
        }
        folded_index++;
    }
    folded_bsr_matrix_indices.push_back(folded_index); // sentinel

    uint32_t in0_block_h = Rt;
    uint32_t in0_block_w = Ct;
    uint32_t in1_block_w = get_Npc_from_BSR_block_size(Nt, in0_block_h, in0_block_w, num_cores_x, num_cores_y, num_tiles_indexing, nnz_rows);

    TT_ASSERT(Mt % in0_block_h == 0);
    TT_ASSERT(Nt % in1_block_w == 0);
    TT_ASSERT(Kt % in0_block_w == 0);

    uint32_t num_blocks_x = Nt / in1_block_w;
    uint32_t num_blocks_y = nnz_rows;
    uint32_t num_blocks_total = num_blocks_x * num_blocks_y;

    uint32_t num_iters_x = (num_blocks_x + num_cores_x - 1) / num_cores_x;
    uint32_t num_iters_y = (num_blocks_y + num_cores_y - 1) / num_cores_y;

    uint32_t num_work_regions = (num_blocks_total + num_iters_x * num_iters_y - 1) / (num_iters_x * num_iters_y);
    uint32_t target_num_cores;
    if (num_work_regions < num_cores_total)
        target_num_cores = num_work_regions;
    else
        target_num_cores = num_cores_total;

    uint32_t out_subblock_h = 0, out_subblock_w = 0;
    for (auto& subblock_hw : bmm_op_utils::SUBBLOCK_HW_CHOICES) {
        out_subblock_h = std::get<0>(subblock_hw);
        out_subblock_w = std::get<1>(subblock_hw);
        if (in0_block_h % out_subblock_h == 0 and in1_block_w % out_subblock_w == 0) {
            break;
        }
    }

    // Load-balanced core grid: rectangular range sized to the actual work
    CoreCoord start_core = {0, 0};
    CoreCoord core_range(0, 0);
    if ((num_blocks_y / num_iters_y) <= num_cores_y &&
        (num_blocks_x / num_iters_x) <= num_cores_x) {
        core_range.x = (num_blocks_x + num_iters_x - 1) / num_iters_x;
        core_range.y = (num_blocks_y + num_iters_y - 1) / num_iters_y;
    }
    uint32_t num_cores_c = core_range.x;
    uint32_t num_cores_r = core_range.y;

    CoreRange all_cores(
        {(std::size_t)start_core.x, (std::size_t)start_core.y},
        {(std::size_t)start_core.x + num_cores_c - 1, (std::size_t)start_core.y + num_cores_r - 1});

    // Circular Buffer sizing
    uint32_t in0_CB_num_tiles = in0_block_h * in0_block_w * 2; // double buffer
    uint32_t in0_CB_size = in0_CB_num_tiles * single_tile_size;
    uint32_t in1_CB_num_tiles = in0_block_w * in1_block_w * 2; // double buffer
    uint32_t in1_CB_size = in1_CB_num_tiles * single_tile_size;
    uint32_t out_CB_num_tiles = in0_block_h * in1_block_w; // single buffer
    uint32_t out_CB_size = out_CB_num_tiles * single_tile_size;

    uint32_t in0_num_subblocks = (in0_block_h / out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;

    uint32_t in1_num_subblocks = (in1_block_w / out_subblock_w);
    uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks;
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    // DRAM buffer initialisation
    uint32_t dram_buffer_dst_row_size = single_tile_size * Rt * Nt;
    uint32_t dram_buffer_dst_total_size = dram_buffer_dst_row_size * nnz_rows;

    uint32_t dram_buffer_A_size = single_tile_size * Rt * Ct * nnz_blocks;
    uint32_t dram_buffer_B_size = single_tile_size * Nt * Kt;

    uint32_t dram_buffer_col_indices_size = sizeof(int) * a.indices.size();
    dram_buffer_col_indices_size = indexing_data_single_tile_size * ((indexing_data_single_tile_size - 1 + dram_buffer_col_indices_size) / indexing_data_single_tile_size);

    uint32_t dram_buffer_indptr_size = sizeof(int) * a.indptr.size();
    dram_buffer_indptr_size = indexing_data_single_tile_size * ((indexing_data_single_tile_size - 1 + dram_buffer_indptr_size) / indexing_data_single_tile_size);

    auto dst_dram_buffer = MakeBuffer(device, dram_buffer_dst_total_size, single_tile_size);
    auto src0_dram_buffer = MakeBuffer(device, dram_buffer_A_size, single_tile_size);
    auto src1_dram_buffer = MakeBuffer(device, dram_buffer_B_size, single_tile_size);
    auto column_indices_dram_buffer = MakeBuffer(device, dram_buffer_col_indices_size, indexing_data_single_tile_size);
    auto indptr_dram_buffer = MakeBuffer(device, dram_buffer_indptr_size, indexing_data_single_tile_size);

    if constexpr (verbose) {
        log_info(tt::LogVerif, " -- DRAM Buffer Sizings in tiles --");
        log_info(
            tt::LogVerif,
            " -- dst_dram={} -- sparse_matrix_data={} -- dense_matrix={} -- col_indices={} -- indptr={} -- idx_data_single_tile_size={}",
            dram_buffer_dst_total_size / single_tile_size,
            dram_buffer_A_size / single_tile_size,
            dram_buffer_B_size / single_tile_size,
            dram_buffer_col_indices_size / indexing_data_single_tile_size,
            dram_buffer_indptr_size / indexing_data_single_tile_size,
            indexing_data_single_tile_size);
    }

    if constexpr (verbose) {
        log_info(tt::LogVerif, " -- Metalium Block and subblock sizing --");
        log_info(
            tt::LogVerif,
            " -- per_core_M={} -- per_block_M={} -- per_core_N={} -- out_subblock_h={} -- out_subblock_w={} --",
            num_iters_y * in0_block_h,
            in0_block_h,
            num_iters_x * in1_block_w,
            out_subblock_h,
            out_subblock_w);
    }

    if constexpr (verbose) {
        log_info(tt::LogVerif, " -- Core Grid Allocation Information --");
        log_info(
            tt::LogVerif,
            " -- available_cores_y={} -- available_cores_x={} -- num_iters_y={} -- num_iters_x={} -- nnz_rows={} --",
            num_cores_y,
            num_cores_x,
            num_iters_y,
            num_iters_x,
            nnz_rows);
    }

    if constexpr (verbose) {
        log_info(tt::LogVerif, " -- Metalium Core Grid Sizing --");
        log_info(
            tt::LogVerif,
            " -- Mt= {} -- Nt= {} -- num_output_blocks= {} -- cores_used={} -- num_blocks_x={} -- num_blocks_y={} --",
            Mt,
            Nt,
            num_blocks_total,
            all_cores,
            num_blocks_x,
            num_blocks_y);
    }

    /*
    SRAM Circular Buffers
    */
    uint32_t src0_cb_index = CBIndex::c_0;
    CircularBufferConfig cb_src0_config = CircularBufferConfig(in0_CB_size, {{src0_cb_index, cb_data_format}})
                                              .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t src1_cb_index = CBIndex::c_1;
    CircularBufferConfig cb_src1_config = CircularBufferConfig(in1_CB_size, {{src1_cb_index, cb_data_format}})
                                              .set_page_size(src1_cb_index, single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    uint32_t output_cb_index = tt::CBIndex::c_16;
    uint32_t interm0_cb_index = tt::CBIndex::c_24;
    std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
        {output_cb_index, cb_data_format}, {interm0_cb_index, cb_data_format}};
    CircularBufferConfig cb_output_config = CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
        .set_page_size(output_cb_index, single_tile_size)
        .set_page_size(interm0_cb_index, single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    uint32_t column_indices_cb_index = CBIndex::c_2;
    CircularBufferConfig cb_column_indices_config = CircularBufferConfig(
        dram_buffer_col_indices_size, {{column_indices_cb_index, tt::DataFormat::Int32}})
                                                .set_page_size(column_indices_cb_index, indexing_data_single_tile_size);
    auto cb_column_indices = tt_metal::CreateCircularBuffer(program, all_cores, cb_column_indices_config);

    auto indptr_cb_index = CBIndex::c_3;
    auto cb_indptr = MakeCircularBuffer(program, all_cores, indptr_cb_index, dram_buffer_indptr_size, indexing_data_single_tile_size, indexing_data_format);

    // Compile-time arguments
    bool src0_is_dram = src0_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool src1_is_dram = src1_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool col_indices_is_dram = column_indices_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool indptr_is_dram = indptr_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)src0_is_dram,
        (std::uint32_t)src1_is_dram,
        (std::uint32_t)col_indices_is_dram,
        (std::uint32_t)indptr_is_dram,

        (std::uint32_t)src0_dram_buffer->address(),     // in0_tensor_addr
        (std::uint32_t)1,                               // in0_tensor_stride_w
        (std::uint32_t)Ct,                              // in0_tensor_stride_h

        (std::uint32_t)in0_block_w,               // in0_block_w
        (std::uint32_t)Rt,                         // in0_block_h
        (std::uint32_t)in0_block_w * Rt,           // in0_block_num_tiles

        (std::uint32_t)src1_dram_buffer->address(),  // in1_tensor_addr
        (std::uint32_t)1,                            // in1_tensor_stride_w
        (std::uint32_t)Nt,                           // in1_tensor_stride_h

        (std::uint32_t)in1_block_w,                // in1_block_w
        (std::uint32_t)in0_block_w,                // in1_block_h
        (std::uint32_t)in1_block_w * in0_block_w,  // in1_block_num_tiles

        (std::uint32_t)column_indices_dram_buffer->address(), // col_indices_addr
        (std::uint32_t)indptr_dram_buffer->address(),         // indptr_addr

        (std::uint32_t)num_tiles_for_col_indices,
        (std::uint32_t)num_tiles_for_indptr,
    };

    // Toggle: set to true to have in1 perform the writeback instead of in0.
    bool in1_is_writer = false;

    // Both in0 and in1 receive the full writer CT args; is_output_writer [20] determines who acts.
    std::vector<uint32_t> reader_in0_compile_time_args = reader_compile_time_args;
    reader_in0_compile_time_args.push_back((std::uint32_t)!in1_is_writer);             // is_output_writer      [20]
    reader_in0_compile_time_args.push_back((std::uint32_t)dst_dram_buffer->address()); // out_tensor_addr       [21]
    reader_in0_compile_time_args.push_back((std::uint32_t)Rt * Nt);                    // RtNt                  [22]
    reader_in0_compile_time_args.push_back((std::uint32_t)Nt);                         // Nt                    [23]
    reader_in0_compile_time_args.push_back((std::uint32_t)out_subblock_w);             // out_subblock_w        [24]
    reader_in0_compile_time_args.push_back((std::uint32_t)out_subblock_h);             // out_subblock_h        [25]

    std::vector<uint32_t> reader_in1_compile_time_args = reader_compile_time_args;
    reader_in1_compile_time_args.push_back((std::uint32_t)in1_is_writer);              // is_output_writer      [20]
    reader_in1_compile_time_args.push_back((std::uint32_t)dst_dram_buffer->address()); // out_tensor_addr       [21]
    reader_in1_compile_time_args.push_back((std::uint32_t)Rt * Nt);                    // RtNt                  [22]
    reader_in1_compile_time_args.push_back((std::uint32_t)Nt);                         // Nt                    [23]
    reader_in1_compile_time_args.push_back((std::uint32_t)out_subblock_w);             // out_subblock_w        [24]
    reader_in1_compile_time_args.push_back((std::uint32_t)out_subblock_h);             // out_subblock_h        [25]

    std::vector<uint32_t> compute_kernel_compile_time_args = {
        (std::uint32_t)in0_block_w,
        (std::uint32_t)in0_num_subblocks,
        (std::uint32_t)in0_block_num_tiles,
        (std::uint32_t)in0_subblock_num_tiles,
        (std::uint32_t)in1_num_subblocks,
        (std::uint32_t)in1_block_num_tiles,
        (std::uint32_t)in1_per_core_w,
        (std::uint32_t)out_subblock_h,
        (std::uint32_t)out_subblock_w,
        (std::uint32_t)out_subblock_num_tiles,
        (std::uint32_t)num_iters_x,
    };

    
    
    // Create Kernels
    auto zone_defines = spmm_zone_config::get_zone_defines();
    bool skip_load_balance = extra_defines.count("SKIP_LOAD_BALANCE") > 0;
    for (auto& [k, v] : extra_defines) {
        if (k != "SKIP_LOAD_BALANCE") zone_defines[k] = v;
    }

    bool transpose_NoCs = use_optimal_noc;
    auto noc_riscv_0 = transpose_NoCs ? NOC::RISCV_1_default : NOC::RISCV_0_default;
    auto noc_riscv_1 = transpose_NoCs ? NOC::RISCV_0_default : NOC::RISCV_1_default;
    auto reader_in0_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/rahmy/SC26_submission/block_spmm/kernels/dataflow/reader_in0_naive.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = noc_riscv_0,
            .compile_args = reader_in0_compile_time_args,
            .defines = zone_defines});

    auto reader_in1_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/rahmy/SC26_submission/block_spmm/kernels/dataflow/reader_in1_naive.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = noc_riscv_1,
            .compile_args = reader_in1_compile_time_args,
            .defines = zone_defines});

    // Create compute kernel
    auto mm_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/rahmy/SC26_submission/block_spmm/kernels/compute/bmm_iter.cpp",
        all_cores,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity,
                                .compile_args = compute_kernel_compile_time_args,
                                .defines = zone_defines});

    // Runtime arguments
    std::vector<std::vector<uint32_t>> output_y_indices(num_cores_r, std::vector<uint32_t>());

    if (skip_load_balance) {
        // No load balancing: assign contiguous rows to each core row
        uint32_t rows_per_core = (nnz_rows + num_cores_r - 1) / num_cores_r;
        for (uint32_t row_idx = 0; row_idx < nnz_rows; row_idx++) {
            output_y_indices[row_idx / rows_per_core].push_back(row_idx);
        }
    } else {
        // Load-balancing: sort nnz rows by work (descending) and distribute to core rows
        // via a zigzag so heavy rows are spread evenly across the grid.
        std::vector<int> nnz_row_diffs;
        for (int i = 0; i < (int)a.indptr.size() - 1; i++){
            int diff = a.indptr[i+1] - a.indptr[i];
            if (diff > 0)
                nnz_row_diffs.push_back(diff);
        }
        std::vector<int> perm(nnz_row_diffs.size());
        sortingPermutation(nnz_row_diffs, perm);

        if constexpr (verbose) {
            std::cout << "folded bsr matrix indices: ";
            for (int i = 0; i < (int)folded_bsr_matrix_indices.size(); i++)
                std::cout << folded_bsr_matrix_indices[i] << ' ';
            std::cout << "\nrow diffs: ";
            for (int i = 0; i < (int)nnz_row_diffs.size(); i++)
                std::cout << nnz_row_diffs[i] << ' ';
            std::cout << "\nperm: ";
            for (int i = 0; i < (int)perm.size(); i++)
                std::cout << perm[i] << ' ';
            std::cout << std::endl;
        }

        // Assign folded row indices to each core row via zigzag
        uint32_t num_rows_assigned = 0;
        uint32_t iter_count = 1;
        uint32_t subarray_iter = 0;
        while (num_rows_assigned < nnz_rows) {
            uint32_t num_rows_to_assign = std::min(num_cores_r, nnz_rows - num_rows_assigned);
            subarray_iter = 0;
            for (uint32_t core_row = 0; core_row < num_rows_to_assign; core_row++){
                if (num_rows_assigned++ >= nnz_rows)
                    break;
                output_y_indices[core_row].push_back(perm[num_cores_r * (iter_count - 1) + subarray_iter]);
                subarray_iter++;
            }
            num_rows_to_assign = std::min(num_cores_r, nnz_rows - num_rows_assigned);
            subarray_iter = num_cores_r - num_rows_to_assign;
            for (uint32_t core_row = num_cores_r; core_row > num_cores_r - num_rows_to_assign; core_row--){
                if (num_rows_assigned++ >= nnz_rows)
                    break;
                output_y_indices[core_row - 1].push_back(perm[nnz_rows - num_cores_r * iter_count + subarray_iter]);
                subarray_iter++;
            }
            iter_count++;
        }
    }

    if constexpr (verbose) {
        std::cout << "\n=== Row-to-core-row assignment (" << (skip_load_balance ? "NO LB" : "LOAD BALANCED") << ") ===\n";
        for (uint32_t cr = 0; cr < num_cores_r; cr++) {
            uint32_t total_nnz = 0;
            std::cout << "  core_row[" << cr << "]: rows={";
            for (uint32_t j = 0; j < output_y_indices[cr].size(); j++) {
                uint32_t folded_idx = output_y_indices[cr][j];
                uint32_t orig_row = folded_bsr_matrix_indices[folded_idx];
                uint32_t row_nnz = a.indptr[orig_row + 1] - a.indptr[orig_row];
                if (j > 0) std::cout << ", ";
                std::cout << orig_row << "(" << row_nnz << ")";
                total_nnz += row_nnz;
            }
            std::cout << "} total_nnz=" << total_nnz << "\n";
        }
        std::cout << "===\n" << std::endl;
    }

    for (uint32_t core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
        for (uint32_t core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
            CoreCoord core(core_idx_x, core_idx_y);
            if constexpr (verbose)
                log_info(tt::LogVerif, "Core x {} y {}", core_idx_x, core_idx_y);

            int output_idx_x_start = (core_idx_x * num_iters_x) % num_blocks_x;

            std::vector<uint32_t> reader_in0_runtime_args;
            std::vector<uint32_t> reader_in1_runtime_args;
            std::vector<uint32_t> compute_runtime_args;

            uint32_t num_iters_y_this_core = output_y_indices[core_idx_y].size();
            uint32_t num_iters_x_this_core = std::min(num_iters_x, num_blocks_x - output_idx_x_start + 1);

            // Both readers: num_iters_x, num_iters_y, output_idx_x_start
            reader_in0_runtime_args.push_back(num_iters_x_this_core);
            reader_in0_runtime_args.push_back(num_iters_y_this_core);
            reader_in0_runtime_args.push_back(output_idx_x_start);
            reader_in1_runtime_args.push_back(num_iters_x_this_core);
            reader_in1_runtime_args.push_back(num_iters_y_this_core);
            reader_in1_runtime_args.push_back(output_idx_x_start);

            compute_runtime_args.push_back(num_iters_y_this_core);
            for (int iter_y = 0; iter_y < (int)num_iters_y_this_core; iter_y++) {
                uint32_t folded_output_idx_y = output_y_indices[core_idx_y][iter_y];
                uint32_t output_idx_y = folded_bsr_matrix_indices[folded_output_idx_y];
                // The writer kernel receives interleaved (y_coord, folded_y_coord) pairs;
                // the non-writer kernel receives only y_coords.
                reader_in0_runtime_args.push_back(output_idx_y);
                if (!in1_is_writer) reader_in0_runtime_args.push_back(folded_output_idx_y);
                reader_in1_runtime_args.push_back(output_idx_y);
                if (in1_is_writer) reader_in1_runtime_args.push_back(folded_output_idx_y);
                compute_runtime_args.push_back(a.indptr[output_idx_y + 1] - a.indptr[output_idx_y]);
            }
            // out_tensor_start_tile_id goes to the writer kernel only
            if (!in1_is_writer)
                reader_in0_runtime_args.push_back(output_idx_x_start * in1_block_w);
            else
                reader_in1_runtime_args.push_back(output_idx_x_start * in1_block_w);

            tt_metal::SetRuntimeArgs(program, reader_in0_id, core, reader_in0_runtime_args);
            tt_metal::SetRuntimeArgs(program, reader_in1_id, core, reader_in1_runtime_args);
            tt_metal::SetRuntimeArgs(program, mm_kernel_id, core, compute_runtime_args);

            if constexpr (verbose) {
                if (num_iters_x_this_core < num_iters_x || num_iters_y_this_core < num_iters_y) {
                    log_info(tt::LogVerif, " -- Num iters diverged! --");
                    log_info(tt::LogVerif, "(num_iters_x_this_core) = {}", num_iters_x_this_core);
                    log_info(tt::LogVerif, "(num_iters_y_this_core) = {}", num_iters_y_this_core);
                }
            }

            if (verbose && core_idx_x == 0 && core_idx_y == 0) {
                log_info(tt::LogVerif, " -- In0 Reader Args --");
                log_info(tt::LogVerif, "reader_in0_arg[0] (num_iters_x) = {}", reader_in0_runtime_args[0]);
                log_info(tt::LogVerif, "reader_in0_arg[1] (num_iters_y) = {}", reader_in0_runtime_args[1]);
                log_info(tt::LogVerif, "reader_in0_arg[2] (output_idx_x_start) = {}", reader_in0_runtime_args[2]);
                for (size_t i = 0; i < num_iters_y_this_core; ++i) {
                    log_info(tt::LogVerif, "reader_in0_arg[{}] (y_coord) = {}", 3 + i*2, reader_in0_runtime_args[3 + i*2]);
                    log_info(tt::LogVerif, "reader_in0_arg[{}] (folded_y) = {}", 3 + i*2 + 1, reader_in0_runtime_args[3 + i*2 + 1]);
                }
                log_info(tt::LogVerif, "reader_in0_arg[{}] (out_tensor_start_tile_id) = {}", 3 + num_iters_y_this_core*2, reader_in0_runtime_args[3 + num_iters_y_this_core*2]);

                log_info(tt::LogVerif, " -- In1 Reader Args --");
                log_info(tt::LogVerif, "reader_in1_arg[0] (num_iters_x) = {}", reader_in1_runtime_args[0]);
                log_info(tt::LogVerif, "reader_in1_arg[1] (num_iters_y) = {}", reader_in1_runtime_args[1]);
                log_info(tt::LogVerif, "reader_in1_arg[2] (output_idx_x_start) = {}", reader_in1_runtime_args[2]);
                for (size_t i = 0; i < num_iters_y_this_core; ++i)
                    log_info(tt::LogVerif, "reader_in1_arg[{}] (y_coord) = {}", i + 3, reader_in1_runtime_args[i+3]);

                log_info(tt::LogVerif, " -- Compute Args --");
                log_info(tt::LogVerif, "compute_arg[0] (num_iters_y) = {}", compute_runtime_args[0]);
                for (size_t i = 1; i < compute_runtime_args.size(); ++i)
                    log_info(tt::LogVerif, "compute_arg[{}] (row_size) = {}", i, compute_runtime_args[i]);
            }
        }
    }

    // Pad indexing data to match tile-aligned DRAM buffer sizes
    std::vector<uint32_t> padded_col_indices(dram_buffer_col_indices_size / sizeof(uint32_t), 0);
    std::copy(a.indices.begin(), a.indices.end(), padded_col_indices.begin());

    std::vector<uint32_t> padded_indptr(dram_buffer_indptr_size / sizeof(uint32_t), 0);
    std::copy(a.indptr.begin(), a.indptr.end(), padded_indptr.begin());

    // EnqueueWriteBuffers
    if constexpr (verbose)
        log_info(tt::LogVerif, " -- Initiating H2D transfers --");
    EnqueueWriteBuffer(cq, src0_dram_buffer, a.data.data(), false);
    EnqueueWriteBuffer(cq, src1_dram_buffer, b.data.data(), false);
    EnqueueWriteBuffer(cq, indptr_dram_buffer, padded_indptr.data(), false);
    EnqueueWriteBuffer(cq, column_indices_dram_buffer, padded_col_indices.data(), false);

    if constexpr (is_profiling) {
        int num_iters = 10;
        EnqueueProgram(cq, program, true);
        ZoneScopedNC("Device program Loop", tracy::Color::Aquamarine);
        for (int i = 0; i < num_iters; i++)
            EnqueueProgram(cq, program, true);
    }
    else {
        if constexpr (verbose)
            log_info(tt::LogVerif, " -- Enqueueing program --");
        EnqueueProgram(cq, program, false);
    }

    if constexpr (verbose)
        log_info(tt::LogVerif, " -- Program returned --");

    // EnqueueReadSubBuffers
    uint32_t nonzero_row_index = 0;
    for (size_t row_index = 0; row_index < a.indptr.size() - 1; row_index++) {
        if (a.indptr[row_index+1] - a.indptr[row_index] == 0)
            continue;
        BufferRegion DRAM_row(nonzero_row_index * dram_buffer_dst_row_size, dram_buffer_dst_row_size);
        EnqueueReadSubBuffer(cq, dst_dram_buffer, output.data.data() + (row_index * R * N), DRAM_row, true);
        nonzero_row_index++;
    }

    if constexpr (verbose)
        log_info(tt::LogVerif, " -- Finished reading output --");
    Finish(cq);
}

// Public thin wrapper (matches original API and HostCodeFunctionPtr)
template<bool verbose, bool is_profiling, bool use_optimal_noc>
void bsr_spmm_multicore_load_balanced_new_DM(
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output,
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K,
    uint32_t R, uint32_t C, uint32_t B, IDevice* device) {
    bsr_spmm_multicore_load_balanced_new_DM_impl<verbose, is_profiling, use_optimal_noc>(a, b, output, bcast_batch, nnz_blocks, M, N, K, R, C, B, device, {});
}

// Ablation skip wrappers
template<bool verbose, bool is_profiling, bool use_optimal_noc>
void bsr_spmm_multicore_load_balanced_new_DM_no_a_read(
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output,
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K,
    uint32_t R, uint32_t C, uint32_t B, IDevice* device) {
    bsr_spmm_multicore_load_balanced_new_DM_impl<verbose, is_profiling, use_optimal_noc>(a, b, output, bcast_batch, nnz_blocks, M, N, K, R, C, B, device, {{"SKIP_IN0_DRAM_READ", "1"}});
}
template<bool verbose, bool is_profiling, bool use_optimal_noc>
void bsr_spmm_multicore_load_balanced_new_DM_no_b_read(
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output,
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K,
    uint32_t R, uint32_t C, uint32_t B, IDevice* device) {
    bsr_spmm_multicore_load_balanced_new_DM_impl<verbose, is_profiling, use_optimal_noc>(a, b, output, bcast_batch, nnz_blocks, M, N, K, R, C, B, device, {{"SKIP_IN1_DRAM_READ", "1"}});
}
template<bool verbose, bool is_profiling, bool use_optimal_noc>
void bsr_spmm_multicore_load_balanced_new_DM_no_compute(
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output,
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K,
    uint32_t R, uint32_t C, uint32_t B, IDevice* device) {
    bsr_spmm_multicore_load_balanced_new_DM_impl<verbose, is_profiling, use_optimal_noc>(a, b, output, bcast_batch, nnz_blocks, M, N, K, R, C, B, device, {{"SKIP_COMPUTE", "1"}});
}
template<bool verbose, bool is_profiling, bool use_optimal_noc>
void bsr_spmm_multicore_load_balanced_new_DM_no_write(
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output,
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K,
    uint32_t R, uint32_t C, uint32_t B, IDevice* device) {
    bsr_spmm_multicore_load_balanced_new_DM_impl<verbose, is_profiling, use_optimal_noc>(a, b, output, bcast_batch, nnz_blocks, M, N, K, R, C, B, device, {{"SKIP_DRAM_WRITE", "1"}});
}

// No-load-balance wrapper
template<bool verbose, bool is_profiling, bool use_optimal_noc>
void bsr_spmm_multicore_load_balanced_new_DM_no_lb(
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output,
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K,
    uint32_t R, uint32_t C, uint32_t B, IDevice* device) {
    bsr_spmm_multicore_load_balanced_new_DM_impl<verbose, is_profiling, use_optimal_noc>(a, b, output, bcast_batch, nnz_blocks, M, N, K, R, C, B, device, {{"SKIP_LOAD_BALANCE", "1"}});
}

// Explicit template instantiations
template void bsr_spmm_multicore_load_balanced_new_DM<false, false>(
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output,
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K,
    uint32_t R, uint32_t C, uint32_t B, IDevice* device);
template void bsr_spmm_multicore_load_balanced_new_DM<true, false>(
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output,
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K,
    uint32_t R, uint32_t C, uint32_t B, IDevice* device);
template void bsr_spmm_multicore_load_balanced_new_DM<false, true>(
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output,
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K,
    uint32_t R, uint32_t C, uint32_t B, IDevice* device);
// Ablation skip wrappers (profiling only)
template void bsr_spmm_multicore_load_balanced_new_DM_no_a_read<false, true>(
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output,
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K,
    uint32_t R, uint32_t C, uint32_t B, IDevice* device);
template void bsr_spmm_multicore_load_balanced_new_DM_no_b_read<false, true>(
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output,
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K,
    uint32_t R, uint32_t C, uint32_t B, IDevice* device);
template void bsr_spmm_multicore_load_balanced_new_DM_no_compute<false, true>(
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output,
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K,
    uint32_t R, uint32_t C, uint32_t B, IDevice* device);
template void bsr_spmm_multicore_load_balanced_new_DM_no_write<false, true>(
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output,
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K,
    uint32_t R, uint32_t C, uint32_t B, IDevice* device);
template void bsr_spmm_multicore_load_balanced_new_DM_no_lb<false, true>(
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output,
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K,
    uint32_t R, uint32_t C, uint32_t B, IDevice* device);

}
