// SNF for in0 (store-and-forward along core columns) + CDA for in1 (chain of direct addressing along core rows)

#include "../host_code.hpp"
#include "spmm_zone_config.hpp"

namespace bsr_host_code{

template<bool verbose, bool is_profiling, bool use_optimal_noc = true,
         bool in0_left_to_right = true, bool in1_bottom_to_top = true>
void bsr_spmm_multicore_snfin0_cdain1_impl(
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

    CommandQueue& cq = device->command_queue();
    Program program{};

    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    uint32_t single_tile_size = detail::TileSize(cb_data_format);

    tt::DataFormat indexing_data_format = tt::DataFormat::Int32;
    uint32_t indexing_data_single_tile_size = detail::TileSize(indexing_data_format);
    uint32_t dram_buffer_indptr_size =
        sizeof(int) * a.indptr.size();
    // Round up to tile size
    dram_buffer_indptr_size = indexing_data_single_tile_size * ((indexing_data_single_tile_size - 1 + dram_buffer_indptr_size) / (indexing_data_single_tile_size));

    uint32_t dram_buffer_col_indices_size =
        sizeof(int) * a.indices.size();
    // Round up to tile size
    dram_buffer_col_indices_size = indexing_data_single_tile_size * ((indexing_data_single_tile_size - 1 + dram_buffer_col_indices_size) / (indexing_data_single_tile_size));

    uint32_t num_tiles_for_output_y_indices = 0;
    uint32_t num_tiles_for_col_indices = dram_buffer_col_indices_size / indexing_data_single_tile_size;
    uint32_t num_tiles_for_indptr = dram_buffer_indptr_size / indexing_data_single_tile_size;
    uint32_t num_tiles_indexing = num_tiles_for_col_indices + num_tiles_for_indptr + num_tiles_for_output_y_indices;

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

    // Core grid assignment
    std::vector<uint32_t> folded_bsr_matrix_indices;
    uint32_t nnz_rows = 0;
    uint32_t folded_index = 0;
    for (uint32_t i = 0; i < a.indptr.size() - 1; i++) {
        if (a.indptr[i+1] - a.indptr[i] > 0){
            folded_bsr_matrix_indices.push_back(folded_index);
            nnz_rows++;
        }
        folded_index++;
    }
    folded_bsr_matrix_indices.push_back(folded_index);
    uint32_t height_of_folded_matrix = Rt * nnz_rows;

    if constexpr (verbose) {
        log_info(tt::LogVerif, " -- folded_bsr_matrix_indices (size={}) --", folded_bsr_matrix_indices.size());
        for (uint32_t i = 0; i < folded_bsr_matrix_indices.size(); i++) {
            log_info(tt::LogVerif, "   folded_bsr_matrix_indices[{}] = {}", i, folded_bsr_matrix_indices[i]);
        }
        log_info(tt::LogVerif, " -- indptr (size={}) --", a.indptr.size());
        for (uint32_t i = 0; i < a.indptr.size(); i++) {
            log_info(tt::LogVerif, "   indptr[{}] = {}", i, a.indptr[i]);
        }
        log_info(tt::LogVerif, " -- nnz_rows={}, num_block_rows={} --", nnz_rows, (uint32_t)(a.indptr.size() - 1));
    }

    uint32_t in0_block_h = Rt;
    uint32_t in0_block_w = Ct;
    uint32_t in1_block_w = get_Npc_from_BSR_block_size(Nt, in0_block_h, in0_block_w, num_cores_x, num_cores_y, num_tiles_indexing, nnz_rows);

    TT_ASSERT(Mt % in0_block_h == 0);
    TT_ASSERT(Nt % in1_block_w == 0);
    TT_ASSERT(Kt % in0_block_w == 0);

    if constexpr (verbose) {
        log_info(tt::LogVerif, "Rt={}, Ct={}, NpC={}", Rt, Ct, in1_block_w);
    }

    uint32_t num_blocks_x = Nt / in1_block_w;
    uint32_t num_blocks_y = nnz_rows;
    uint32_t num_blocks_total = num_blocks_x * num_blocks_y;

    uint32_t num_iters_x = (num_blocks_x + num_cores_x - 1) / num_cores_x;
    uint32_t num_iters_y = (num_blocks_y + num_cores_y - 1) / num_cores_y;

    uint32_t num_work_regions = (num_blocks_total + num_iters_x * num_iters_y - 1)/ (num_iters_x * num_iters_y);
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

    CoreCoord start_core = {0, 0};
    CoreCoord core_range(0, 0);
    if ( (num_blocks_y / num_iters_y) <= num_cores_y &&
        (num_blocks_x / num_iters_x) <= num_cores_x) {
        core_range.x = (num_blocks_x + num_iters_x - 1) / num_iters_x;
        core_range.y = (num_blocks_y + num_iters_y - 1) / num_iters_y;
    }
    uint32_t start_core_x = start_core.x;
    uint32_t start_core_y = start_core.y;
    uint32_t num_cores_c = core_range.x;
    uint32_t num_cores_r = core_range.y;

    // this variable exists purely to allow the constructor to CoreRange to not fail when we only use one column of the core grid
    CoreRange all_cores(
        {(std::size_t)start_core_x, (std::size_t)start_core_y},
        {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + num_cores_r - 1});

    // in0 SNF: injector column reads from DRAM, receivers get data forwarded
    // L2R: injector = leftmost column, R2L: injector = rightmost column
    uint32_t injector_col = in0_left_to_right ? start_core_x : start_core_x + num_cores_c - 1;
    CoreRange in0_injector_cores(
        {(std::size_t)injector_col, (std::size_t)start_core_y},
        {(std::size_t)injector_col, (std::size_t)start_core_y + num_cores_r - 1});

    uint32_t column_offset = num_cores_c > 1 ? num_cores_c : num_cores_c + 1;
    // Receiver cores: all columns except the injector column
    uint32_t recv_start_x, recv_end_x;
    if constexpr (in0_left_to_right) {
        recv_start_x = start_core_x + 1;
        recv_end_x   = start_core_x + column_offset - 1;
    } else {
        recv_start_x = start_core_x;
        recv_end_x   = start_core_x + column_offset - 2;
    }
    CoreRange in0_receiver_cores(
        {(std::size_t)recv_start_x, (std::size_t)start_core_y},
        {(std::size_t)recv_end_x, (std::size_t)start_core_y + num_cores_r - 1});

    CoreRange top_row(
        {(std::size_t)start_core_x, (std::size_t)start_core_y},
        {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y});

    uint32_t row_offset = num_cores_r > 1 ? num_cores_r : num_cores_r + 1;
    CoreRange all_but_top_row(
        {(std::size_t)start_core_x, (std::size_t)start_core_y + 1},
        {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + row_offset - 1});

    if constexpr (verbose) {
        log_info(tt::LogVerif, "core_range         {}", core_range);
        log_info(tt::LogVerif, "all cores          {}", all_cores);
        log_info(tt::LogVerif, "in0 injector cores {}", in0_injector_cores);
        log_info(tt::LogVerif, "in0 receiver cores {}, used? {}", in0_receiver_cores, num_cores_c > 1);
    }

    auto in0_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto in0_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto in1_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto in1_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto in1_barrier_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto in1_release_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);

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

    // DRAM buffers initialization
    uint32_t dram_buffer_dst_row_size =
        single_tile_size * Rt * Nt;

    uint32_t dram_buffer_dst_total_size = dram_buffer_dst_row_size * nnz_rows;

    uint32_t dram_buffer_A_size =
        single_tile_size * Rt * Ct * nnz_blocks;
    uint32_t dram_buffer_B_size =
        single_tile_size * Nt * Kt;

    uint32_t src0_block_size = in0_block_w * in0_block_h * single_tile_size;
    uint32_t src1_block_size = in1_block_w * in0_block_w * single_tile_size;

    auto dst_dram_buffer = MakeBuffer(device, dram_buffer_dst_total_size, single_tile_size);
    auto src0_dram_buffer = MakeBuffer(device, dram_buffer_A_size, src0_block_size); // TODO: will this let the D2Inj performance easaier to reason about?
    auto src1_dram_buffer = MakeBuffer(device, dram_buffer_B_size, src1_block_size);
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
        log_info(tt::LogVerif, " -- Core Grid Allocaiton Information --");
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
    uint32_t src0_cb_index = CBIndex::c_0;  // 0
    CircularBufferConfig cb_src0_config = CircularBufferConfig(in0_CB_size, {{src0_cb_index, cb_data_format}})
                                              .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t src1_cb_index = CBIndex::c_1;  // 1
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

    uint32_t column_indices_cb_index = CBIndex::c_2;  // 2
    CircularBufferConfig cb_column_indices_config = CircularBufferConfig(
        dram_buffer_col_indices_size, {{column_indices_cb_index, tt::DataFormat::Int32}})
                                                .set_page_size(column_indices_cb_index, indexing_data_single_tile_size);
    auto cb_column_indices = tt_metal::CreateCircularBuffer(program, all_cores, cb_column_indices_config);

    auto indptr_cb_index = CBIndex::c_3; // 3
    auto cb_indptr = MakeCircularBuffer(program, all_cores, indptr_cb_index, dram_buffer_indptr_size, indexing_data_single_tile_size, indexing_data_format);

    /*
        Compile-time arguments.
    */

    bool in1_is_writer = false;  // flip to switch which RISC writes output to DRAM

    bool src0_is_dram = src0_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool src1_is_dram = src1_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool col_indices_is_dram = column_indices_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool indptr_is_dram = indptr_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    // in0 injector compile-time args (IDENTICAL to SNF)
    std::vector<uint32_t> in0_injector_compile_time_args = {
        (std::uint32_t)src0_is_dram,
        (std::uint32_t)src1_is_dram,
        (std::uint32_t)col_indices_is_dram,
        (std::uint32_t)indptr_is_dram,

        (std::uint32_t)src0_dram_buffer->address(),     // in0_tensor_addr
        (std::uint32_t)1,                               // in0_tensor_stride_w
        (std::uint32_t)Ct,                              // in0_tensor_stride_h

        (std::uint32_t)in0_block_w,               // in0_block_w
        (std::uint32_t)Rt,                         // in0_block_h
        (std::uint32_t)in0_block_w * Rt,  // in0_block_num_tiles

        (std::uint32_t)src1_dram_buffer->address(),  // in1_tensor_addr
        (std::uint32_t)1,                            // in1_tensor_stride_w
        (std::uint32_t)Nt,                           // in1_tensor_stride_h

        (std::uint32_t)in1_block_w,                // in1_block_w
        (std::uint32_t)in0_block_w,               // in1_block_h
        (std::uint32_t)in1_block_w * in0_block_w,  // in1_block_num_tiles

        (std::uint32_t)column_indices_dram_buffer->address(), // NoC args, column indices
        (std::uint32_t)indptr_dram_buffer->address(), // NoC args, indptr

        (std::uint32_t)num_tiles_for_col_indices,
        (std::uint32_t)num_tiles_for_indptr,

        in0_sender_semaphore_id,
        in0_receiver_semaphore_id,
        (std::uint32_t)true,                            // is_injector_core
        (std::uint32_t)!in1_is_writer,                  // is_output_writer
        (std::uint32_t)dst_dram_buffer->address(),      // out_buffer_addr

        (std::uint32_t)Rt * Nt,  // RtNt
        (std::uint32_t)Nt,

        // writer args
        (std::uint32_t)out_subblock_w,
        (std::uint32_t)out_subblock_h,
    };

    // in0 receiver compile-time args (IDENTICAL to SNF)
    std::vector<uint32_t> in0_receiver_compile_time_args = {
        (std::uint32_t)src0_is_dram,
        (std::uint32_t)src1_is_dram,
        (std::uint32_t)col_indices_is_dram,
        (std::uint32_t)indptr_is_dram,

        (std::uint32_t)src0_dram_buffer->address(),     // in0_tensor_addr
        (std::uint32_t)1,                               // in0_tensor_stride_w
        (std::uint32_t)Ct,                              // in0_tensor_stride_h

        (std::uint32_t)in0_block_w,               // in0_block_w
        (std::uint32_t)Rt,                         // in0_block_h
        (std::uint32_t)in0_block_w * Rt,  // in0_block_num_tiles

        (std::uint32_t)src1_dram_buffer->address(),  // in1_tensor_addr
        (std::uint32_t)1,                            // in1_tensor_stride_w
        (std::uint32_t)Nt,                           // in1_tensor_stride_h

        (std::uint32_t)in1_block_w,                // in1_block_w
        (std::uint32_t)in0_block_w,               // in1_block_h
        (std::uint32_t)in1_block_w * in0_block_w,  // in1_block_num_tiles

        (std::uint32_t)column_indices_dram_buffer->address(), // NoC args, column indices
        (std::uint32_t)indptr_dram_buffer->address(), // NoC args, indptr

        (std::uint32_t)num_tiles_for_col_indices,
        (std::uint32_t)num_tiles_for_indptr,
        in0_sender_semaphore_id,
        in0_receiver_semaphore_id,
        (std::uint32_t)false,                    // is_injector_core
        (std::uint32_t)!in1_is_writer,           // is_output_writer
        (std::uint32_t)dst_dram_buffer->address(),      // out_buffer_addr

        (std::uint32_t)Rt * Nt,  // RtNt
        (std::uint32_t)Nt,

        // writer args
        (std::uint32_t)out_subblock_w,
        (std::uint32_t)out_subblock_h,
    };

    // in1 CDA reader compile-time args (slots 0-25 identical to SNF in1, + slots 26-27 for CDA semaphores)
    std::vector<uint32_t> in1_reader_compile_time_args = {
        (std::uint32_t)src0_is_dram,
        (std::uint32_t)src1_is_dram,
        (std::uint32_t)col_indices_is_dram,
        (std::uint32_t)indptr_is_dram,

        (std::uint32_t)src0_dram_buffer->address(),     // in0_tensor_addr
        (std::uint32_t)1,                               // in0_tensor_stride_w
        (std::uint32_t)Ct,                              // in0_tensor_stride_h

        (std::uint32_t)in0_block_w,               // in0_block_w
        (std::uint32_t)Rt,                         // in0_block_h
        (std::uint32_t)in0_block_w * Rt,  // in0_block_num_tiles

        (std::uint32_t)src1_dram_buffer->address(),  // in1_tensor_addr
        (std::uint32_t)1,                            // in1_tensor_stride_w
        (std::uint32_t)Nt,                           // in1_tensor_stride_h

        (std::uint32_t)in1_block_w,                // in1_block_w
        (std::uint32_t)in0_block_w,               // in1_block_h
        (std::uint32_t)in1_block_w * in0_block_w,  // in1_block_num_tiles

        (std::uint32_t)column_indices_dram_buffer->address(), // NoC args, column indices
        (std::uint32_t)indptr_dram_buffer->address(), // NoC args, indptr

        (std::uint32_t)num_tiles_for_col_indices,
        (std::uint32_t)num_tiles_for_indptr,

        (std::uint32_t)in1_is_writer,                            // is_output_writer [20]
        (std::uint32_t)dst_dram_buffer->address(),               // out_tensor_addr [21]
        (std::uint32_t)Rt * Nt,                                  // RtNt [22]
        (std::uint32_t)Nt,                                       // Nt [23]
        (std::uint32_t)out_subblock_w,                           // out_subblock_w [24]
        (std::uint32_t)out_subblock_h,                           // out_subblock_h [25]

        in1_sender_semaphore_id,                                 // [26]
        in1_receiver_semaphore_id,                               // [27]
        in1_barrier_semaphore_id,                                // [28]
        in1_release_semaphore_id,                                // [29]
    };

    if constexpr (verbose) {
        log_info(tt::LogVerif, " -- in1 CDA reader compile-time args (size={}) --", in1_reader_compile_time_args.size());
        const char* in1_ct_labels[] = {
            "src0_is_dram", "src1_is_dram", "col_indices_is_dram", "indptr_is_dram",
            "in0_tensor_addr", "in0_tensor_stride_w", "in0_tensor_stride_h",
            "in0_block_w", "in0_block_h(Rt)", "in0_block_num_tiles",
            "in1_tensor_addr", "in1_tensor_stride_w", "in1_tensor_stride_h",
            "in1_block_w", "in1_block_h", "in1_block_num_tiles",
            "col_indices_addr", "indptr_addr",
            "num_tiles_col_indices", "num_tiles_indptr",
            "is_output_writer", "out_tensor_addr", "RtNt", "Nt",
            "out_subblock_w", "out_subblock_h",
            "in1_sender_sem", "in1_receiver_sem",
            "in1_barrier_sem", "in1_release_sem"
        };
        for (uint32_t i = 0; i < in1_reader_compile_time_args.size(); i++) {
            const char* label = (i < 30) ? in1_ct_labels[i] : "???";
            log_info(tt::LogVerif, "   ct_arg[{}] ({}) = {}", i, label, in1_reader_compile_time_args[i]);
        }
    }

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

    bool out_is_dram = dst_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)out_is_dram,
        (std::uint32_t)num_iters_x,
    };

    // Check for host-level flags before merging into kernel defines
    bool skip_load_balance = extra_defines.count("SKIP_LOAD_BALANCE") > 0;

    // Create Kernels
    auto zone_defines = spmm_zone_config::get_zone_defines();
    for (auto& [k, v] : extra_defines) {
        if (k != "SKIP_LOAD_BALANCE") zone_defines[k] = v;
    }

    bool transpose_NoCs = use_optimal_noc;
    auto noc_riscv_0 = transpose_NoCs ? NOC::RISCV_1_default : NOC::RISCV_0_default;
    auto noc_riscv_1 = transpose_NoCs ? NOC::RISCV_0_default : NOC::RISCV_1_default;

    auto compute_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/rahmy/SC26_submission/block_spmm/kernels/compute/bmm_iter.cpp",
        all_cores,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en=false,
            .math_approx_mode=false,
            .compile_args = compute_kernel_compile_time_args,
            .defines = zone_defines});

    // CDA in1 reader (replaces SNF in1 reader)
    std::string in1_kernel_path = in1_bottom_to_top
        ? "tt_metal/programming_examples/rahmy/SC26_submission/block_spmm/kernels/dataflow/reader_in1_chain_of_direct_addressing.cpp"
        : "tt_metal/programming_examples/rahmy/SC26_submission/block_spmm/kernels/dataflow/reader_in1_chain_of_direct_addressingV2.cpp";
    auto in1_reader_id = tt_metal::CreateKernel(
        program,
        in1_kernel_path,
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = noc_riscv_1,
            .compile_args = in1_reader_compile_time_args,
            .defines = zone_defines});

    // SNF in0 reader (UNCHANGED from SNF version)
    auto in0_injector_and_writer_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/rahmy/SC26_submission/block_spmm/kernels/dataflow/reader_snf_in0_reader.cpp",
        in0_injector_cores,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = noc_riscv_0,
            .compile_args = in0_injector_compile_time_args,
            .defines = zone_defines});

    KernelHandle in0_receiver_and_writer_id = 0;
    if (num_cores_c > 1){
        if constexpr (verbose) {
                log_info(tt::LogVerif, "receiver cores {}", in0_receiver_cores);
        }
        in0_receiver_and_writer_id = tt_metal::CreateKernel(
            program,
            "tt_metal/programming_examples/rahmy/SC26_submission/block_spmm/kernels/dataflow/reader_snf_in0_reader.cpp",
            in0_receiver_cores,
            tt_metal::DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = noc_riscv_0,
                .compile_args = in0_receiver_compile_time_args,
                .defines = zone_defines});
    }

    // 1. initialize a vector for each row of cores
    std::vector<std::vector<uint32_t>> output_y_indices(num_cores_r, std::vector<uint32_t>());

    if (skip_load_balance) {
        // No load balancing: assign contiguous rows to each core row
        uint32_t rows_per_core = (nnz_rows + num_cores_r - 1) / num_cores_r;
        for (uint32_t row_idx = 0; row_idx < nnz_rows; row_idx++) {
            output_y_indices[row_idx / rows_per_core].push_back(row_idx);
        }
    } else {
        // Find Perms -- sort only the nnz rows so perm values are folded indices
        std::vector<int> nnz_row_diffs;
        for (int i = 0; i < a.indptr.size() - 1; i++){
            int diff = a.indptr[i+1] - a.indptr[i];
            if (diff > 0) {
                nnz_row_diffs.push_back(diff);
            }
        }
        std::vector<int> perm(nnz_row_diffs.size());
        sortingPermutation(nnz_row_diffs, perm);

        if constexpr (verbose) {
            log_info(tt::LogVerif, " -- nnz_row_diffs (size={}) --", nnz_row_diffs.size());
            for (uint32_t i = 0; i < nnz_row_diffs.size(); i++) {
                log_info(tt::LogVerif, "   nnz_row_diffs[{}] = {}", i, nnz_row_diffs[i]);
            }
            log_info(tt::LogVerif, " -- perm (size={}) --", perm.size());
            for (uint32_t i = 0; i < perm.size(); i++) {
                log_info(tt::LogVerif, "   perm[{}] = {} (nnz_row_diffs[perm[{}]] = {})", i, perm[i], i, nnz_row_diffs[perm[i]]);
            }
        }

        // 2. While count is less than num output blocks, sweep the core grid
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

    // Assign runtime args
    for (uint32_t core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
        for (uint32_t core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
            CoreCoord core(core_idx_x, core_idx_y);

            int output_idx_x_start = (core_idx_x * num_iters_x) % num_blocks_x;

            std::vector<uint32_t> in0_snf_reader_runtime_args;
            std::vector<uint32_t> compute_runtime_args;
            std::vector<uint32_t> in1_reader_runtime_args;

            uint32_t num_iters_y_this_core = output_y_indices[core_idx_y].size();
            uint32_t num_iters_x_this_core = std::min(num_iters_x, num_blocks_x - output_idx_x_start + 1);

            if constexpr (verbose) {
                log_info(tt::LogVerif, " -- Core ({},{}) iter assignment: num_iters_x={}, num_iters_y={}, output_idx_x_start={} --",
                    core_idx_x, core_idx_y, num_iters_x_this_core, num_iters_y_this_core, output_idx_x_start);
            }

            // ── in0 SNF reader runtime args (IDENTICAL to SNF version) ──
            in0_snf_reader_runtime_args.push_back(num_iters_x_this_core);
            in0_snf_reader_runtime_args.push_back(num_iters_y_this_core);
            in0_snf_reader_runtime_args.push_back(output_idx_x_start);

            // ── in1 CDA reader runtime args (NEW layout) ──
            in1_reader_runtime_args.push_back(num_iters_x_this_core);       // [0] num_iters_x
            in1_reader_runtime_args.push_back(num_iters_y_this_core);       // [1] num_iters_y
            in1_reader_runtime_args.push_back(output_idx_x_start);          // [2] output_idx_x_start
            in1_reader_runtime_args.push_back(core_idx_y);                  // [3] my_core_idx_y
            in1_reader_runtime_args.push_back(num_cores_r);                 // [4] num_cores_in_column

            // [5] noc_x_for_column (physical NOC x from column's logical x)
            auto column_phys_core = device->worker_core_from_logical_core(CoreCoord(core_idx_x, 0));
            in1_reader_runtime_args.push_back((std::uint32_t)column_phys_core.x);

            // [6..6+N-1] noc_y_table[0..num_cores_r-1] (physical NOC y per core row)
            for (uint32_t r = 0; r < num_cores_r; r++) {
                auto phys_core = device->worker_core_from_logical_core(CoreCoord(core_idx_x, r));
                in1_reader_runtime_args.push_back((std::uint32_t)phys_core.y);
            }

            // if constexpr (verbose) {
            //     log_info(tt::LogVerif, " -- Core ({},{}) CDA noc coords: noc_x_for_column={} --",
            //         core_idx_x, core_idx_y, (uint32_t)column_phys_core.x);
            //     for (uint32_t r = 0; r < num_cores_r; r++) {
            //         auto phys_core = device->worker_core_from_logical_core(CoreCoord(core_idx_x, r));
            //         log_info(tt::LogVerif, "   noc_y_table[{}] = {} (logical row {})", r, (uint32_t)phys_core.y, r);
            //     }
            // }

            // ── compute runtime args ──
            compute_runtime_args.push_back(num_iters_y_this_core);

            // ── Per-iter_y y_coords and folded_y_coords ──
            for (int iter_y = 0; iter_y < num_iters_y_this_core; iter_y++) {
                uint32_t folded_output_idx_y = output_y_indices[core_idx_y][iter_y];
                uint32_t output_idx_y = folded_bsr_matrix_indices[folded_output_idx_y];
                // if constexpr (verbose) {
                //     log_info(tt::LogVerif, " -- Core ({},{}) iter_y={}: folded_output_idx_y={} -> output_idx_y={} --",
                //         core_idx_x, core_idx_y, iter_y, folded_output_idx_y, output_idx_y);
                //     if (output_idx_y + 1 < a.indptr.size()) {
                //         log_info(tt::LogVerif, "     indptr[{}]={}, indptr[{}]={}, row_nnz={}",
                //             output_idx_y, a.indptr[output_idx_y],
                //             output_idx_y + 1, a.indptr[output_idx_y + 1],
                //             a.indptr[output_idx_y + 1] - a.indptr[output_idx_y]);
                //     } else {
                //         log_info(tt::LogVerif, "     *** BUG: output_idx_y+1={} >= indptr.size()={}, would access out-of-bounds! ***",
                //             output_idx_y + 1, a.indptr.size());
                //     }
                // }
                // in0 SNF reader: y_coord and folded_y_coord (always both)
                in0_snf_reader_runtime_args.push_back(output_idx_y);
                in0_snf_reader_runtime_args.push_back(folded_output_idx_y);

                // in1 CDA reader: y_coord (always), folded_y_coord (only if is_output_writer)
                in1_reader_runtime_args.push_back(output_idx_y);
                if (in1_is_writer) in1_reader_runtime_args.push_back(folded_output_idx_y);

                // compute: num_blocks_in_row for this iter_y
                compute_runtime_args.push_back(a.indptr[output_idx_y + 1] - a.indptr[output_idx_y]);
            }

            // in0 SNF reader: out_tensor_start_tile_id (always parsed by in0 kernel)
            in0_snf_reader_runtime_args.push_back(output_idx_x_start * in1_block_w);
            // in1 CDA reader: out_tensor_start_tile_id (only if in1_is_writer)
            if (in1_is_writer) in1_reader_runtime_args.push_back(output_idx_x_start * in1_block_w);

            // in0 SNF reader: num_cores_y, dest/sender noc coords, is_sink_core
            in0_snf_reader_runtime_args.push_back(num_iters_y_this_core);

            bool is_injector_core, is_sink_core;
            CoreCoord in0_prev_core, in0_next_core;
            if constexpr (in0_left_to_right) {
                is_injector_core = core_idx_x == 0;
                is_sink_core     = core_idx_x == (num_cores_c - 1);
                in0_prev_core = CoreCoord(is_injector_core ? 0 : core_idx_x - 1, core_idx_y);
                in0_next_core = CoreCoord(is_sink_core ? core_idx_x : core_idx_x + 1, core_idx_y);
            } else {
                is_injector_core = core_idx_x == (num_cores_c - 1);
                is_sink_core     = core_idx_x == 0;
                in0_prev_core = CoreCoord(is_injector_core ? core_idx_x : core_idx_x + 1, core_idx_y);
                in0_next_core = CoreCoord(is_sink_core ? core_idx_x : core_idx_x - 1, core_idx_y);
            }

            auto in0_prev_core_physical = device->worker_core_from_logical_core(in0_prev_core);
            auto in0_next_core_physical = device->worker_core_from_logical_core(in0_next_core);

            in0_snf_reader_runtime_args.push_back((std::uint32_t)in0_next_core_physical.x);
            in0_snf_reader_runtime_args.push_back((std::uint32_t)in0_next_core_physical.y);
            in0_snf_reader_runtime_args.push_back((std::uint32_t)in0_prev_core_physical.x);
            in0_snf_reader_runtime_args.push_back((std::uint32_t)in0_prev_core_physical.y);

            in0_snf_reader_runtime_args.push_back(is_sink_core);

            // ── in1 CDA reader: column-wide schedule ──
            // all_num_iters_y[0..num_cores_r-1]
            for (uint32_t r = 0; r < num_cores_r; r++) {
                in1_reader_runtime_args.push_back((std::uint32_t)output_y_indices[r].size());
            }
            // all_y_coords flattened (for each core r, for each iy, resolved BSR row)
            for (uint32_t r = 0; r < num_cores_r; r++) {
                for (uint32_t iy = 0; iy < output_y_indices[r].size(); iy++) {
                    uint32_t resolved_row = folded_bsr_matrix_indices[output_y_indices[r][iy]];
                    in1_reader_runtime_args.push_back(resolved_row);
                }
            }

            // if constexpr (verbose) {
            //     log_info(tt::LogVerif, " -- Core ({},{}) CDA column-wide schedule --", core_idx_x, core_idx_y);
            //     log_info(tt::LogVerif, "   all_num_iters_y:");
            //     for (uint32_t r = 0; r < num_cores_r; r++) {
            //         log_info(tt::LogVerif, "     core_row[{}] num_iters_y = {}", r, output_y_indices[r].size());
            //     }
            //     log_info(tt::LogVerif, "   all_y_coords (flattened):");
            //     for (uint32_t r = 0; r < num_cores_r; r++) {
            //         for (uint32_t iy = 0; iy < output_y_indices[r].size(); iy++) {
            //             uint32_t resolved_row = folded_bsr_matrix_indices[output_y_indices[r][iy]];
            //             log_info(tt::LogVerif, "     core_row[{}] iy[{}] -> resolved BSR row = {}", r, iy, resolved_row);
            //         }
            //     }
            //     log_info(tt::LogVerif, "   total in1_reader_runtime_args size = {}", in1_reader_runtime_args.size());
            // }

            // Set runtime args for in0 reader
            if (is_injector_core){
                tt_metal::SetRuntimeArgs(program, in0_injector_and_writer_id, core, in0_snf_reader_runtime_args);
                if constexpr (verbose) {
                    log_info(tt::LogVerif, "Core x {} y {} injector", core_idx_x, core_idx_y);
                    log_info(tt::LogVerif, "sink? {}", is_sink_core);
                    log_info(tt::LogVerif, "num runtime args: {}", in0_snf_reader_runtime_args.size());
                    log_info(tt::LogVerif, "num comptime args: {}", in0_injector_compile_time_args.size());
                }
            }
            else {
                tt_metal::SetRuntimeArgs(program, in0_receiver_and_writer_id, core, in0_snf_reader_runtime_args);
                if constexpr (verbose)
                    log_info(tt::LogVerif, "Core x {} y {} receiver", core_idx_x, core_idx_y);
            }
            if (verbose && core_idx_x == 0 && core_idx_y == 1){
                log_info(tt::LogVerif, "in0 reader runtime args for core {} , {} :", core_idx_x, core_idx_y);
                for (size_t arg_idx = 0; arg_idx < in0_snf_reader_runtime_args.size(); arg_idx++){
                    log_info(tt::LogVerif, "arg {} : {}", arg_idx, in0_snf_reader_runtime_args[arg_idx]);
                }
                log_info(tt::LogVerif, "in1 CDA reader runtime args for core {} , {} :", core_idx_x, core_idx_y);
                for (size_t arg_idx = 0; arg_idx < in1_reader_runtime_args.size(); arg_idx++){
                    log_info(tt::LogVerif, "arg {} : {}", arg_idx, in1_reader_runtime_args[arg_idx]);
                }
            }
            tt_metal::SetRuntimeArgs(program, in1_reader_id, core, in1_reader_runtime_args);
            tt_metal::SetRuntimeArgs(program, compute_id, core, compute_runtime_args);
        }
    }

    // Pad indexing data to match tile-aligned DRAM buffer sizes
    std::vector<uint32_t> padded_col_indices(dram_buffer_col_indices_size / sizeof(uint32_t), 0);
    std::copy(a.indices.begin(), a.indices.end(), padded_col_indices.begin());

    std::vector<uint32_t> padded_indptr(dram_buffer_indptr_size / sizeof(uint32_t), 0);
    std::copy(a.indptr.begin(), a.indptr.end(), padded_indptr.begin());

    EnqueueWriteBuffer(cq, src0_dram_buffer, a.data.data(), false);
    EnqueueWriteBuffer(cq, src1_dram_buffer, b.data.data(), false);
    EnqueueWriteBuffer(cq, column_indices_dram_buffer, padded_col_indices.data(), false);
    EnqueueWriteBuffer(cq, indptr_dram_buffer, padded_indptr.data(), true);

    if constexpr (is_profiling){
        int num_iters = 10;
        EnqueueProgram(cq, program, true);
        ZoneScopedNC("Device program Loop", tracy::Color::Aquamarine);
        for (int i = 0; i < num_iters; i++){
            EnqueueProgram(cq, program, true);
        }
    }
    else if constexpr (verbose){
        log_info(tt::LogVerif, " -- Entering Program --");
        EnqueueProgram(cq, program, true);
    }
    else {
        EnqueueProgram(cq, program, false);
    }

    if constexpr (verbose)
        log_info(tt::LogVerif, " -- Program returned --");

    if constexpr (!is_profiling){
        uint32_t nonzero_row_index = 0;
        for (size_t row_index = 0; row_index < a.indptr.size() - 1; row_index++) {
            if (a.indptr[row_index+1] - a.indptr[row_index] == 0)
                continue;
            if constexpr (verbose) {
                log_info(tt::LogVerif, "Reading output row {} from device", row_index);
            }
            BufferRegion DRAM_row(nonzero_row_index * dram_buffer_dst_row_size, dram_buffer_dst_row_size);
            EnqueueReadSubBuffer(cq, dst_dram_buffer, output.data.data() + (row_index * R * N), DRAM_row, true);
            nonzero_row_index++;
        }
    }

    Finish(cq);
    if constexpr (verbose)
        log_info(tt::LogVerif, " -- Finished reading output --");
}

// Public thin wrapper (matches original API and HostCodeFunctionPtr)
template<bool verbose, bool is_profiling, bool use_optimal_noc,
         bool in0_left_to_right, bool in1_bottom_to_top>
void bsr_spmm_multicore_snfin0_cdain1(
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output,
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K,
    uint32_t R, uint32_t C, uint32_t B, IDevice* device) {
    bsr_spmm_multicore_snfin0_cdain1_impl<verbose, is_profiling, use_optimal_noc, in0_left_to_right, in1_bottom_to_top>(a, b, output, bcast_batch, nnz_blocks, M, N, K, R, C, B, device, {});
}

// Ablation skip wrappers
template<bool verbose, bool is_profiling, bool use_optimal_noc,
         bool in0_left_to_right, bool in1_bottom_to_top>
void bsr_spmm_multicore_snfin0_cdain1_no_a_read(
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output,
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K,
    uint32_t R, uint32_t C, uint32_t B, IDevice* device) {
    bsr_spmm_multicore_snfin0_cdain1_impl<verbose, is_profiling, use_optimal_noc, in0_left_to_right, in1_bottom_to_top>(a, b, output, bcast_batch, nnz_blocks, M, N, K, R, C, B, device, {{"SKIP_IN0_DRAM_READ", "1"}});
}
template<bool verbose, bool is_profiling, bool use_optimal_noc,
         bool in0_left_to_right, bool in1_bottom_to_top>
void bsr_spmm_multicore_snfin0_cdain1_no_b_read(
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output,
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K,
    uint32_t R, uint32_t C, uint32_t B, IDevice* device) {
    bsr_spmm_multicore_snfin0_cdain1_impl<verbose, is_profiling, use_optimal_noc, in0_left_to_right, in1_bottom_to_top>(a, b, output, bcast_batch, nnz_blocks, M, N, K, R, C, B, device, {{"SKIP_IN1_DRAM_READ", "1"}});
}
template<bool verbose, bool is_profiling, bool use_optimal_noc,
         bool in0_left_to_right, bool in1_bottom_to_top>
void bsr_spmm_multicore_snfin0_cdain1_no_compute(
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output,
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K,
    uint32_t R, uint32_t C, uint32_t B, IDevice* device) {
    bsr_spmm_multicore_snfin0_cdain1_impl<verbose, is_profiling, use_optimal_noc, in0_left_to_right, in1_bottom_to_top>(a, b, output, bcast_batch, nnz_blocks, M, N, K, R, C, B, device, {{"SKIP_COMPUTE", "1"}});
}
template<bool verbose, bool is_profiling, bool use_optimal_noc,
         bool in0_left_to_right, bool in1_bottom_to_top>
void bsr_spmm_multicore_snfin0_cdain1_no_write(
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output,
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K,
    uint32_t R, uint32_t C, uint32_t B, IDevice* device) {
    bsr_spmm_multicore_snfin0_cdain1_impl<verbose, is_profiling, use_optimal_noc, in0_left_to_right, in1_bottom_to_top>(a, b, output, bcast_batch, nnz_blocks, M, N, K, R, C, B, device, {{"SKIP_DRAM_WRITE", "1"}});
}

// No-load-balance wrapper
template<bool verbose, bool is_profiling, bool use_optimal_noc,
         bool in0_left_to_right, bool in1_bottom_to_top>
void bsr_spmm_multicore_snfin0_cdain1_no_lb(
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output,
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K,
    uint32_t R, uint32_t C, uint32_t B, IDevice* device) {
    bsr_spmm_multicore_snfin0_cdain1_impl<verbose, is_profiling, use_optimal_noc, in0_left_to_right, in1_bottom_to_top>(a, b, output, bcast_batch, nnz_blocks, M, N, K, R, C, B, device, {{"SKIP_LOAD_BALANCE", "1"}});
}

// Explicit template instantiations
template void bsr_spmm_multicore_snfin0_cdain1<false, false>(
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output,
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K,
    uint32_t R, uint32_t C, uint32_t B, IDevice* device);
template void bsr_spmm_multicore_snfin0_cdain1<true, false>(
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output,
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K,
    uint32_t R, uint32_t C, uint32_t B, IDevice* device);
template void bsr_spmm_multicore_snfin0_cdain1<false, true>(
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output,
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K,
    uint32_t R, uint32_t C, uint32_t B, IDevice* device);
// Ablation skip wrappers (profiling only)
template void bsr_spmm_multicore_snfin0_cdain1_no_a_read<false, true>(
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output,
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K,
    uint32_t R, uint32_t C, uint32_t B, IDevice* device);
template void bsr_spmm_multicore_snfin0_cdain1_no_b_read<false, true>(
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output,
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K,
    uint32_t R, uint32_t C, uint32_t B, IDevice* device);
template void bsr_spmm_multicore_snfin0_cdain1_no_compute<false, true>(
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output,
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K,
    uint32_t R, uint32_t C, uint32_t B, IDevice* device);
template void bsr_spmm_multicore_snfin0_cdain1_no_write<false, true>(
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output,
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K,
    uint32_t R, uint32_t C, uint32_t B, IDevice* device);
template void bsr_spmm_multicore_snfin0_cdain1_no_lb<false, true>(
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output,
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K,
    uint32_t R, uint32_t C, uint32_t B, IDevice* device);

}
