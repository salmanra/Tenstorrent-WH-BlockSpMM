// SDDMM BC Reader kernel (RISCV_0)
//
// Reads the sparse mask block B[i,j] and streams dense C blocks
// along the K reduction dimension for each assigned output block.

#include <stdint.h>
#include "dataflow_api.h"
#include "tt_metal/programming_examples/rahmy/SC26_submission/block_spmm/kernels/common/spmm_tile_ops.hpp"

TODO: this will not work. What data sharing scheme will? 

void kernel_main() {
    // ── Compile-time args ────────────────────────────────────────────
    constexpr bool sparse_is_dram          = get_compile_time_arg_val(0);
    constexpr bool dense_c_is_dram         = get_compile_time_arg_val(1);
    constexpr uint32_t sparse_tensor_addr  = get_compile_time_arg_val(2);
    constexpr uint32_t sparse_block_num_tiles = get_compile_time_arg_val(3);  // Rt*Ct
    constexpr uint32_t dense_c_tensor_addr = get_compile_time_arg_val(4);
    constexpr uint32_t dense_c_stride_w    = get_compile_time_arg_val(5);     // 1
    constexpr uint32_t dense_c_stride_h    = get_compile_time_arg_val(6);     // Kt
    constexpr uint32_t dense_c_block_h     = get_compile_time_arg_val(7);     // Rt
    constexpr uint32_t dense_c_block_w     = get_compile_time_arg_val(8);     // block_k
    constexpr uint32_t dense_c_block_num_tiles = get_compile_time_arg_val(9); // Rt * block_k
    constexpr uint32_t num_blocks_k        = get_compile_time_arg_val(10);

    constexpr uint32_t is_injector_core = get_compile_time_arg_val(11);


    // CB IDs (from sddmm_reader_common.hpp)
    constexpr uint32_t cb_sparse = 0;   // B mask block
    constexpr uint32_t cb_dense_c = 1;  // C reduction blocks

    // Tile size
    constexpr uint32_t sparse_tile_size = get_tile_size(cb_sparse);
    constexpr uint32_t dense_c_tile_size = get_tile_size(cb_dense_c);
    constexpr auto sparse_df = get_dataformat(cb_sparse);
    constexpr auto dense_c_df = get_dataformat(cb_dense_c);

    // Address generators
    const InterleavedAddrGenFast<sparse_is_dram> sparse_addr_gen = {
        .bank_base_address = sparse_tensor_addr,
        .page_size = sparse_tile_size,
        .data_format = sparse_df};

    const InterleavedAddrGenFast<dense_c_is_dram> dense_c_addr_gen = {
        .bank_base_address = dense_c_tensor_addr,
        .page_size = dense_c_tile_size,
        .data_format = dense_c_df};

    // ── Runtime args ─────────────────────────────────────────────────
    uint32_t arg_idx = 0;
    const uint32_t num_output_blocks = get_arg_val<uint32_t>(arg_idx++);
    uint32_t block_row_indices[num_output_blocks];
    uint32_t block_data_indices[num_output_blocks];
    for (uint32_t ob = 0; ob < num_output_blocks; ob++){
        block_row_indices[ob] = get_arg_val<uint32_t>(arg_idx++);
        block_data_indices[ob] = get_arg_val<uint32_t>(arg_idx++);
    }

    // ── SnF Runtime args ─────────────────────────────────────────────
    const uint32_t inC_dest_noc_x = get_arg_val<uint32_t>(arg_index++);
    const uint32_t inC_dest_noc_y = get_arg_val<uint32_t>(arg_index++);
    const uint32_t inC_sender_noc_x = get_arg_val<uint32_t>(arg_index++);
    const uint32_t inC_sender_noc_y = get_arg_val<uint32_t>(arg_index++);
    const uint32_t is_sink_core = get_arg_val<uint32_t>(arg_index++);

    // SnF semaphores
    volatile tt_l1_ptr uint32_t* inC_sender_semaphore_addr_ptr =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(inC_sender_semaphore_addr);
    *(inC_sender_semaphore_addr_ptr) = 0;
    const uint64_t inC_sender_semaphore_noc_addr =
    get_noc_addr(inC_sender_noc_x, inC_sender_noc_y, inC_sender_semaphore_addr);

    volatile tt_l1_ptr uint32_t* inC_receiver_semaphore_addr_ptr =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(inC_receiver_semaphore_addr);
    *(inC_receiver_semaphore_addr_ptr) = 0;
    const uint64_t inC_receiver_semaphore_noc_addr =
    get_noc_addr(inC_dest_noc_x, inC_dest_noc_y, inC_receiver_semaphore_addr);

    // CB IDs (from sddmm_reader_common.hpp)
    constexpr uint32_t cb_sparse = 0;   // B mask block
    constexpr uint32_t cb_dense_c = 1;  // C reduction blocks

    // Tile size
    constexpr uint32_t sparse_tile_size = get_tile_size(cb_sparse);
    constexpr uint32_t dense_c_tile_size = get_tile_size(cb_dense_c);
    constexpr auto sparse_df = get_dataformat(cb_sparse);
    constexpr auto dense_c_df = get_dataformat(cb_dense_c);

    // Address generators
    const InterleavedAddrGenFast<sparse_is_dram> sparse_addr_gen = {
        .bank_base_address = sparse_tensor_addr,
        .page_size = sparse_tile_size,
        .data_format = sparse_df};

    const InterleavedAddrGenFast<dense_c_is_dram> dense_c_addr_gen = {
        .bank_base_address = dense_c_tensor_addr,
        .page_size = dense_c_tile_size,
        .data_format = dense_c_df};

    // ── Main loop ────────────────────────────────────────────────────
    for (uint32_t ob = 0; ob < num_output_blocks; ob++) {
        // TODO: each output block defines its own SnF multicast group
        uint32_t block_row_i = block_row_indices[ob];
        uint32_t blk_data_idx = block_data_indices[ob];

        // ── Read sparse mask block B[i,j] into c_0 ──────────────────
        // B data is stored as contiguous blocks: block blk_data_idx
        // starts at tile blk_data_idx * sparse_block_num_tiles.
        // Within a block: Rt rows of Ct tiles, stride_h=Ct, stride_w=1.
        cb_reserve_back(cb_sparse, sparse_block_num_tiles);
        uint32_t l1_write_addr = get_write_ptr(cb_sparse);
        uint32_t sparse_start_tile = blk_data_idx * sparse_block_num_tiles;
        // Sparse block is contiguous: Rt*Ct tiles laid out sequentially
        for (uint32_t t = 0; t < sparse_block_num_tiles; t++) {
            noc_async_read_tile(sparse_start_tile + t, sparse_addr_gen, l1_write_addr);
            l1_write_addr += sparse_tile_size;
        }
        noc_async_read_barrier();
        cb_push_back(cb_sparse, sparse_block_num_tiles);

        // ── Stream C blocks for reduction ────────────────────────────
        // TODO: Discover what my role is, then do CDA, then do a barrier at the end of the output_block iter instead of the reduction step
        //
        /*
        share set size == indptr[my index this iter + 1] - indptr[my index this iter]
        -> my share set is all the cores IN MY CORE ROW who are assigned this B matrix row
        -> R2L: My injector core is the max over core_idx_x in this share set
        -> My sender is the min over core_idx_x in the core set which are greater than my core_idx_x
        -> My receiver is the max over core_idx_x in the core set which are less than my core_idx_x
        Which data structures does the host have to send over for the cores to know this data?
            For each core in this grid row, for each output block each core is processing, 
        */
        for (uint32_t k = 0; k < num_blocks_k; k++) {
            // TODO: replace this block with CDA-like code for each k
            cb_reserve_back(cb_dense_c, dense_c_block_num_tiles);
            uint32_t l1_addr_c = get_write_ptr(cb_dense_c);
            uint32_t l1_write_addr_c_start = l1_addr_c;  // Save start address for forwarding

            if constexpr(is_injector){
                // C block for reduction step k:
                // Rows: [block_row_i * Rt .. (block_row_i+1) * Rt)
                // Cols: [k * block_k .. (k+1) * block_k)
                // Start tile = block_row_i * Rt * Kt + k * block_k
                uint32_t c_start_tile = block_row_i * dense_c_block_h * dense_c_stride_h + k * dense_c_block_w;
                spmm::read_block_by_tile(
                    c_start_tile, dense_c_addr_gen, l1_addr_c,
                    dense_c_tile_size, dense_c_block_h, dense_c_block_w,
                    dense_c_stride_h, dense_c_stride_w);
                noc_async_read_barrier();
            }
            else {
                // semaphore shenanigans
                // Read inC block from neighbor
                noc_semaphore_set(inC_receiver_semaphore_addr_ptr, 0);
                noc_semaphore_inc(inC_sender_semaphore_noc_addr, 1);
                noc_semaphore_wait(inC_receiver_semaphore_addr_ptr, 1);
            }
            cb_push_back(cb_dense_c, dense_c_block_num_tiles);

            if constexpr (!is_sink_core) {
                // semaphore shenanigans
                // noc async write shenagnigans
                noc_semaphore_wait(inC_sender_semaphore_addr_ptr, 1);
                noc_semaphore_set(inC_sender_semaphore_addr_ptr, 0);

                uint64_t inC_unicast_data_addr = get_noc_addr(inC_dest_noc_x, inC_dest_noc_y, l1_write_addr_c_start);
                noc_async_write(l1_write_addr_c_start, inC_unicast_data_addr, current_block_bytes);
                noc_async_write_barrier();
                noc_semaphore_inc(inC_receiver_semaphore_noc_addr, 1);
            }
        }
        // TODO: barrier with my entire row of cores (not just the share set!), leader is max core_idx_x
    }
}
