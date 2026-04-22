// SDDMM D Reader + Output Writer kernel with CDA (RISCV_1)
//
// Streams dense D blocks along the K reduction dimension using CDA
// (Chain of Direct Addressing) to share D reads across cores in the
// same column that need the same block_col_j. Then writes the output
// block A[i,j] back to DRAM.
//
// CDA direction: T2B (inject at max core_idx_y, chain toward min)
//   Injector = max core_idx_y in share set (reads D from DRAM)
//   Chain: max_y -> max_y-1 -> ... -> min_y
//   Barrier: per output slot, star topology across the column
//   Barrier timing: AFTER output writeback (not just after K-loop)

#include <stdint.h>
#include "dataflow_api.h"
#include "tt_metal/programming_examples/Tenstorrent-WH-BlockSpMM/block_spmm/kernels/common/spmm_tile_ops.hpp"

// Ablation skip flags (set to 1 via CreateKernel defines to skip that phase)
#ifndef SKIP_D_DRAM_READ
#define SKIP_D_DRAM_READ 0
#endif
#ifndef SKIP_DRAM_WRITE
#define SKIP_DRAM_WRITE 0
#endif

constexpr uint32_t SENTINEL = UINT32_MAX;

// CDA action codes
constexpr uint32_t CDA_SOLO      = 0;
constexpr uint32_t CDA_DRAM_READ = 1;
constexpr uint32_t CDA_RECEIVE   = 2;

void kernel_main() {
    // ── Compile-time args ────────────────────────────────────────────
    constexpr bool dense_d_is_dram             = get_compile_time_arg_val(0);
    constexpr bool out_is_dram                 = get_compile_time_arg_val(1);
    constexpr uint32_t dense_d_tensor_addr     = get_compile_time_arg_val(2);
    constexpr uint32_t dense_d_stride_w        = get_compile_time_arg_val(3);     // 1
    constexpr uint32_t dense_d_stride_h        = get_compile_time_arg_val(4);     // Nt
    constexpr uint32_t dense_d_block_h         = get_compile_time_arg_val(5);     // block_k
    constexpr uint32_t dense_d_block_w         = get_compile_time_arg_val(6);     // Ct
    constexpr uint32_t dense_d_block_num_tiles = get_compile_time_arg_val(7);     // block_k * Ct
    constexpr uint32_t out_tensor_addr         = get_compile_time_arg_val(8);
    constexpr uint32_t out_block_num_tiles     = get_compile_time_arg_val(9);     // Rt * Ct
    constexpr uint32_t out_subblock_w          = get_compile_time_arg_val(10);
    constexpr uint32_t out_subblock_h          = get_compile_time_arg_val(11);
    constexpr uint32_t num_blocks_k            = get_compile_time_arg_val(12);
    constexpr uint32_t Rt                      = get_compile_time_arg_val(13);
    constexpr uint32_t Ct                      = get_compile_time_arg_val(14);

    // CDA semaphore IDs
    uint32_t sender_semaphore_addr   = get_semaphore(get_compile_time_arg_val(15));
    uint32_t receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(16));
    uint32_t barrier_semaphore_addr  = get_semaphore(get_compile_time_arg_val(17));
    uint32_t release_semaphore_addr  = get_semaphore(get_compile_time_arg_val(18));

    // CB IDs
    constexpr uint32_t cb_dense_d = 2;   // D reduction blocks (double-buffered)
    constexpr uint32_t cb_out     = 16;  // Output block

    // Tile sizes and formats
    constexpr uint32_t dense_d_tile_size = get_tile_size(cb_dense_d);
    constexpr uint32_t out_tile_size     = get_tile_size(cb_out);
    constexpr auto dense_d_df = get_dataformat(cb_dense_d);
    constexpr auto out_df     = get_dataformat(cb_out);

    // Address generators
    const InterleavedAddrGenFast<dense_d_is_dram> dense_d_addr_gen = {
        .bank_base_address = dense_d_tensor_addr,
        .page_size = dense_d_tile_size,
        .data_format = dense_d_df};
    const InterleavedAddrGenFast<out_is_dram> out_addr_gen = {
        .bank_base_address = out_tensor_addr,
        .page_size = out_tile_size,
        .data_format = out_df};

    // Subblock iteration counts
    constexpr uint32_t out_num_subblocks_h = Rt / out_subblock_h;
    constexpr uint32_t out_num_subblocks_w = Ct / out_subblock_w;

    // ── Runtime args ─────────────────────────────────────────────────
    uint32_t arg_idx = 0;
    const uint32_t max_output_blocks    = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t my_core_idx_y        = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_cores_in_column  = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t noc_x_for_column     = get_arg_val<uint32_t>(arg_idx++);

    // NOC y coordinates for all cores in this column
    uint32_t noc_y_table[num_cores_in_column];
    for (uint32_t r = 0; r < num_cores_in_column; r++) {
        noc_y_table[r] = get_arg_val<uint32_t>(arg_idx++);
    }

    // This core's per-slot data indices (SENTINEL for no-work slots)
    uint32_t this_data_idx[max_output_blocks];
    for (uint32_t s = 0; s < max_output_blocks; s++) {
        this_data_idx[s] = get_arg_val<uint32_t>(arg_idx++);
    }

    // All cores' block_col_j values for share set discovery
    // Layout: row-major [core_idx_y][slot], SENTINEL for no-work
    uint32_t total_schedule = num_cores_in_column * max_output_blocks;
    uint32_t all_block_cols[total_schedule];
    for (uint32_t i = 0; i < total_schedule; i++) {
        all_block_cols[i] = get_arg_val<uint32_t>(arg_idx++);
    }

    // ── Semaphore initialization ─────────────────────────────────────
    volatile tt_l1_ptr uint32_t* sender_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_semaphore_addr);
    *(sender_sem_ptr) = 0;
    volatile tt_l1_ptr uint32_t* receiver_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_semaphore_addr);
    *(receiver_sem_ptr) = 0;
    volatile tt_l1_ptr uint32_t* barrier_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_semaphore_addr);
    *(barrier_sem_ptr) = 0;
    volatile tt_l1_ptr uint32_t* release_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(release_semaphore_addr);
    *(release_sem_ptr) = 0;

    // ── CB slot addresses for CDA double-buffering ───────────────────
    uint32_t dense_d_cb_base = get_write_ptr(cb_dense_d);
    uint32_t dense_d_single_buf_size = dense_d_tile_size * dense_d_block_num_tiles;
    uint32_t block_bytes = dense_d_tile_size * dense_d_block_num_tiles;

    // ── Main loop: iterate in lockstep with all cores in column ──────
    for (uint32_t s = 0; s < max_output_blocks; s++) {
        // Look up this core's block_col_j from the all-cores schedule
        uint32_t my_block_col_j = all_block_cols[my_core_idx_y * max_output_blocks + s];
        bool has_work = (my_block_col_j != SENTINEL);

        if (has_work) {
            uint32_t blk_data_idx = this_data_idx[s];

#if SKIP_D_DRAM_READ == 0
            // ── Share set discovery (T2B along core column) ──────────
            // Find cores in this column with the same block_col_j at this slot.
            // Multiple independent chains can coexist in the same column.
            uint32_t share_set_size = 0;
            uint32_t injector_idx = my_core_idx_y;  // guaranteed in set
            bool found_sender = false, found_downstream = false;
            uint32_t my_sender_idx = 0, my_downstream_idx = 0;

            for (uint32_t r = 0; r < num_cores_in_column; r++) {
                uint32_t their_col = all_block_cols[r * max_output_blocks + s];
                if (their_col == SENTINEL) continue;
                if (their_col == my_block_col_j) {
                    share_set_size++;
                    // Injector = max core_idx_y in share set
                    if (r > injector_idx) injector_idx = r;
                    // Sender = min y > my y (closest core toward injector, sends TO me)
                    if (r > my_core_idx_y) {
                        if (!found_sender || r < my_sender_idx) {
                            my_sender_idx = r;
                            found_sender = true;
                        }
                    }
                    // Downstream = max y < my y (closest core away from injector, I send TO)
                    if (r < my_core_idx_y) {
                        if (!found_downstream || r > my_downstream_idx) {
                            my_downstream_idx = r;
                            found_downstream = true;
                        }
                    }
                }
            }

            // ── Determine action ─────────────────────────────────────
            uint32_t action;
            if (share_set_size <= 1) {
                action = CDA_SOLO;
            } else if (my_core_idx_y == injector_idx) {
                action = CDA_DRAM_READ;
            } else {
                action = CDA_RECEIVE;
            }

            // ── K-loop: all K steps use the SAME share set ──────────
            for (uint32_t k = 0; k < num_blocks_k; k++) {
                cb_reserve_back(cb_dense_d, dense_d_block_num_tiles);
                uint32_t l1_addr_d = get_write_ptr(cb_dense_d);
                uint32_t l1_addr_d_start = l1_addr_d;  // Save for forwarding

                if (action == CDA_SOLO || action == CDA_DRAM_READ) {
                    // DRAM read: D block at K step k, column block_col_j
                    uint32_t d_start_tile = k * dense_d_block_h * dense_d_stride_h
                                          + my_block_col_j * dense_d_block_w;
                    spmm::read_block_by_tile(
                        d_start_tile, dense_d_addr_gen, l1_addr_d,
                        dense_d_tile_size, dense_d_block_h, dense_d_block_w,
                        dense_d_stride_h, dense_d_stride_w);
                    noc_async_read_barrier();
                } else {
                    // RECEIVE: wait for sender to write data to our CB
                    noc_semaphore_set(receiver_sem_ptr, 0);
                    uint64_t sender_sem_noc = get_noc_addr(
                        noc_x_for_column, noc_y_table[my_sender_idx], sender_semaphore_addr);
                    uint32_t my_slot_bit = (l1_addr_d_start != dense_d_cb_base) ? 1 : 0;
                    noc_semaphore_inc(sender_sem_noc, 1 + my_slot_bit);
                    noc_semaphore_wait(receiver_sem_ptr, 1);
                }

                cb_push_back(cb_dense_d, dense_d_block_num_tiles);

                // Forward to downstream if applicable
                if (found_downstream && action != CDA_SOLO) {
                    while (*sender_sem_ptr == 0) {}
                    uint32_t receiver_slot_bit = *sender_sem_ptr - 1;
                    noc_semaphore_set(sender_sem_ptr, 0);

                    uint32_t receiver_dest_addr = dense_d_cb_base + receiver_slot_bit * dense_d_single_buf_size;
                    uint64_t dest_data_addr = get_noc_addr(
                        noc_x_for_column, noc_y_table[my_downstream_idx], receiver_dest_addr);
                    noc_async_write(l1_addr_d_start, dest_data_addr, block_bytes);
                    noc_async_write_barrier();

                    uint64_t dest_recv_sem = get_noc_addr(
                        noc_x_for_column, noc_y_table[my_downstream_idx], receiver_semaphore_addr);
                    noc_semaphore_inc(dest_recv_sem, 1);
                    noc_async_atomic_barrier();
                }
            } // end k loop
#else
            // Skip D reads: just push dummy blocks to maintain CB protocol
            for (uint32_t k = 0; k < num_blocks_k; k++) {
                cb_reserve_back(cb_dense_d, dense_d_block_num_tiles);
                cb_push_back(cb_dense_d, dense_d_block_num_tiles);
            }
#endif

            // ── Write output block to DRAM ───────────────────────────
            cb_wait_front(cb_out, out_block_num_tiles);
#if SKIP_DRAM_WRITE == 0
            uint32_t l1_read_addr = get_read_ptr(cb_out);
            uint32_t out_start_tile = blk_data_idx * out_block_num_tiles;
            uint32_t out_sbh_start = out_start_tile;
            for (uint32_t sbh = 0; sbh < out_num_subblocks_h; sbh++) {
                uint32_t out_sbw_start = out_sbh_start;
                for (uint32_t sbw = 0; sbw < out_num_subblocks_w; sbw++) {
                    spmm::write_subblock_by_tile(
                        out_sbw_start, out_addr_gen, l1_read_addr,
                        out_tile_size, out_subblock_h, out_subblock_w,
                        Ct, 1);  // stride_h = Ct within block, stride_w = 1
                    out_sbw_start += out_subblock_w;
                }
                out_sbh_start += out_subblock_h * Ct;
            }
            noc_async_write_barrier();
#endif
            cb_pop_front(cb_out, out_block_num_tiles);
        } // end has_work

        // ── Column-wide barrier (star topology, AFTER writeback) ─────
        // ALL cores in the column participate at every slot, even those
        // without work.  This prevents idle cores from racing ahead to
        // slot s+1 and signalling sender_sem while the injector is still
        // processing slot s — which would corrupt the CDA handshake.
        if (num_cores_in_column > 1) {
            constexpr uint32_t barrier_leader_idx = 0;
            bool is_barrier_leader = (my_core_idx_y == barrier_leader_idx);

            uint64_t leader_barrier_noc = get_noc_addr(
                noc_x_for_column, noc_y_table[barrier_leader_idx], barrier_semaphore_addr);

            if (is_barrier_leader) {
                noc_semaphore_wait(barrier_sem_ptr, num_cores_in_column - 1);
                noc_semaphore_set(barrier_sem_ptr, 0);
                for (uint32_t r = 1; r < num_cores_in_column; r++) {
                    uint64_t their_release = get_noc_addr(
                        noc_x_for_column, noc_y_table[r], release_semaphore_addr);
                    noc_semaphore_inc(their_release, 1);
                }
                noc_async_atomic_barrier();
            } else {
                noc_semaphore_inc(leader_barrier_noc, 1);
                noc_async_atomic_barrier();
                noc_semaphore_wait(release_sem_ptr, 1);
                noc_semaphore_set(release_sem_ptr, 0);
            }
        }
    } // end slot loop
}
