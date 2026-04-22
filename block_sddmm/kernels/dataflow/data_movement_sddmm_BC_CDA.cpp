// SDDMM BC Reader kernel with CDA (RISCV_0)
//
// Reads the sparse mask block B[i,j] (always from DRAM, no sharing) and
// streams dense C blocks along the K reduction dimension using CDA
// (Chain of Direct Addressing) to share C reads across cores in the
// same row that need the same block_row_i.
//
// CDA direction: R2L (Right-to-Left along core rows)
//   Injector = max core_idx_x in share set (reads C from DRAM)
//   Chain: max_x -> max_x-1 -> ... -> min_x
//   Barrier: per output slot, star topology across the row

#include <stdint.h>
#include "dataflow_api.h"
#include "tt_metal/programming_examples/rahmy/SC26_submission/block_spmm/kernels/common/spmm_tile_ops.hpp"

// Ablation skip flags (set to 1 via CreateKernel defines to skip that phase)
#ifndef SKIP_SPARSE_DRAM_READ
#define SKIP_SPARSE_DRAM_READ 0
#endif
#ifndef SKIP_C_DRAM_READ
#define SKIP_C_DRAM_READ 0
#endif

constexpr uint32_t SENTINEL = UINT32_MAX;

// CDA action codes
constexpr uint32_t CDA_SOLO      = 0;  // Only this core needs this data
constexpr uint32_t CDA_DRAM_READ = 1;  // I'm the injector, read from DRAM
constexpr uint32_t CDA_RECEIVE   = 2;  // Wait for data from sender

void kernel_main() {
    // ── Compile-time args ────────────────────────────────────────────
    constexpr bool sparse_is_dram              = get_compile_time_arg_val(0);
    constexpr bool dense_c_is_dram             = get_compile_time_arg_val(1);
    constexpr uint32_t sparse_tensor_addr      = get_compile_time_arg_val(2);
    constexpr uint32_t sparse_block_num_tiles  = get_compile_time_arg_val(3);  // Rt*Ct
    constexpr uint32_t dense_c_tensor_addr     = get_compile_time_arg_val(4);
    constexpr uint32_t dense_c_stride_w        = get_compile_time_arg_val(5);  // 1
    constexpr uint32_t dense_c_stride_h        = get_compile_time_arg_val(6);  // Kt
    constexpr uint32_t dense_c_block_h         = get_compile_time_arg_val(7);  // Rt
    constexpr uint32_t dense_c_block_w         = get_compile_time_arg_val(8);  // block_k
    constexpr uint32_t dense_c_block_num_tiles = get_compile_time_arg_val(9);  // Rt * block_k
    constexpr uint32_t num_blocks_k            = get_compile_time_arg_val(10);

    // CDA semaphore IDs
    uint32_t sender_semaphore_addr   = get_semaphore(get_compile_time_arg_val(11));
    uint32_t receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(12));
    uint32_t barrier_semaphore_addr  = get_semaphore(get_compile_time_arg_val(13));
    uint32_t release_semaphore_addr  = get_semaphore(get_compile_time_arg_val(14));

    // CB IDs
    constexpr uint32_t cb_sparse  = 0;   // B mask block
    constexpr uint32_t cb_dense_c = 1;   // C reduction blocks (double-buffered)

    // Tile sizes and formats
    constexpr uint32_t sparse_tile_size  = get_tile_size(cb_sparse);
    constexpr uint32_t dense_c_tile_size = get_tile_size(cb_dense_c);
    constexpr auto sparse_df  = get_dataformat(cb_sparse);
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
    const uint32_t max_output_blocks = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t my_core_idx_x     = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_cores_in_row  = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t noc_y_for_row     = get_arg_val<uint32_t>(arg_idx++);

    // NOC x coordinates for all cores in this row
    uint32_t noc_x_table[num_cores_in_row];
    for (uint32_t c = 0; c < num_cores_in_row; c++) {
        noc_x_table[c] = get_arg_val<uint32_t>(arg_idx++);
    }

    // This core's per-slot data indices (SENTINEL for no-work slots)
    uint32_t this_data_idx[max_output_blocks];
    for (uint32_t s = 0; s < max_output_blocks; s++) {
        this_data_idx[s] = get_arg_val<uint32_t>(arg_idx++);
    }

    // All cores' block_row_i values for share set discovery
    // Layout: row-major [core_idx_x][slot], SENTINEL for no-work
    uint32_t total_schedule = num_cores_in_row * max_output_blocks;
    uint32_t all_block_rows[total_schedule];
    for (uint32_t i = 0; i < total_schedule; i++) {
        all_block_rows[i] = get_arg_val<uint32_t>(arg_idx++);
    }

    // ── Semaphore initialization ─────────────────────────────────────
    // MUST happen before any core can signal us (race with fast neighbors)
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
    // Dense C CB is double-buffered: slot 0 at base, slot 1 at base + single_buf_size.
    // Receivers encode their slot in the readiness signal so senders
    // write to the correct address on the receiver's L1.
    uint32_t dense_c_cb_base = get_write_ptr(cb_dense_c);
    uint32_t dense_c_single_buf_size = dense_c_tile_size * dense_c_block_num_tiles;
    uint32_t block_bytes = dense_c_tile_size * dense_c_block_num_tiles;

    // ── Main loop: iterate in lockstep with all cores in row ─────────
    for (uint32_t s = 0; s < max_output_blocks; s++) {
        // Look up this core's block_row_i from the all-cores schedule
        uint32_t my_block_row_i = all_block_rows[my_core_idx_x * max_output_blocks + s];
        bool has_work = (my_block_row_i != SENTINEL);

        if (has_work) {
            uint32_t blk_data_idx = this_data_idx[s];

            // ── Read sparse mask block B[i,j] (always from DRAM, no sharing)
            cb_reserve_back(cb_sparse, sparse_block_num_tiles);
#if SKIP_SPARSE_DRAM_READ == 0
            uint32_t l1_write_addr = get_write_ptr(cb_sparse);
            uint32_t sparse_start_tile = blk_data_idx * sparse_block_num_tiles;
            for (uint32_t t = 0; t < sparse_block_num_tiles; t++) {
                noc_async_read_tile(sparse_start_tile + t, sparse_addr_gen, l1_write_addr);
                l1_write_addr += sparse_tile_size;
            }
            noc_async_read_barrier();
#endif
            cb_push_back(cb_sparse, sparse_block_num_tiles);

#if SKIP_C_DRAM_READ == 0
            // ── Share set discovery (R2L along core row) ─────────────
            // Find which other cores in this row need the same block_row_i at this slot.
            // CDA's point-to-point chain topology allows multiple independent share sets
            // to operate simultaneously within the same row at the same slot.
            uint32_t share_set_size = 0;
            uint32_t injector_idx = my_core_idx_x;  // guaranteed in set
            bool found_sender = false, found_downstream = false;
            uint32_t my_sender_idx = 0, my_downstream_idx = 0;

            for (uint32_t c = 0; c < num_cores_in_row; c++) {
                uint32_t their_row = all_block_rows[c * max_output_blocks + s];
                if (their_row == SENTINEL) continue;
                if (their_row == my_block_row_i) {
                    share_set_size++;
                    // Injector = max core_idx_x in share set
                    if (c > injector_idx) injector_idx = c;
                    // Sender = min x > my x (closest core toward injector, sends TO me)
                    if (c > my_core_idx_x) {
                        if (!found_sender || c < my_sender_idx) {
                            my_sender_idx = c;
                            found_sender = true;
                        }
                    }
                    // Downstream = max x < my x (closest core away from injector, I send TO)
                    if (c < my_core_idx_x) {
                        if (!found_downstream || c > my_downstream_idx) {
                            my_downstream_idx = c;
                            found_downstream = true;
                        }
                    }
                }
            }

            // ── Determine action ─────────────────────────────────────
            uint32_t action;
            if (share_set_size <= 1) {
                action = CDA_SOLO;
            } else if (my_core_idx_x == injector_idx) {
                action = CDA_DRAM_READ;
            } else {
                action = CDA_RECEIVE;
            }

            // ── K-loop: all K steps use the SAME share set ──────────
            // The share set is fixed for a given output block because
            // block_row_i doesn't change across K steps. This is the key
            // simplification vs SpMM CDA where share sets change per vstep.
            for (uint32_t k = 0; k < num_blocks_k; k++) {
                cb_reserve_back(cb_dense_c, dense_c_block_num_tiles);
                uint32_t l1_addr_c = get_write_ptr(cb_dense_c);
                uint32_t l1_addr_c_start = l1_addr_c;  // Save for forwarding

                if (action == CDA_SOLO || action == CDA_DRAM_READ) {
                    // DRAM read: C block at row block_row_i, K step k
                    uint32_t c_start_tile = my_block_row_i * dense_c_block_h * dense_c_stride_h
                                          + k * dense_c_block_w;
                    spmm::read_block_by_tile(
                        c_start_tile, dense_c_addr_gen, l1_addr_c,
                        dense_c_tile_size, dense_c_block_h, dense_c_block_w,
                        dense_c_stride_h, dense_c_stride_w);
                    noc_async_read_barrier();
                } else {
                    // RECEIVE: wait for sender to write data to our CB
                    noc_semaphore_set(receiver_sem_ptr, 0);
                    uint64_t sender_sem_noc = get_noc_addr(
                        noc_x_table[my_sender_idx], noc_y_for_row, sender_semaphore_addr);
                    // Encode which CB slot we want data written to
                    uint32_t my_slot_bit = (l1_addr_c_start != dense_c_cb_base) ? 1 : 0;
                    noc_semaphore_inc(sender_sem_noc, 1 + my_slot_bit);
                    noc_semaphore_wait(receiver_sem_ptr, 1);
                }

                cb_push_back(cb_dense_c, dense_c_block_num_tiles);

                // Forward to downstream if applicable
                if (found_downstream && action != CDA_SOLO) {
                    // Wait for downstream to signal readiness (value encodes CB slot bit)
                    while (*sender_sem_ptr == 0) {}
                    uint32_t receiver_slot_bit = *sender_sem_ptr - 1;
                    noc_semaphore_set(sender_sem_ptr, 0);

                    // Write data to downstream's CB at the slot it requested
                    uint32_t receiver_dest_addr = dense_c_cb_base + receiver_slot_bit * dense_c_single_buf_size;
                    uint64_t dest_data_addr = get_noc_addr(
                        noc_x_table[my_downstream_idx], noc_y_for_row, receiver_dest_addr);
                    noc_async_write(l1_addr_c_start, dest_data_addr, block_bytes);
                    noc_async_write_barrier();

                    // Signal downstream that data has arrived
                    uint64_t dest_recv_sem = get_noc_addr(
                        noc_x_table[my_downstream_idx], noc_y_for_row, receiver_semaphore_addr);
                    noc_semaphore_inc(dest_recv_sem, 1);
                    noc_async_atomic_barrier();
                }
            } // end k loop
#else
            // Skip C reads: just push dummy blocks to maintain CB protocol
            for (uint32_t k = 0; k < num_blocks_k; k++) {
                cb_reserve_back(cb_dense_c, dense_c_block_num_tiles);
                cb_push_back(cb_dense_c, dense_c_block_num_tiles);
            }
#endif
        } // end has_work

        // ── Row-wide barrier (star topology) ─────────────────────────
        // ALL cores in the row participate at every slot, even those
        // without work.  This prevents idle cores from racing ahead to
        // slot s+1 and signalling sender_sem while the injector is still
        // processing slot s — which would corrupt the CDA handshake.
        if (num_cores_in_row > 1) {
            constexpr uint32_t barrier_leader_idx = 0;
            bool is_barrier_leader = (my_core_idx_x == barrier_leader_idx);

            uint64_t leader_barrier_noc = get_noc_addr(
                noc_x_table[barrier_leader_idx], noc_y_for_row, barrier_semaphore_addr);

            if (is_barrier_leader) {
                noc_semaphore_wait(barrier_sem_ptr, num_cores_in_row - 1);
                noc_semaphore_set(barrier_sem_ptr, 0);
                for (uint32_t c = 1; c < num_cores_in_row; c++) {
                    uint64_t their_release = get_noc_addr(
                        noc_x_table[c], noc_y_for_row, release_semaphore_addr);
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
