// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/kernel_structs.h"

// in1 reader for CDA: chains dense B tiles along a column of cores.
// Each core row within a column processes a different set of row-blocks. For a fixed
// N-tile and K-step, cores that need the same B tile (same pair index) form a share
// set; the highest-indexed core in the set reads from DRAM and the others receive
// from the next core above them in the set.
void kernel_main() {
    constexpr std::uint32_t b_is_dram = get_compile_time_arg_val(0);
    constexpr std::uint32_t max_kt_dim = get_compile_time_arg_val(1);
    constexpr std::uint32_t num_output_cols = get_compile_time_arg_val(2);
    constexpr std::uint32_t num_cores_y = get_compile_time_arg_val(3);

    constexpr std::uint32_t cb_id_b = tt::CBIndex::c_1;

    constexpr std::uint32_t tile_size_b = 512;  // 16 * 16 * 2 bytes
    constexpr std::uint32_t tile_hw_b = 256;    // 16 * 16 datums
    constexpr std::uint32_t INVALID_CORE = 0xFFFFFFFFU;

    const std::uint32_t b_addr = get_arg_val<std::uint32_t>(0);
    const std::uint32_t core_x = get_arg_val<std::uint32_t>(1);
    const std::uint32_t core_y = get_arg_val<std::uint32_t>(2);
    const std::uint32_t num_iters_x = get_arg_val<std::uint32_t>(3);
    const std::uint32_t num_iters_y = get_arg_val<std::uint32_t>(4);
    const std::uint32_t output_idx_x_start = get_arg_val<std::uint32_t>(5);
    const std::uint32_t sem_sender = get_semaphore(get_arg_val<std::uint32_t>(6));
    const std::uint32_t sem_receiver = get_semaphore(get_arg_val<std::uint32_t>(7));
    const std::uint32_t sem_barrier = get_semaphore(get_arg_val<std::uint32_t>(8));
    const std::uint32_t sem_release = get_semaphore(get_arg_val<std::uint32_t>(9));
    const std::uint32_t column_noc_x = get_arg_val<std::uint32_t>(10);

    std::uint32_t arg_idx = 11;
    std::uint32_t noc_y_table[num_cores_y];
    for (std::uint32_t r = 0; r < num_cores_y; ++r) {
        noc_y_table[r] = get_arg_val<std::uint32_t>(arg_idx++);
    }

    std::uint32_t all_num_iters_y[num_cores_y];
    std::uint32_t all_pair_offsets[num_cores_y];
    std::uint32_t total_pair_entries = 0;
    for (std::uint32_t r = 0; r < num_cores_y; ++r) {
        all_num_iters_y[r] = get_arg_val<std::uint32_t>(arg_idx++);
        all_pair_offsets[r] = total_pair_entries;
        total_pair_entries += all_num_iters_y[r] * max_kt_dim;
    }

    // Pair indices for all participating (core_row, iter_y, k) in this column.
    // Stored as uint16 to stay within the RISC-V stack budget.
    std::uint16_t all_pair_indices[total_pair_entries];
    for (std::uint32_t i = 0; i < total_pair_entries; ++i) {
        all_pair_indices[i] = static_cast<std::uint16_t>(get_arg_val<std::uint32_t>(arg_idx++));
    }

    volatile tt_l1_ptr std::uint32_t* sem_sender_ptr = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(sem_sender);
    volatile tt_l1_ptr std::uint32_t* sem_receiver_ptr =
        reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(sem_receiver);
    volatile tt_l1_ptr std::uint32_t* sem_barrier_ptr =
        reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(sem_barrier);
    volatile tt_l1_ptr std::uint32_t* sem_release_ptr =
        reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(sem_release);

    const auto b_reader = InterleavedAddrGenFast<b_is_dram, tile_hw_b>{b_addr, tile_size_b, DataFormat::Float16_b};

    const std::uint32_t cb_base = get_write_ptr(cb_id_b);

#define PAIR_IDX(r, iy, kk) (all_pair_indices[all_pair_offsets[(r)] + (iy) * max_kt_dim + (kk)])

    for (std::uint32_t iter_y = 0; iter_y < num_iters_y; ++iter_y) {
        // Determine barrier participants and leader for this iter_y.
        std::uint32_t barrier_participants = 0;
        std::uint32_t barrier_leader = core_y;
        bool leader_found = false;
        for (std::uint32_t r = 0; r < num_cores_y; ++r) {
            if (iter_y < all_num_iters_y[r]) {
                barrier_participants++;
                if (!leader_found) {
                    barrier_leader = r;
                    leader_found = true;
                }
            }
        }
        const bool is_barrier_leader = (core_y == barrier_leader);
        const std::uint64_t leader_barrier_noc_addr =
            get_noc_addr(column_noc_x, noc_y_table[barrier_leader], sem_barrier);

        for (std::uint32_t iter_x = 0; iter_x < num_iters_x; ++iter_x) {
            const std::uint32_t nt = output_idx_x_start + iter_x;

            for (std::uint32_t k = 0; k < max_kt_dim; ++k) {
                const std::uint32_t my_pair_idx = PAIR_IDX(core_y, iter_y, k);

                // Compute share set for this column, iter_y, and k.
                std::uint32_t share_set_size = 0;
                std::uint32_t injector_idx = core_y;
                std::uint32_t sender_idx = INVALID_CORE;
                std::uint32_t downstream_idx = INVALID_CORE;

                for (std::uint32_t r = 0; r < num_cores_y; ++r) {
                    if (iter_y >= all_num_iters_y[r]) {
                        continue;
                    }
                    const std::uint32_t their_pair_idx = PAIR_IDX(r, iter_y, k);
                    if (their_pair_idx == my_pair_idx) {
                        share_set_size++;
                        if (r > injector_idx) {
                            injector_idx = r;
                        }
                        if (r > core_y && (sender_idx == INVALID_CORE || r < sender_idx)) {
                            sender_idx = r;
                        }
                        if (r < core_y && (downstream_idx == INVALID_CORE || r > downstream_idx)) {
                            downstream_idx = r;
                        }
                    }
                }

                const bool is_solo = (share_set_size <= 1);
                const bool is_injector = (core_y == injector_idx);

                cb_reserve_back(cb_id_b, 1);
                std::uint32_t b_l1 = get_write_ptr(cb_id_b);
                const std::uint32_t slot_bit = (b_l1 == cb_base) ? 0 : 1;

                if (is_solo || is_injector) {
                    const std::uint32_t b_tile_idx = my_pair_idx * num_output_cols + nt;
                    noc_async_read(b_reader.get_noc_addr(b_tile_idx), b_l1, tile_size_b);
                    noc_async_read_barrier();
                } else {
                    // Receive from the sender core above us in the share set.
                    noc_semaphore_set(sem_receiver_ptr, 0);
                    const std::uint64_t sender_sem_noc_addr =
                        get_noc_addr(column_noc_x, noc_y_table[sender_idx], sem_sender);
                    noc_semaphore_inc(sender_sem_noc_addr, 1 + slot_bit);
                    noc_semaphore_wait(sem_receiver_ptr, 1);
                }

                cb_push_back(cb_id_b, 1);

                if (downstream_idx != INVALID_CORE) {
                    // Forward the B tile to the downstream core in the share set.
                    while (*sem_sender_ptr == 0) {
                    }
                    const std::uint32_t receiver_slot_bit = *sem_sender_ptr - 1;
                    noc_semaphore_set(sem_sender_ptr, 0);

                    const std::uint32_t receiver_dest = cb_base + receiver_slot_bit * tile_size_b;
                    const std::uint64_t dest_data_addr =
                        get_noc_addr(column_noc_x, noc_y_table[downstream_idx], receiver_dest);
                    noc_async_write(b_l1, dest_data_addr, tile_size_b);
                    noc_async_write_barrier();

                    const std::uint64_t dest_recv_sem =
                        get_noc_addr(column_noc_x, noc_y_table[downstream_idx], sem_receiver);
                    noc_semaphore_inc(dest_recv_sem, 1);
                    noc_async_atomic_barrier();
                }

                // Column-wide barrier before moving to the next K-step.
                if (barrier_participants > 1) {
                    if (is_barrier_leader) {
                        noc_semaphore_wait(sem_barrier_ptr, barrier_participants - 1);
                        noc_semaphore_set(sem_barrier_ptr, 0);
                        for (std::uint32_t r = 0; r < num_cores_y; ++r) {
                            if (r == core_y) {
                                continue;
                            }
                            if (iter_y >= all_num_iters_y[r]) {
                                continue;
                            }
                            const std::uint64_t their_release = get_noc_addr(column_noc_x, noc_y_table[r], sem_release);
                            noc_semaphore_inc(their_release, 1);
                        }
                        noc_async_atomic_barrier();
                    } else {
                        noc_semaphore_inc(leader_barrier_noc_addr, 1);
                        noc_async_atomic_barrier();
                        noc_semaphore_wait(sem_release_ptr, 1);
                        noc_semaphore_set(sem_release_ptr, 0);
                    }
                }
            }
        }
    }
}
