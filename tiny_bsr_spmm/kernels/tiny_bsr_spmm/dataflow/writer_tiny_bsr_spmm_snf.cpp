// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/kernel_structs.h"

// in0 reader/writer for SnF: reads/forwards sparse A faces along a row of cores
// (one tile at a time) and writes the output tiles back to DRAM.
void kernel_main() {
    constexpr std::uint32_t a_is_dram = get_compile_time_arg_val(0);
    constexpr std::uint32_t out_is_dram = get_compile_time_arg_val(1);
    constexpr std::uint32_t max_kt_dim = get_compile_time_arg_val(2);
    constexpr std::uint32_t num_output_rows = get_compile_time_arg_val(3);
    constexpr std::uint32_t num_output_cols = get_compile_time_arg_val(4);
    constexpr std::uint32_t num_cores_x = get_compile_time_arg_val(5);
    constexpr std::uint32_t num_cores_y = get_compile_time_arg_val(6);

    constexpr std::uint32_t cb_id_a = tt::CBIndex::c_0;
    constexpr std::uint32_t cb_id_out = tt::CBIndex::c_2;

    constexpr std::uint32_t tile_size_a = 256;    // 8 * 16 * 2 bytes
    constexpr std::uint32_t tile_size_out = 256;  // 8 * 16 * 2 bytes
    constexpr std::uint32_t tile_hw_a = 128;      // 8 * 16 datums
    constexpr std::uint32_t tile_hw_out = 128;    // 8 * 16 datums
    constexpr std::uint32_t a_block_bytes = max_kt_dim * tile_size_a;

    const std::uint32_t a_addr = get_arg_val<std::uint32_t>(0);
    const std::uint32_t out_addr = get_arg_val<std::uint32_t>(1);
    const std::uint32_t core_x = get_arg_val<std::uint32_t>(2);
    const std::uint32_t core_y = get_arg_val<std::uint32_t>(3);
    const std::uint32_t sem_sender = get_semaphore(get_arg_val<std::uint32_t>(4));
    const std::uint32_t sem_receiver = get_semaphore(get_arg_val<std::uint32_t>(5));
    const std::uint32_t next_core_noc_x = get_arg_val<std::uint32_t>(6);
    const std::uint32_t next_core_noc_y = get_arg_val<std::uint32_t>(7);
    const std::uint32_t prev_core_noc_x = get_arg_val<std::uint32_t>(8);
    const std::uint32_t prev_core_noc_y = get_arg_val<std::uint32_t>(9);
    const std::uint32_t is_injector = get_arg_val<std::uint32_t>(10);
    const std::uint32_t is_sink = get_arg_val<std::uint32_t>(11);

    volatile tt_l1_ptr std::uint32_t* sem_sender_ptr = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(sem_sender);
    volatile tt_l1_ptr std::uint32_t* sem_receiver_ptr =
        reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(sem_receiver);

    const std::uint64_t upstream_sender_noc_addr = get_noc_addr(prev_core_noc_x, prev_core_noc_y, sem_sender);
    const std::uint64_t downstream_receiver_noc_addr = get_noc_addr(next_core_noc_x, next_core_noc_y, sem_receiver);

    const auto a_reader = InterleavedAddrGenFast<a_is_dram, tile_hw_a>{a_addr, tile_size_a, DataFormat::Float16_b};
    const auto out_writer =
        InterleavedAddrGenFast<out_is_dram, tile_hw_out>{out_addr, tile_size_out, DataFormat::Float16_b};

    for (std::uint32_t rb = core_y; rb < num_output_rows; rb += num_cores_y) {
        for (std::uint32_t nt = core_x; nt < num_output_cols; nt += num_cores_x) {
            const std::uint32_t tile_idx = rb * num_output_cols + nt;

            cb_reserve_back(cb_id_a, max_kt_dim);
            std::uint32_t a_l1 = get_write_ptr(cb_id_a);

            if (is_injector) {
                for (std::uint32_t k = 0; k < max_kt_dim; ++k) {
                    const std::uint32_t idx = tile_idx * max_kt_dim + k;
                    noc_async_read(a_reader.get_noc_addr(idx), a_l1 + k * tile_size_a, tile_size_a);
                }
                noc_async_read_barrier();
            } else {
                // Rendezvous with upstream: declare readiness, then wait for the data.
                noc_semaphore_set(sem_receiver_ptr, 0);
                noc_semaphore_inc(upstream_sender_noc_addr, 1);
                noc_semaphore_wait(sem_receiver_ptr, 1);
            }

            cb_push_back(cb_id_a, max_kt_dim);

            if (!is_sink) {
                // Forward the whole A-face block to the next core's A CB (same L1 address).
                noc_semaphore_wait(sem_sender_ptr, 1);
                noc_semaphore_set(sem_sender_ptr, 0);

                noc_async_write(a_l1, get_noc_addr(next_core_noc_x, next_core_noc_y, a_l1), a_block_bytes);
                noc_async_write_barrier();

                noc_semaphore_inc(downstream_receiver_noc_addr, 1);
            }

            cb_wait_front(cb_id_out, 1);
            const std::uint32_t out_l1 = get_read_ptr(cb_id_out);

            noc_async_write(out_l1, out_writer.get_noc_addr(tile_idx), tile_size_out);
            noc_async_write_barrier();

            cb_pop_front(cb_id_out, 1);
        }
    }
}
