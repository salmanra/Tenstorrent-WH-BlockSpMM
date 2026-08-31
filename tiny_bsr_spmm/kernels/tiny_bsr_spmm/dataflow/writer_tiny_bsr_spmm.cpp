// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/kernel_structs.h"

void kernel_main() {
    constexpr std::uint32_t out_is_dram = get_compile_time_arg_val(0);
    constexpr std::uint32_t num_output_rows = get_compile_time_arg_val(1);
    constexpr std::uint32_t num_output_cols = get_compile_time_arg_val(2);
    constexpr std::uint32_t num_cores_x = get_compile_time_arg_val(3);
    constexpr std::uint32_t num_cores_y = get_compile_time_arg_val(4);

    constexpr std::uint32_t cb_id_out = tt::CBIndex::c_2;
    constexpr std::uint32_t tile_size_out = 256;  // 8 * 16 * 2 bytes
    constexpr std::uint32_t tile_hw_out = 128;    // 8 * 16 datums

    const std::uint32_t out_addr = get_arg_val<std::uint32_t>(0);
    const std::uint32_t core_x = get_arg_val<std::uint32_t>(1);
    const std::uint32_t core_y = get_arg_val<std::uint32_t>(2);
    const std::uint32_t sem_writer_reader = get_semaphore(get_arg_val<std::uint32_t>(3));

    volatile tt_l1_ptr std::uint32_t* sem_writer_reader_ptr =
        reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(sem_writer_reader);

    const auto out_writer =
        InterleavedAddrGenFast<out_is_dram, tile_hw_out>{out_addr, tile_size_out, DataFormat::Float16_b};

    for (std::uint32_t rb = core_y; rb < num_output_rows; rb += num_cores_y) {
        for (std::uint32_t nt = core_x; nt < num_output_cols; nt += num_cores_x) {
            const std::uint32_t tile_idx = rb * num_output_cols + nt;

            cb_wait_front(cb_id_out, 1);
            const std::uint32_t out_l1 = get_read_ptr(cb_id_out);

            noc_async_write(out_l1, out_writer.get_noc_addr(tile_idx), tile_size_out);
            noc_async_write_barrier();

            cb_pop_front(cb_id_out, 1);

            // Signal the reader that it may start the next tile.
            const std::uint64_t sem_writer_reader_noc_addr = get_noc_addr(sem_writer_reader);
            noc_semaphore_inc(sem_writer_reader_noc_addr, 1);
        }
    }
}
