// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/kernel_structs.h"

void kernel_main() {
    constexpr std::uint32_t a_is_dram = get_compile_time_arg_val(0);
    constexpr std::uint32_t b_is_dram = get_compile_time_arg_val(1);
    constexpr std::uint32_t kt_dim = get_compile_time_arg_val(2);

    constexpr std::uint32_t cb_id_a = tt::CBIndex::c_0;
    constexpr std::uint32_t cb_id_b = tt::CBIndex::c_1;

    constexpr std::uint32_t tile_size_a = 256;  // 8 * 16 * 2 bytes
    constexpr std::uint32_t tile_size_b = 512;  // 16 * 16 * 2 bytes
    constexpr std::uint32_t tile_hw_a = 128;    // 8 * 16 datums
    constexpr std::uint32_t tile_hw_b = 256;    // 16 * 16 datums

    const std::uint32_t a_addr = get_arg_val<std::uint32_t>(0);
    const std::uint32_t b_addr = get_arg_val<std::uint32_t>(1);

    const auto a_reader = InterleavedAddrGenFast<a_is_dram, tile_hw_a>{a_addr, tile_size_a, DataFormat::Float16_b};
    const auto b_reader = InterleavedAddrGenFast<b_is_dram, tile_hw_b>{b_addr, tile_size_b, DataFormat::Float16_b};

    for (std::uint32_t k = 0; k < kt_dim; ++k) {
        cb_reserve_back(cb_id_a, 1);
        cb_reserve_back(cb_id_b, 1);

        const std::uint32_t a_l1 = get_write_ptr(cb_id_a);
        const std::uint32_t b_l1 = get_write_ptr(cb_id_b);

        noc_async_read(a_reader.get_noc_addr(k), a_l1, tile_size_a);
        noc_async_read(b_reader.get_noc_addr(k), b_l1, tile_size_b);
        noc_async_read_barrier();

        cb_push_back(cb_id_a, 1);
        cb_push_back(cb_id_b, 1);
    }
}
