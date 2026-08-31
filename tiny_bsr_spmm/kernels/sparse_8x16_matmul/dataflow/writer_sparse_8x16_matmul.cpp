// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/kernel_structs.h"

void kernel_main() {
    constexpr std::uint32_t out_is_dram = get_compile_time_arg_val(0);

    constexpr std::uint32_t cb_id_out = tt::CBIndex::c_2;
    constexpr std::uint32_t tile_size_out = 256;  // 8 * 16 * 2 bytes
    constexpr std::uint32_t tile_hw_out = 128;    // 8 * 16 datums

    const std::uint32_t out_addr = get_arg_val<std::uint32_t>(0);

    const auto out_writer =
        InterleavedAddrGenFast<out_is_dram, tile_hw_out>{out_addr, tile_size_out, DataFormat::Float16_b};

    cb_wait_front(cb_id_out, 1);
    const std::uint32_t out_l1 = get_read_ptr(cb_id_out);

    noc_async_write(out_l1, out_writer.get_noc_addr(0), tile_size_out);
    noc_async_write_barrier();

    cb_pop_front(cb_id_out, 1);
}
