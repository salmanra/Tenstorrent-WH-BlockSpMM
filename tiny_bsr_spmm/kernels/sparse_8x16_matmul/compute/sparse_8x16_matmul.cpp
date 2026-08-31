// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "llk_math_sparse_8x16_matmul.h"
#include "llk_pack_sparse_8x16.h"
#include "llk_unpack_sparse_8x16.h"

constexpr std::uint32_t cb_id_a = tt::CBIndex::c_0;
constexpr std::uint32_t cb_id_b = tt::CBIndex::c_1;
constexpr std::uint32_t cb_id_out = tt::CBIndex::c_2;

constexpr ckernel::DstSync dst_sync = ckernel::DstSync::SyncHalf;
constexpr bool is_fp32_dest_acc_en = false;
constexpr std::uint32_t bf16_format = static_cast<std::uint32_t>(DataFormat::Float16_b);

#if defined(UCK_CHLKC_UNPACK)

void kernel_main() {
    constexpr std::uint32_t kt_dim = get_compile_time_arg_val(0);

    _llk_unpack_sparse_8x16_hw_configure_<is_fp32_dest_acc_en>(bf16_format, bf16_format, 256, 512);
    _llk_unpack_sparse_8x16_init_();

    for (std::uint32_t k = 0; k < kt_dim; ++k) {
        cb_wait_front(cb_id_a, 1);
        cb_wait_front(cb_id_b, 1);

        const std::uint32_t a_addr = L1_ADDRESS(get_local_cb_interface(cb_id_a).fifo_rd_ptr << 4);
        const std::uint32_t b_addr = L1_ADDRESS(get_local_cb_interface(cb_id_b).fifo_rd_ptr << 4);

        _llk_unpack_sparse_8x16_(a_addr, b_addr);

        cb_pop_front(cb_id_a, 1);
        cb_pop_front(cb_id_b, 1);
    }
}

#endif

#if defined(UCK_CHLKC_MATH)

void kernel_main() {
    constexpr std::uint32_t kt_dim = get_compile_time_arg_val(0);

    _llk_math_pack_sync_init_<dst_sync, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(bf16_format, bf16_format);
    _llk_math_sparse_8x16_matmul_init_<MATH_FIDELITY>(0, 1, 1);

    _llk_math_wait_for_dest_available_<dst_sync>();

    for (std::uint32_t k = 0; k < kt_dim; ++k) {
        _llk_math_sparse_8x16_matmul_(0, 1, 1);
    }

    _llk_math_dest_section_done_<dst_sync, is_fp32_dest_acc_en>();
}

#endif

#if defined(UCK_CHLKC_PACK)

void kernel_main() {
    _llk_pack_sparse_8x16_hw_configure_<is_fp32_dest_acc_en>(bf16_format, bf16_format, 256);
    _llk_pack_sparse_8x16_init_<dst_sync, is_fp32_dest_acc_en>(bf16_format);

    _llk_packer_wait_for_math_done_();

    cb_reserve_back(cb_id_out, 1);
    const std::uint32_t out_addr = L1_ADDRESS(get_local_cb_interface(cb_id_out).fifo_wr_ptr << 4);
    _llk_pack_sparse_8x16_<dst_sync, is_fp32_dest_acc_en>(0, out_addr);
    cb_push_back(cb_id_out, 1);

    _llk_pack_dest_section_done_<dst_sync, is_fp32_dest_acc_en>();
}

#endif
