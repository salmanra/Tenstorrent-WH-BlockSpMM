// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_pack.h"
#include "llk_pack_common.h"

using namespace ckernel;

/**
 * @brief Standalone packer LLK for the 8x16x16 sparse matmul unit.
 *
 * Packs one 8x16 output face from the destination register to L1 per call.
 */

template <bool is_fp32_dest_acc_en>
inline void _llk_pack_sparse_8x16_hw_configure_(
    const std::uint32_t pack_src_format,
    const std::uint32_t pack_dst_format,
    const std::uint32_t tile_size = 256) // 8 x 16 x sizeof(bfloat16)
{
    // 8x16 output: one face with 8 rows, 16 columns.
    // partial_face=true tells the packer to write the 8-row face correctly.
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, PackMode::Default>(
        pack_src_format, pack_dst_format, tile_size, 8 /* face_r_dim */, 16 /* tile_c_dim */, 1 /* num_faces */, true /* partial_face */);
}

template <DstSync dest_sync, bool is_fp32_dest_acc_en>
inline void _llk_pack_sparse_8x16_init_(const std::uint32_t pack_src_format)
{
    // Init packer MOP for 8x16 output face. Strides and x-end are configured here.
    _llk_pack_init_<PackMode::Default, false /* zero_output */>(
        pack_src_format, 8U /* face_r_dim */, 16U /* tile_c_dim */, 1U /* num_faces */, 1U /* num_tiles */, true /* partial_face */);

    _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
}

/**
 * @brief Pack one 8x16 output face from Dst to L1.
 *
 * @param dst_index: Destination tile index to read from.
 * @param addr: L1 address to write the 8x16 output tile to.
 */
template <DstSync dest_sync, bool is_fp32_dest_acc_en>
inline void _llk_pack_sparse_8x16_(const std::uint32_t dst_index, const std::uint32_t addr)
{
    _llk_packer_wait_for_math_done_();
    _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(dst_index, addr);
}
