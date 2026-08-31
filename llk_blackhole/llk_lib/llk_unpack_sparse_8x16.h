// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "cunpack_common.h"
#include "llk_unpack_common.h"

using namespace ckernel;
using namespace ckernel::unpacker;

/**
 * @brief Standalone unpacker LLK for the 8x16x16 sparse matmul unit.
 *
 * Unpacks one 8x16 SrcB tile (sparse A pair) and one 16x16 SrcA tile (dense B slab) per call.
 * Designed for explicit sparse gather: caller supplies the L1 address of each tile per K-step.
 * Does not use the dense ct_dim/rt_dim/kt_dim grid of the stock matmul unpacker.
 */

template <bool is_fp32_dest_acc_en>
inline void _llk_unpack_sparse_8x16_hw_configure_(
    const std::uint32_t src_format,
    const std::uint32_t dst_format,
    const std::uint32_t tile_size_in0 = 256, // 8 x 16 x sizeof(bfloat16)
    const std::uint32_t tile_size_in1 = 512) // 16 x 16 x sizeof(bfloat16)
{
    // unpA = operand A = in0 = SrcB (8x16, one partial face)
    // unpB = operand B = in1 = SrcA (16x16, one full face)
    configure_unpack_AB<is_fp32_dest_acc_en, false, false, false>(
        src_format,
        src_format,
        dst_format,
        dst_format,
        8 /* unpA_face_r_dim */,
        16 /* unpB_face_r_dim */,
        false /* transpose_xy_srca_en */,
        1 /* unpA_num_faces */,
        1 /* unpB_num_faces */);

    // Set tile-size GPRs used by the unpacker MOP for address advance.
    TT_SETDMAREG(0, LOWER_HALFWORD(tile_size_in0), 0, LO_16(p_gpr_unpack::TILE_SIZE_A));
    TT_SETDMAREG(0, LOWER_HALFWORD(tile_size_in1), 0, LO_16(p_gpr_unpack::TILE_SIZE_B));
}

inline void _llk_unpack_sparse_8x16_init_()
{
    // Set x_end for the single UNPACR per operand. The unpacker config mapping follows the
    // matmul convention: UNP_A is for SrcA/in1 (operand B, 16x16 full face), UNP_B is for
    // SrcB/in0 (operand A, 8x16 partial face).
    TT_SETADCXX(p_setadc::UNP_A, FACE_R_DIM * FACE_C_DIM - 1, 0x0); // 16x16 => 255
    TT_SETADCXX(p_setadc::UNP_B, 8 * FACE_C_DIM - 1, 0x0);          // 8x16  => 127

    // Reset config context to 0 (same as stock matmul init).
    reset_config_context();
}

/**
 * @brief Unpack one 8x16 SrcB tile and one 16x16 SrcA tile.
 *
 * @param addr_in0: L1 address of the 8x16 sparse A tile (goes to SrcB).
 * @param addr_in1: L1 address of the 16x16 dense B tile (goes to SrcA).
 */
inline void _llk_unpack_sparse_8x16_(const std::uint32_t addr_in0, const std::uint32_t addr_in1)
{
    wait_for_next_context(2);

    volatile std::uint32_t *cfg = get_cfg_pointer();
    // First argument -> THCON_SEC0 (SrcA/in1), second -> THCON_SEC1 (SrcB/in0).
    _llk_unpack_configure_addresses_(addr_in1, addr_in0, cfg);

    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    // SrcB/in0 is an 8x16 partial face: zero the register, load the two halves, then reset the
    // face counter so the next step starts at the top of the face.
    TTI_UNPACR_NOP(SrcB, 0, 0, 0 /*Set Dvalid*/, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    TTI_UNPACR(SrcB, 0b00010001, 0, 0, 0, 1 /*Set OvrdThreadId*/, 0 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_UNPACR(SrcB, 0b00010001, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TT_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 0, 0b0101); // Set ch0_z=0, ch1_z=0

    // SrcA/in1 is a full 16x16 face.
    TTI_UNPACR(SrcA, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}

inline void _llk_unpack_sparse_8x16_uninit_()
{
    // x-end is transient and reprogrammed by the next operation; nothing to restore.
}
