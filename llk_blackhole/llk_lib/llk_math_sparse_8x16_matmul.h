// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_math_common.h"

using namespace ckernel;

/**
 * @brief Standalone 8x16x16 matmul LLK for Blackhole.
 *
 * Computes one 8x16 output face per call from:
 *   SrcB[8,16] (sparse A pair: two 8x8 K-blocks side by side)
 *   SrcA[16,16] (dense B slab: two 8-row K-slices stacked)
 *
 * The kernel is intentionally narrow: it does not reuse the existing llk_math_matmul.h,
 * so it does not risk regressing the stock matmul paths. It uses the same low-level
 * primitives (addr_mod_t, ckernel_template, replay buffer, TTI_MVMUL).
 */

template <MathFidelity math_fidelity>
inline void sparse_8x16_matmul_configure_addrmod()
{
    constexpr bool high_fidelity = is_high_fidelity(math_fidelity);

    // One MVMUL writes the whole 8x16 face. No src/dest increment needed within the tile.
    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 0, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_0);

    if constexpr (high_fidelity)
    {
        // Advance the fidelity phase between high-fidelity accumulation steps.
        addr_mod_t {
            .fidelity = {.incr = 1, .clr = 0},
        }
            .set(ADDR_MOD_1);

        // Clear the fidelity phase after the last high-fidelity step. Counters are not
        // modified because the single 8x16 face never increments them.
        addr_mod_t {
            .fidelity = {.incr = 0, .clr = 1},
        }
            .set(ADDR_MOD_2);
    }
}

template <MathFidelity math_fidelity>
inline void sparse_8x16_matmul_configure_mop()
{
    constexpr bool high_fidelity = is_high_fidelity(math_fidelity);

    if constexpr (high_fidelity)
    {
        // High fidelity: emit one MVMUL per phase, incrementing fidelity between phases
        // and clearing it after the last phase. The end-op clears both sources and resets
        // all source/destination counters so the next K-step can load fresh tiles.
        constexpr std::uint32_t num_phases = to_underlying(math_fidelity);

        load_replay_buf(
            ckernel::math::replay_buf_offset,
            num_phases,
            []
            {
                for (std::uint32_t phase = 0; phase < num_phases - 1; phase++)
                {
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
                }
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0);
            });

        ckernel_template tmp(1 /* outer loop */, 1, lltt::replay_insn(ckernel::math::replay_buf_offset, num_phases));
        tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD_F));
        tmp.program();
    }
    else
    {
        // Low fidelity: one MVMUL per K-step. The end-op clears both sources and resets
        // all source/destination counters so the next K-step can load fresh tiles.
        load_replay_buf(ckernel::math::replay_buf_offset, 1, [] { TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); });

        ckernel_template tmp(1 /* outer loop */, 1, lltt::replay_insn(ckernel::math::replay_buf_offset, 1));
        tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD_F));
        tmp.program();
    }
}

/**
 * @brief Initialize the math thread for the 8x16x16 sparse matmul unit.
 *
 * @tparam math_fidelity: MathFidelity value (LoFi/HiFi2/HiFi3/HiFi4).
 * @param transpose: 0 only; non-zero reserved for future transposed B faces.
 * @param ct_dim: Number of 16-wide output column tiles (N dimension / 16).
 * @param rt_dim: Number of 8-row output row tiles (M dimension / 8).
 *
 * @note Call `_llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>()` and
 *       `_llk_math_hw_configure_<is_fp32_dest_acc_en>()` before this, as in the stock
 *       matmul test pattern.
 */
template <MathFidelity math_fidelity>
inline void _llk_math_sparse_8x16_matmul_init_(const std::uint32_t transpose = 0, const std::uint32_t ct_dim = 1, const std::uint32_t rt_dim = 1)
{
    (void)transpose;
    (void)ct_dim;
    (void)rt_dim;

    sparse_8x16_matmul_configure_addrmod<math_fidelity>();
    sparse_8x16_matmul_configure_mop<math_fidelity>();
    math::reset_counters(p_setrwc::SET_ABD_F);
}

/**
 * @brief Execute one 8x16x16 sparse matmul block.
 *
 * Iterates over the output tile grid, accumulating K-steps into each 8x16 output tile.
 * The caller must feed one SrcB tile (8x16) and one SrcA tile (16x16) per K-step before
 * calling this function, using the matching unpack helper.
 *
 * @param dst_index: Base destination tile index for the output block.
 * @param ct_dim: Number of 16-wide output column tiles.
 * @param rt_dim: Number of 8-row output row tiles.
 */
inline void _llk_math_sparse_8x16_matmul_(std::uint32_t dst_index, const std::uint32_t ct_dim = 1, const std::uint32_t rt_dim = 1)
{
    const bool reuse_a          = ct_dim >= rt_dim; // SrcB/in0 is reused across output columns
    const std::uint32_t t_dim   = reuse_a ? rt_dim : ct_dim;
    const std::uint32_t rut_dim = reuse_a ? ct_dim : rt_dim; // streamed dimension

    for (std::uint32_t t = 0; t < t_dim; t++)
    {
        for (std::uint32_t rut = 0; rut < rut_dim; rut++)
        {
            math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index + (reuse_a ? ct_dim * t + rut : t + rut * ct_dim));

            ckernel_template::run();
        }
    }
}
