#include "../host_code.hpp"
#include "../host_code_utils.hpp"

namespace bsr_host_code {

// Re-export shared buffer helpers so existing SpMM code can call them unqualified
using sparse_common::MakeBuffer;
using sparse_common::MakeCircularBuffer;
using sparse_common::MakeCircularBufferFP32;

CoreCoord clamped_prev(const std::vector<CoreCoord>& order, uint32_t index) {
    return order.at(index == 0 ? 0 : index - 1);
}

CoreCoord clamped_next(const std::vector<CoreCoord>& order, uint32_t index) {
    const uint32_t last = static_cast<uint32_t>(order.size() - 1);
    return order.at(index >= last ? last : index + 1);
}

uint32_t _get_maximum_block_dim_with_NoC_args(int32_t block_dim, int32_t in0_block_w, int32_t num_tiles_in_NoC_args) {
    int32_t num_available_tiles_in_SRAM = 400; // as provided by TT code. roughly: SRAM size in bytes divided by tile size in bytes
                                               // but i think this is the Grayskull number. not important for now
    num_available_tiles_in_SRAM -= num_tiles_in_NoC_args;
    int32_t other_dim = (num_available_tiles_in_SRAM - 2 * in0_block_w * block_dim) / (2 * in0_block_w + block_dim);
    if (other_dim > 0) {
        return other_dim;
    }
    return 0;
}


uint32_t get_Npc_from_BSR_block_size(uint32_t Nt, uint32_t Mpc, uint32_t in0_block_w, uint32_t num_cores_x, uint32_t num_cores_y, uint32_t num_tiles_for_indexing, uint32_t nnz_rows) {
    auto Nt_fac = get_prime_factors(Nt);
    uint32_t Npc_min = 1;
    for (auto it = Nt_fac.begin(); it != Nt_fac.end(); ++it) {
        auto ele = *it;
        if (ele > num_cores_x) {
            Npc_min *= ele;
            Nt_fac.erase(it);
            --it;
        }
    }
    uint32_t num_iters_y = (nnz_rows + num_cores_y - 1) / num_cores_y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    uint32_t Npc = Npc_min;
    auto Npc_choices = get_possible_products(Nt_fac);
    auto Npc_max = _get_maximum_block_dim_with_NoC_args(Mpc, in0_block_w, num_tiles_for_indexing);
    for (auto& ele : Npc_choices) {
        uint32_t candidate_NpC = ele * Npc_min;
        uint32_t num_blocks_x = Nt / candidate_NpC;
        uint32_t num_iters_x = (num_blocks_x + num_cores_x - 1) / num_cores_x;
        uint32_t num_blocks_total = nnz_rows * num_blocks_x;
        uint32_t num_work_regions = (num_blocks_total + num_iters_x * num_iters_y - 1)/ (num_iters_x * num_iters_y);
        if (num_blocks_x < num_cores_x){
            // hmmm... can it be that NWR<NCT but we can still do better than the current NPC?
            break;
        }
        if (ele * Npc_min <= Npc_max) {
            Npc = ele * Npc_min;
        } else {
            break;
        }
    }

    return Npc;
}

} // namespace bsr_host_code
