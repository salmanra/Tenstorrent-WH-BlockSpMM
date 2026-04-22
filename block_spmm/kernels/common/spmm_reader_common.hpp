#pragma once
#include <stdint.h>
#include "dataflow_api.h"

namespace spmm {

constexpr uint32_t cb_id_in0         = tt::CBIndex::c_0;
constexpr uint32_t cb_id_in1         = tt::CBIndex::c_1;
constexpr uint32_t cb_id_col_indices = tt::CBIndex::c_2;
constexpr uint32_t cb_id_indptr      = tt::CBIndex::c_3;
constexpr uint32_t cb_id_out         = tt::CBIndex::c_16;

struct TileInfo {
    uint32_t in0_tile_size;
    DataFormat in0_format;
    uint32_t in1_tile_size;
    DataFormat in1_format;
    uint32_t col_indices_tile_size;
    DataFormat col_indices_format;
    uint32_t indptr_tile_size;
    DataFormat indptr_format;
};

inline TileInfo get_tile_info() {
    return TileInfo{
        .in0_tile_size         = get_tile_size(cb_id_in0),
        .in0_format            = get_dataformat(cb_id_in0),
        .in1_tile_size         = get_tile_size(cb_id_in1),
        .in1_format            = get_dataformat(cb_id_in1),
        .col_indices_tile_size = get_tile_size(cb_id_col_indices),
        .col_indices_format    = get_dataformat(cb_id_col_indices),
        .indptr_tile_size      = get_tile_size(cb_id_indptr),
        .indptr_format         = get_dataformat(cb_id_indptr),
    };
}

}  // namespace spmm
