#pragma once
#include <stdint.h>
#include "dataflow_api.h"

namespace spmm {

// Read a block of tiles from DRAM into L1.
// Iterates row-major over a (block_h x block_w) tile grid.
// l1_addr is advanced by block_h * block_w * tile_size.
template <typename AddrGen>
inline void read_block_by_tile(
    uint32_t start_tile_id,
    const AddrGen& addr_gen,
    uint32_t& l1_addr,
    uint32_t tile_size,
    uint32_t block_h,
    uint32_t block_w,
    uint32_t stride_h,
    uint32_t stride_w)
{
    uint32_t row_start_tile_id = start_tile_id;
    for (uint32_t h = 0; h < block_h; h++) {
        uint32_t tile_id = row_start_tile_id;
        for (uint32_t w = 0; w < block_w; w++) {
            noc_async_read_tile(tile_id, addr_gen, l1_addr);
            l1_addr += tile_size;
            tile_id += stride_w;
        }
        row_start_tile_id += stride_h;
    }
}



// Write a subblock of tiles from L1 to DRAM.
// Iterates row-major over a (subblock_h x subblock_w) tile grid.
// l1_addr is advanced by subblock_h * subblock_w * tile_size.
template <typename AddrGen>
inline void write_subblock_by_tile(
    uint32_t start_tile_id,
    const AddrGen& addr_gen,
    uint32_t& l1_addr,
    uint32_t tile_size,
    uint32_t subblock_h,
    uint32_t subblock_w,
    uint32_t stride_h,
    uint32_t stride_w)
{
    uint32_t row_start_tile_id = start_tile_id;
    for (uint32_t h = 0; h < subblock_h; h++) {
        uint32_t tile_id = row_start_tile_id;
        for (uint32_t w = 0; w < subblock_w; w++) {
            noc_async_write_tile(tile_id, addr_gen, l1_addr);
            l1_addr += tile_size;
            tile_id += stride_w;
        }
        row_start_tile_id += stride_h;
    }
}

}  // namespace spmm
