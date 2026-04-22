#pragma once
#include <stdint.h>
#include "dataflow_api.h"

namespace spmm {

// Load indexing data as a single contiguous buffer read.
// Used by reader_block_iter and reader_snf_in0_reader.
// CB must be configured with page_size = total buffer size (1 page = full buffer).
// Returns pointer to the loaded uint32_t data.
template <bool is_dram>
inline uint32_t* load_indexing_contiguous(
    uint32_t cb_id,
    uint32_t bank_base_addr,
    uint32_t tile_size,
    DataFormat data_format,
    uint32_t num_tiles)
{
    const uint32_t total_size = num_tiles * tile_size;
    const InterleavedAddrGenFast<is_dram> addr_gen = {
        .bank_base_address = bank_base_addr,
        .page_size = tile_size,
        .data_format = data_format};

    cb_reserve_back(cb_id, 1);
    uint32_t l1_addr = get_write_ptr(cb_id);
    uint64_t noc_addr = get_noc_addr(0, addr_gen);
    noc_async_read(noc_addr, l1_addr, total_size);
    noc_async_read_barrier();
    cb_push_back(cb_id, 1);
    return reinterpret_cast<uint32_t*>(l1_addr);
}

// Load indexing data tile-by-tile.
// Used by reader_block_iter_in0_receiver and reader_block_iter_in0_sender.
// CB must be configured with page_size = tile_size (num_tiles pages).
// Returns pointer to the loaded uint32_t data.
template <bool is_dram>
inline uint32_t* load_indexing_tiled(
    uint32_t cb_id,
    uint32_t bank_base_addr,
    uint32_t tile_size,
    DataFormat data_format,
    uint32_t num_tiles)
{
    const InterleavedAddrGenFast<is_dram> addr_gen = {
        .bank_base_address = bank_base_addr,
        .page_size = tile_size,
        .data_format = data_format};

    cb_reserve_back(cb_id, num_tiles);
    uint32_t l1_addr = get_write_ptr(cb_id);
    uint32_t l1_start = l1_addr;
    for (uint32_t i = 0; i < num_tiles; i++) {
        noc_async_read_tile(i, addr_gen, l1_addr);
        l1_addr += tile_size;
    }
    noc_async_read_barrier();
    cb_push_back(cb_id, num_tiles);
    return reinterpret_cast<uint32_t*>(l1_start);
}

// Wait for indexing data that was loaded by another RISC on the same core.
// Returns pointer to the data in the CB.
inline uint32_t* wait_for_indexing(uint32_t cb_id, uint32_t num_tiles = 1) {
    cb_wait_front(cb_id, num_tiles);
    return reinterpret_cast<uint32_t*>(get_read_ptr(cb_id));
}

}  // namespace spmm
