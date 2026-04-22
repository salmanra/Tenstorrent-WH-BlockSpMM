#pragma once
#include <stdint.h>

// Circular buffer assignments for SDDMM.
// SDDMM needs CB slots for:
//   - Sparse sampling mask B (block data + indptr + indices)
//   - Two dense inputs C and D
//   - Output and intermediate accumulation

namespace sddmm {

constexpr uint32_t cb_id_sparse_data   = 0;   // B block data (sampling mask)
constexpr uint32_t cb_id_dense_c       = 1;   // C data tiles (dense M×K)
constexpr uint32_t cb_id_dense_d       = 2;   // D data tiles (dense K×N)
constexpr uint32_t cb_id_sparse_indices = 3;  // B col_indices
constexpr uint32_t cb_id_sparse_indptr  = 4;  // B indptr
constexpr uint32_t cb_id_out           = 16;  // Output block data
constexpr uint32_t cb_id_intermed      = 24;  // Intermediate dense product before hadamard

} // namespace sddmm
