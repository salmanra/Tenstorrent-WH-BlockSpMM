#pragma once

#include "include_me.hpp"

using namespace tt;
using namespace tt::tt_metal;

using CoreSpec = std::variant<CoreCoord, CoreRange, CoreRangeSet>;

namespace sparse_common {

std::shared_ptr<Buffer> MakeBuffer(IDevice* device, uint32_t size, uint32_t page_size, bool sram = false);
std::shared_ptr<Buffer> MakeBuffer(IDevice* device, uint32_t n_tiles, size_t element_size, bool sram = false);

CBHandle MakeCircularBuffer(
    Program& program, const CoreSpec& core, tt::CBIndex cb, uint32_t size, uint32_t page_size, tt::DataFormat format);

CBHandle MakeCircularBufferFP32(Program& program, const CoreSpec& core, tt::CBIndex cb, uint32_t n_tiles);

} // namespace sparse_common
