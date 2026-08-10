#pragma once

#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt_stl/span.hpp>
#include <umd/device/cluster.hpp>
#include <umd/device/cluster_descriptor.hpp>

namespace bspmm_compat {

inline tt::ChipId select_blackhole_device_id() {
    static const tt::ChipId selected = [] {
        if (const char* override_id = std::getenv("TT_METAL_BSPMM_DEVICE_ID")) {
            return static_cast<tt::ChipId>(std::stoi(override_id));
        }

        auto cluster = tt::umd::Cluster::create_cluster_descriptor();
        TT_FATAL(cluster && cluster->get_number_of_chips() > 0, "No Tenstorrent devices detected");

        std::ostringstream detected;
        for (tt::ChipId chip : cluster->get_chips_local_first(cluster->get_all_chips())) {
            detected << chip << "(arch=" << static_cast<int>(cluster->get_arch(chip))
                     << ", mmio=" << cluster->is_chip_mmio_capable(chip) << ") ";
            if (cluster->is_chip_mmio_capable(chip) && cluster->get_arch(chip) == tt::ARCH::BLACKHOLE) {
                return chip;
            }
        }

        TT_FATAL(false, "No Blackhole PCIe device found. Detected chips: {}", detected.str());
    }();
    return selected;
}

template <typename DType>
inline void write_buffer_blocking(const std::shared_ptr<tt::tt_metal::Buffer>& buffer, const DType* data) {
    tt::tt_metal::detail::WriteToBuffer(
        *buffer, ttsl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(data), buffer->size()));
}

// Buffer::view() keeps the root base address in address() and stores the region offset
// separately, but the slow-dispatch interleaved read path derives device addresses from
// address() alone, so regioned reads silently start at the buffer base. Read whole.
template <typename DType>
inline void read_buffer_blocking(const std::shared_ptr<tt::tt_metal::Buffer>& buffer, DType* data) {
    tt::tt_metal::detail::ReadFromBuffer(*buffer, reinterpret_cast<uint8_t*>(data));
}

inline void launch_program_blocking(tt::tt_metal::IDevice* device, tt::tt_metal::Program& program) {
    tt::tt_metal::detail::LaunchProgram(device, program, true);
}

inline tt::tt_metal::IDevice* create_blackhole_device_slow_dispatch() {
    const tt::ChipId device_id = select_blackhole_device_id();
    // Slow-dispatch bring-up must not initialize fast-dispatch firmware; teardown assumes fast dispatch.
    auto devices = tt::tt_metal::detail::CreateDevices(
        {device_id},
        1,
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        tt::tt_metal::DispatchCoreConfig{},
        {},
        DEFAULT_WORKER_L1_SIZE,
        true,
        false,
        false);
    TT_FATAL(devices.contains(device_id), "Failed to create Blackhole device {}", device_id);
    return devices.at(device_id);
}

inline void close_device_slow_dispatch(tt::tt_metal::IDevice* device) {
    if (device == nullptr) {
        return;
    }
    tt::tt_metal::detail::CloseDevices({{device->id(), device}});
}

}  // namespace bspmm_compat
