#pragma once

#include <cstdlib>
#include <map>
#include <string>

namespace sddmm_zone_config {

inline constexpr const char* ZONE_FLAGS[] = {
    "PROFILE_READ_BC",
    "PROFILE_READ_D",
    "PROFILE_WRITE_OUT",
    "PROFILE_COMPUTE",
};

inline std::map<std::string, std::string> get_zone_defines() {
    std::map<std::string, std::string> defines;
    for (const char* name : ZONE_FLAGS) {
        const char* val = std::getenv(name);
        if (val && std::string(val) == "0") {
            defines[name] = "0";
        }
    }
    return defines;
}

} // namespace sddmm_zone_config
