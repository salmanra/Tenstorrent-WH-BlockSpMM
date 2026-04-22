#pragma once
#include <cstdlib>
#include <map>
#include <string>

namespace spmm_zone_config {

// Zone flag names — must match #ifndef names in kernel files.
inline constexpr const char* ZONE_FLAGS[] = {
    "PROFILE_READ_IN0",
    "PROFILE_WAIT_IN0",
    "PROFILE_WRITE_OUT",
    "PROFILE_READ_IN1",
    "PROFILE_COMPUTE",
};

// Build defines map from environment variables.
// Only includes flags explicitly set to "0" — unset flags use kernel defaults (1 = enabled).
// This keeps the defines map empty for normal profiling runs, avoiding unnecessary recompilation.
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

} // namespace spmm_zone_config
