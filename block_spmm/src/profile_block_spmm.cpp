#include <common/TracyColor.hpp>
#include <cstdio>
#include <string>
#include "../inc/include_me.hpp"
#include "../inc/profiling_suite.hpp"
#include "../inc/host_code.hpp"
#include "../inc/host_code/spmm_zone_config.hpp"

#include <system_error>
#include <tracy/Tracy.hpp>
#include "hostdevcommon/profiler_common.h"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include "hostdevcommon/profiler_common.h"

#include <cstdlib>
#include <unistd.h>

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

using namespace bsr_host_code;
using namespace profiling_suite;

using HostCodeRegistryType = std::pair<HostCodeFunctionPtr, std::string>;

void profile_test(
    HostCodeFunctionPtr host_func,
    bsr_matrix<bfloat16>& a,
    dense_matrix<bfloat16>& b,
    std::string& test_name,
    int num_iters = 10);

void capture_profile(
    int host_code_num,
    int test_num,
    ProfileCaseFunctionPtr *Registry,
    std::string registry_name,
    HostCodeRegistryType* hc_registry,
    const std::string& output_dir,
    int num_iters = 10);

int main(int argc, char** argv) {
    bool run_all_profiles = argc > 1 ? std::string(argv[1]) == "all" : true;
    bool run_all_host_codes = argc > 2 ? std::string(argv[2]) == "all" : true;
    int test_num = 0;
    int host_code_num = 0;
    if (!run_all_profiles)
        test_num = argc > 1 ? std::stoi(argv[1]) : 0;
    if (!run_all_host_codes)
        host_code_num = argc > 2 ? std::stoi(argv[2]) : 0;

    int registry_number = argc > 3 ? std::stoi(argv[3]) : 0;
    std::string output_dir = argc > 4 ? argv[4] : "profiles_sc26_april5";

    HostCodeRegistryType* hc_registry = HostCodeRegistryProfiling;
    int num_host_programs = sizeof(HostCodeRegistryProfiling) / sizeof(HostCodeRegistryProfiling[0]);

    if (registry_number < 0 || registry_number >= NUM_REGISTRIES) {
        fprintf(stderr, "Invalid registry number %d (valid: 0-%d)\n", registry_number, NUM_REGISTRIES - 1);
        return 1;
    }

    ProfileCaseFunctionPtr *Registry = Registries[registry_number];
    std::string registry_name = RegistryNames[registry_number];
    int num_profiles = RegistrySizes[registry_number];

    if (run_all_profiles && !run_all_host_codes) {
        for (int i = 0; i < num_profiles; i++) {
            capture_profile(host_code_num, i, Registry, registry_name, hc_registry, output_dir, 10);
        }
    } else if (run_all_profiles && run_all_host_codes) {
        for (int i = 0; i < num_profiles; i++) {
            for (int j = 0; j < num_host_programs; j++) {
                capture_profile(j, i, Registry, registry_name, hc_registry, output_dir, 10);
            }
        }
    } else {
        capture_profile(host_code_num, test_num, Registry, registry_name, hc_registry, output_dir, 10);
    }
}

void capture_profile(int host_code_num, int test_num, ProfileCaseFunctionPtr *Registry, std::string registry_name, HostCodeRegistryType* hc_registry, const std::string& output_dir, int num_iters) {
    HostCodeFunctionPtr host_function = hc_registry[host_code_num].first;
    std::string host_function_name = hc_registry[host_code_num].second;
    auto [a, b, test_name] = Registry[test_num]();

    auto zone_defines = spmm_zone_config::get_zone_defines();

    std::string disabled_zones = zone_defines.empty() ? "" : "_Disable_";
    for (auto it = zone_defines.begin(); it != zone_defines.end(); it++) {
        std::string zone_name = it->first;
        disabled_zones += "_" + zone_name;
    }

    char buf[1000];
    size_t n = sprintf(buf, "/home/user/tt-metal/%s/bsr/%s/%s/", output_dir.c_str(), registry_name.c_str(), host_function_name.c_str());
    std::string trace_directory(buf, n);
    std::string trace_file_location = trace_directory + test_name + disabled_zones + ".tracy";

    n = sprintf(buf, "mkdir -p %s", trace_directory.c_str());
    std::string mkdir_command(buf, n);

    n = sprintf(buf, "nohup ./capture-release -f -o %s &", trace_file_location.c_str());
    std::string capture_trace_command(buf, n);

    std::system(mkdir_command.c_str());
    std::system(capture_trace_command.c_str());

    profile_test(host_function, a, b, test_name, num_iters);
}

void profile_test(
        HostCodeFunctionPtr host_func,
        bsr_matrix<bfloat16>& a,
        dense_matrix<bfloat16>& b,
        std::string& test_name,
        int num_iters) {
    constexpr int device_id = 0;
    IDevice* device = CreateDevice(device_id);
    std::cout << "Checking for tracy profiler connection to device" << std::endl;
    while (!tracy::GetProfiler().IsConnected()) {
        std::cout << "Waiting for tracy profiler to connect to device" << std::endl;
        sleep(1);
    }

    {
        ZoneScopedNC("Post-device setup", tracy::Color::DarkOliveGreen);
        uint32_t M = a.H;
        uint32_t N = b.W;
        uint32_t K = a.W;
        uint32_t R = a.R;
        uint32_t C = a.C;
        uint32_t nblocks = a.nblocks;

        dense_matrix<float> tmp(M, N, 0.0f);
        dense_matrix<bfloat16> output = tmp.bfloat16_cast();

        host_func(a, b, output, false, nblocks, M, N, K, R, C, 1, device);
    }

    tt_metal::detail::ReadDeviceProfilerResults(device);
    CloseDevice(device);
}
