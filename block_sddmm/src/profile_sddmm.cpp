#include <common/TracyColor.hpp>
#include <cstdio>
#include <string>
#include "../inc/include_me.hpp"
#include "../inc/profiling_suite.hpp"
#include "../inc/host_code.hpp"
#include "../inc/host_code/sddmm_zone_config.hpp"

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

using namespace bsr_sddmm_host_code;
using namespace sddmm_profiling_suite;

void profile_test(
    SDDMMHostCodeFunctionPtr host_func,
    bsr_matrix<bfloat16>& mask,
    dense_matrix<bfloat16>& c,
    dense_matrix<bfloat16>& d,
    std::string& test_name,
    int num_iters = 10);

using SDDMMHostCodeRegistryType = std::pair<SDDMMHostCodeFunctionPtr, std::string>;

void capture_profile(
    int host_code_num,
    int test_num,
    ProfileCaseFunctionPtr *Registry,
    std::string registry_name,
    SDDMMHostCodeRegistryType* hc_registry,
    const std::string& output_subdir,
    int num_iters = 10);

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: profile_sddmm <test_num> <host_code_num> <registry_number>" << std::endl;
        return 1;
    }

    int test_num = std::stoi(argv[1]);
    int host_code_num = std::stoi(argv[2]);
    int registry_number = std::stoi(argv[3]);
    std::string output_subdir = argc > 4 ? argv[4] : "sddmm_profiles/opt_noc";

    SDDMMHostCodeRegistryType* hc_registry = HostCodeRegistryProfiling;

    ProfileCaseFunctionPtr *Registry = nullptr;
    std::string registry_name = "";
    switch (registry_number) {
        case 0:
            Registry = ProfileCaseRegistry;
            registry_name = "SDDMMProfileSuite";
            break;
        case 1:
            Registry = ProfileSweepNRegistry;
            registry_name = "SDDMMSweepN";
            break;
        case 2:
            Registry = ProfileSweepDensityRegistry;
            registry_name = "SDDMMSweepDensity";
            break;
        case 3:
            Registry = ProfileSweepKRegistry;
            registry_name = "SDDMMSweepK";
            break;
        case 4:
            Registry = ProfileSweepBlockSizeRegistry;
            registry_name = "SDDMMSweepBlockSize";
            break;
    }

    capture_profile(host_code_num, test_num, Registry, registry_name, hc_registry, output_subdir, 10);
}

void capture_profile(int host_code_num, int test_num, ProfileCaseFunctionPtr *Registry, std::string registry_name, SDDMMHostCodeRegistryType* hc_registry, const std::string& output_subdir, int num_iters){
    SDDMMHostCodeFunctionPtr host_function = hc_registry[host_code_num].first;
    std::string host_function_name = hc_registry[host_code_num].second;
    auto [mask, c, d, test_name] = Registry[test_num]();

    auto zone_defines = sddmm_zone_config::get_zone_defines();

    std::string disabled_zones = zone_defines.empty() ? "" : "_Disable_";
    for (auto it = zone_defines.begin(); it != zone_defines.end(); it++){
        std::string zone_name = it->first;
        disabled_zones += "_" + zone_name;
    }

    char buf[1000];
    size_t n = sprintf(buf, "/home/user/tt-metal/%s/traces/%s/%s/", output_subdir.c_str(), registry_name.c_str(), host_function_name.c_str());
    std::string trace_directory(buf, n);
    std::string trace_file_location = trace_directory + test_name + disabled_zones + ".tracy";

    n = sprintf(buf, "mkdir -p %s", trace_directory.c_str());
    std::string mkdir_command(buf, n);

    n = sprintf(buf, "nohup ./capture-release -f -o %s &", trace_file_location.c_str());
    std::string capture_trace_command(buf, n);

    std::system(mkdir_command.c_str());
    std::system(capture_trace_command.c_str());

    profile_test(host_function, mask, c, d, test_name, num_iters);
}

void profile_test(
        SDDMMHostCodeFunctionPtr host_func,
        bsr_matrix<bfloat16>& mask,
        dense_matrix<bfloat16>& c,
        dense_matrix<bfloat16>& d,
        std::string& test_name,
        int num_iters) {
    constexpr int device_id = 0;
    IDevice* device = CreateDevice(device_id);
    std::cout << "Checking for tracy profiler connection to device" << std::endl;
    while (!tracy::GetProfiler().IsConnected()){
        std::cout << "Waiting for tracy profiler to connect to device" << std::endl;
        sleep(1);
    }

    {
        ZoneScopedNC("Post-device setup", tracy::Color::DarkOliveGreen);
        uint32_t M = mask.H;
        uint32_t N = mask.W;
        uint32_t K = c.W;
        uint32_t R = mask.R;
        uint32_t C_block = mask.C;

        bsr_matrix<bfloat16> output;

        host_func(mask, c, d, output, M, N, K, R, C_block, 1, device);
    }

    tt_metal::detail::ReadDeviceProfilerResults(device);
    CloseDevice(device);
}
