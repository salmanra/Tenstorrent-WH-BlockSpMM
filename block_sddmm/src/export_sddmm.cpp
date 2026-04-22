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

#include <cstdlib>

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

using namespace bsr_sddmm_host_code;
using namespace sddmm_profiling_suite;

using SDDMMHostCodeRegistryType = std::pair<SDDMMHostCodeFunctionPtr, std::string>;

void export_to_csv(int host_code_num, int test_num, ProfileCaseFunctionPtr *Registry, std::string registry_name, SDDMMHostCodeRegistryType* hc_registry, const std::string& output_subdir = "sddmm_profiles/opt_noc"){
    SDDMMHostCodeFunctionPtr host_function = hc_registry[host_code_num].first;
    std::string host_function_name = hc_registry[host_code_num].second;
    auto [mask, c, d, test_name] = Registry[test_num]();

    auto zone_defines = sddmm_zone_config::get_zone_defines();

    std::string disabled_zones = zone_defines.empty() ? "" : "_Disable_";
    for (auto it = zone_defines.begin(); it != zone_defines.end(); it++){
        std::string zone_name = it->first;
        disabled_zones += "_" + zone_name;
    }

    std::string test_file_name = test_name + disabled_zones;

    char buf[1000];
    const char* tt_metal_home = std::getenv("TT_METAL_HOME");
    if (!tt_metal_home) {
        std::cerr << "TT_METAL_HOME must be set" << std::endl;
        std::exit(1);
    }
    size_t n = sprintf(buf, "%s/%s/traces/%s/%s/", tt_metal_home, output_subdir.c_str(), registry_name.c_str(), host_function_name.c_str());
    std::string trace_directory(buf, n);
    std::string trace_file_location = trace_directory + test_file_name + ".tracy";

    n = sprintf(buf, "%s/%s/csvs/%s/%s/", tt_metal_home, output_subdir.c_str(), registry_name.c_str(), host_function_name.c_str());
    std::string csv_directory(buf);
    std::string csv_file_location = csv_directory + test_file_name + ".csv";

    n = sprintf(buf, "mkdir -p %s", csv_directory.c_str());
    std::string csv_mkdir_command(buf, n);

    n = sprintf(buf, "./csvexport-release %s > %s", trace_file_location.c_str(), csv_file_location.c_str());
    std::string csvexport_command(buf);

    std::system(csv_mkdir_command.c_str());
    std::system(csvexport_command.c_str());

    // Log mask (sparse BSR)
    std::string mask_log_file = csv_directory + test_file_name + "_mask.log";
    {
        std::ofstream os_mask(mask_log_file);
        mask.pretty_print(os_mask);
        os_mask << "Block Size (R x C): " << mask.R << " x " << mask.C << std::endl;
    }  // close before Python appends via >>

    // Log C (dense M×K)
    std::string c_log_file = csv_directory + test_file_name + "_C.log";
    std::ofstream os_c(c_log_file);
    c.pretty_print(os_c);
    os_c << "Dimensions: " << c.H << " x " << c.W << std::endl;

    // Log D (dense K×N)
    std::string d_log_file = csv_directory + test_file_name + "_D.log";
    std::ofstream os_d(d_log_file);
    d.pretty_print(os_d);
    os_d << "Dimensions: " << d.H << " x " << d.W << std::endl;

    // Compute device TFLOPs via profiler log.
    // Path is relative to the working dir that block_sparse binaries are invoked from (repo root).
    n = sprintf(buf, "python tt_metal/programming_examples/Tenstorrent-WH-BlockSpMM/GEMM_profiling/read_sddmm_profiler.py --nblocks %zu --R %zu --C %zu --N %zu --M %zu --K %zu >> %s ",
                mask.nblocks, mask.R, mask.C, mask.W, mask.H, c.W, mask_log_file.c_str());
    std::string python_TFLOPs_command(buf);
    std::system(python_TFLOPs_command.c_str());
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: export_sddmm <test_num> <host_code_num> <registry_number>" << std::endl;
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

    export_to_csv(host_code_num, test_num, Registry, registry_name, hc_registry, output_subdir);

    return 0;
}
