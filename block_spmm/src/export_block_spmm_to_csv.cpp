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

#include <cstdlib>

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

using namespace bsr_host_code;
using namespace profiling_suite;

using HostCodeRegistryType = std::pair<HostCodeFunctionPtr, std::string>;

void export_to_csv(int host_code_num, int test_num, ProfileCaseFunctionPtr *Registry, std::string registry_name, HostCodeRegistryType* hc_registry, const std::string& output_dir) {
    HostCodeFunctionPtr host_function = hc_registry[host_code_num].first;
    std::string host_function_name = hc_registry[host_code_num].second;
    auto [a, b, test_name] = Registry[test_num]();

    auto zone_defines = spmm_zone_config::get_zone_defines();

    std::string disabled_zones = zone_defines.empty() ? "" : "_Disable_";
    for (auto it = zone_defines.begin(); it != zone_defines.end(); it++) {
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
    size_t n = sprintf(buf, "%s/%s/bsr/%s/%s/", tt_metal_home, output_dir.c_str(), registry_name.c_str(), host_function_name.c_str());
    std::string trace_directory(buf, n);
    std::string trace_file_location = trace_directory + test_file_name + ".tracy";

    n = sprintf(buf, "%s/%s/csvs/%s/%s/", tt_metal_home, output_dir.c_str(), registry_name.c_str(), host_function_name.c_str());
    std::string csv_directory(buf);
    std::string csv_file_location = csv_directory + test_file_name + ".csv";

    n = sprintf(buf, "mkdir -p %s", csv_directory.c_str());
    std::string csv_mkdir_command(buf, n);

    n = sprintf(buf, "./csvexport-release %s > %s", trace_file_location.c_str(), csv_file_location.c_str());
    std::string csvexport_command(buf);

    std::system(csv_mkdir_command.c_str());
    std::system(csvexport_command.c_str());

    // Write sparse and dense matrix metadata logs
    std::string sparse_log_file = csv_directory + test_file_name + "_sparse.log";
    std::ofstream os_sparse(sparse_log_file);

    std::string dense_log_file = csv_directory + test_file_name + "_dense.log";
    std::ofstream os_dense(dense_log_file);

    a.pretty_print(os_sparse);
    b.pretty_print(os_dense);
    os_dense << "Block Size (R x C): " << a.R << " x " << a.C << std::endl;

    uint32_t Nt = b.W / TILE_WIDTH;
    uint32_t Rt = a.R / TILE_HEIGHT;
    uint32_t Ct = a.C / TILE_WIDTH;

    tt::DataFormat indexing_data_format = tt::DataFormat::Int32;
    uint32_t indexing_tile_size = detail::TileSize(indexing_data_format);
    uint32_t indptr_buf_size = sizeof(int) * a.indptr.size();
    indptr_buf_size = indexing_tile_size * ((indexing_tile_size - 1 + indptr_buf_size) / indexing_tile_size);
    uint32_t col_idx_buf_size = sizeof(int) * a.indices.size();
    col_idx_buf_size = indexing_tile_size * ((indexing_tile_size - 1 + col_idx_buf_size) / indexing_tile_size);
    uint32_t num_tiles_indexing = indptr_buf_size / indexing_tile_size + col_idx_buf_size / indexing_tile_size;

    uint32_t nnz_rows = 0;
    for (uint32_t i = 0; i + 1 < a.indptr.size(); i++) {
        if (a.indptr[i + 1] - a.indptr[i] > 0) nnz_rows++;
    }

    constexpr uint32_t num_cores_x = 8;
    constexpr uint32_t num_cores_y = 8;

    uint32_t in1_block_w = get_Npc_from_BSR_block_size(Nt, Rt, Ct, num_cores_x, num_cores_y, num_tiles_indexing, nnz_rows);
    os_dense << "Dense block width (in1_block_w): " << in1_block_w << " tiles"
             << " (" << in1_block_w * TILE_WIDTH << " columns)" << std::endl;

    n = sprintf(buf, "python tt_metal/programming_examples/Tenstorrent-WH-BlockSpMM/GEMM_profiling/read_spmm_profiler.py --nblocks %zu --R %zu --C %zu --N %zu --M %zu --K %zu >> %s ", a.nblocks, a.R, a.C, b.W, a.H, a.W, sparse_log_file.c_str());
    std::string python_TFLOPs_command(buf);
    std::system(python_TFLOPs_command.c_str());
}

int main(int argc, char** argv) {
    bool export_all_profiles = argc > 1 ? std::string(argv[1]) == "all" : true;
    bool export_all_host_codes = argc > 2 ? std::string(argv[2]) == "all" : true;
    int test_num = 0;
    int host_code_num = 0;
    if (!export_all_profiles)
        test_num = argc > 1 ? std::stoi(argv[1]) : 0;
    if (!export_all_host_codes)
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

    if (export_all_profiles && !export_all_host_codes) {
        for (int i = 0; i < num_profiles; i++) {
            export_to_csv(host_code_num, i, Registry, registry_name, hc_registry, output_dir);
        }
    } else if (export_all_profiles && export_all_host_codes) {
        for (int i = 0; i < num_profiles; i++) {
            for (int j = 0; j < num_host_programs; j++) {
                export_to_csv(j, i, Registry, registry_name, hc_registry, output_dir);
            }
        }
    } else {
        export_to_csv(host_code_num, test_num, Registry, registry_name, hc_registry, output_dir);
    }

    return 0;
}
