
#include <cstdio>
#include <cstdlib>
#include <string>
#include "../inc/include_me.hpp"
#include "../inc/test_suite.hpp"
#include "../inc/profiling_suite.hpp"
#include "../inc/host_code.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/device.hpp>

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

using namespace bsr_sddmm_test_suite;
using namespace bsr_sddmm_host_code;
using namespace sddmm_profiling_suite;

// Both TestFunctionPtr and ProfileCaseFunctionPtr are the same underlying type
using TestCaseFunctionPtr = ProfileCaseFunctionPtr;
using HostCodeEntry = std::pair<SDDMMHostCodeFunctionPtr, std::string>;

void print_usage() {
    fprintf(stderr, "Usage: run_sddmm <test_num> <host_code_num> <registry_number>\n");
    fprintf(stderr, "  Runs device kernel only (no CPU reference). For debugging deadlocks.\n");
    fprintf(stderr, "  test_num:       index into the selected registry (-1 = run all)\n");
    fprintf(stderr, "  host_code_num:  index into HostCodeRegistryVerbose\n");
    fprintf(stderr, "  registry_number:\n");
    fprintf(stderr, "    0 = ProfileCaseRegistry\n");
    fprintf(stderr, "    1 = ProfileSweepNRegistry\n");
    fprintf(stderr, "    2 = ProfileSweepDensityRegistry\n");
    fprintf(stderr, "    3 = ProfileSweepKRegistry\n");
    fprintf(stderr, "    4 = ProfileSweepBlockSizeRegistry\n");
    fprintf(stderr, "    5 = TestRegistry (test_suite.hpp)\n");
}

struct RegistryInfo {
    TestCaseFunctionPtr* entries;
    size_t count;
    std::string name;
};

RegistryInfo get_registry(int registry_number) {
    switch (registry_number) {
        case 0: return {ProfileCaseRegistry,
                        sizeof(ProfileCaseRegistry) / sizeof(ProfileCaseRegistry[0]),
                        "SDDMMProfileSuite"};
        case 1: return {ProfileSweepNRegistry,
                        sizeof(ProfileSweepNRegistry) / sizeof(ProfileSweepNRegistry[0]),
                        "SDDMMSweepN"};
        case 2: return {ProfileSweepDensityRegistry,
                        sizeof(ProfileSweepDensityRegistry) / sizeof(ProfileSweepDensityRegistry[0]),
                        "SDDMMSweepDensity"};
        case 3: return {ProfileSweepKRegistry,
                        sizeof(ProfileSweepKRegistry) / sizeof(ProfileSweepKRegistry[0]),
                        "SDDMMSweepK"};
        case 4: return {ProfileSweepBlockSizeRegistry,
                        sizeof(ProfileSweepBlockSizeRegistry) / sizeof(ProfileSweepBlockSizeRegistry[0]),
                        "SDDMMSweepBlockSize"};
        case 5: return {TestRegistry,
                        sizeof(TestRegistry) / sizeof(TestRegistry[0]),
                        "TestRegistry"};
        default:
            fprintf(stderr, "Unknown registry_number: %d\n", registry_number);
            exit(1);
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        print_usage();
        return 1;
    }

    int test_num = std::stoi(argv[1]);
    int host_code_num = std::stoi(argv[2]);
    int registry_number = std::stoi(argv[3]);

    auto [Registry, registry_size, registry_name] = get_registry(registry_number);

    HostCodeEntry* hc_registry = HostCodeRegistryVerbose;
    std::string hf_name = hc_registry[host_code_num].second;
    SDDMMHostCodeFunctionPtr host_func = hc_registry[host_code_num].first;

    // Determine test range
    size_t start = 0, end = registry_size;
    if (test_num >= 0) {
        start = (size_t)test_num;
        end = start + 1;
    }
    if (start >= registry_size) {
        fprintf(stderr, "test_num %d out of range (registry has %zu entries)\n", test_num, registry_size);
        return 1;
    }

    IDevice* device = CreateDevice(0);

    for (size_t i = start; i < end; i++) {
        auto [mask, c, d, test_name] = Registry[i]();

        uint32_t M = mask.H;
        uint32_t N = mask.W;
        uint32_t K = c.W;
        uint32_t R = mask.R;
        uint32_t C_block = mask.C;

        printf("[%s] Test %zu: %s, host_code: %s\n",
               registry_name.c_str(), i, test_name.c_str(), hf_name.c_str());

        if (test_num >= 0) {
            mask.pretty_print();
        }

        printf("Running SDDMM: mask(%zux%zu, %zu blocks) @ (C(%zux%zu) x D(%zux%zu))\n",
               mask.H, mask.W, mask.nblocks, c.H, c.W, d.H, d.W);

        // Tilize inputs for device
        {
            std::vector<bfloat16> tilized_mask;
            tilized_mask.reserve(mask.data.size());
            size_t block_elems = R * C_block;
            for (size_t b = 0; b < mask.nblocks; b++) {
                auto begin = mask.data.begin() + b * block_elems;
                auto end = begin + block_elems;
                std::vector<bfloat16> block_data(begin, end);
                block_data = tilize_nfaces(block_data, R, C_block);
                tilized_mask.insert(tilized_mask.end(), block_data.begin(), block_data.end());
            }
            mask.data = std::move(tilized_mask);
        }
        c.data = tilize_nfaces(c.data, M, K);
        d.data = tilize_nfaces(d.data, K, N);

        // Run device kernel
        bsr_matrix<bfloat16> output;
        host_func(mask, c, d, output, M, N, K, R, C_block, 1, device);

        printf("COMPLETE: output %zux%zu with %zu blocks\n\n", output.H, output.W, output.nblocks);
    }

    CloseDevice(device);
    return 0;
}
