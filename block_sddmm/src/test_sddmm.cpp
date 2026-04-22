
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

int run_device_test(
    SDDMMHostCodeFunctionPtr host_func,
    const std::string& host_func_name,
    bsr_matrix<bfloat16>& mask,
    dense_matrix<bfloat16>& c,
    dense_matrix<bfloat16>& d,
    const std::string& test_name,
    IDevice* device);

int run_cpu_test(
    bsr_matrix<bfloat16>& mask,
    dense_matrix<bfloat16>& c,
    dense_matrix<bfloat16>& d,
    const std::string& test_name);

void print_usage() {
    fprintf(stderr, "Usage: test_sddmm <test_num> <host_code_num> <registry_number>\n");
    fprintf(stderr, "  Runs device kernel + CPU kernel and compares for correctness.\n");
    fprintf(stderr, "  test_num:       index into the selected registry (-1 = run all)\n");
    fprintf(stderr, "  host_code_num:  index into HostCodeRegistryVerbose (-1 = CPU-only)\n");
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
    bool cpu_only = (host_code_num < 0);

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

    IDevice* device = nullptr;
    if (!cpu_only) {
        device = CreateDevice(0);
    }

    int num_passed = 0;
    int num_failed = 0;

    for (size_t i = start; i < end; i++) {
        auto [mask, c, d, test_name] = Registry[i]();

        if (test_num >= 0) {
            mask.pretty_print();
        }

        if (cpu_only) {
            printf("[%s] Test %zu: %s (CPU-only) ... ", registry_name.c_str(), i, test_name.c_str());
            int rc = run_cpu_test(mask, c, d, test_name);
            if (rc == 0) { num_passed++; } else { num_failed++; }
        } else {
            std::string hf_name = hc_registry[host_code_num].second;
            printf("[%s] Test %zu: %s, host_code: %s ... ",
                   registry_name.c_str(), i, test_name.c_str(), hf_name.c_str());
            int rc = run_device_test(
                hc_registry[host_code_num].first, hf_name,
                mask, c, d, test_name, device);
            if (rc == 0) { num_passed++; } else { num_failed++; }
        }
    }

    if (device) {
        CloseDevice(device);
    }

    printf("\n--- SDDMM Test Summary ---\n");
    printf("Registry: %s | Host code: %s\n",
           registry_name.c_str(),
           cpu_only ? "CPU-only" : hc_registry[host_code_num].second.c_str());
    printf("Passed: %d / %d\n", num_passed, num_passed + num_failed);
    if (num_failed > 0) {
        printf("Failed: %d\n", num_failed);
    }
    return num_failed > 0 ? 1 : 0;
}

int run_cpu_test(
    bsr_matrix<bfloat16>& mask,
    dense_matrix<bfloat16>& c,
    dense_matrix<bfloat16>& d,
    const std::string& test_name) {

    bsr_matrix<bfloat16> result = mask.sddmm(c, d);

    bool pass = true;
    if (result.H != mask.H || result.W != mask.W || result.R != mask.R || result.C != mask.C) {
        printf("FAIL (dimension mismatch)\n");
        pass = false;
    }
    if (result.nblocks != mask.nblocks) {
        printf("FAIL (nblocks=%zu, expected %zu)\n", result.nblocks, mask.nblocks);
        pass = false;
    }
    if (result.indptr != mask.indptr) {
        printf("FAIL (indptr mismatch)\n");
        pass = false;
    }
    if (result.indices != mask.indices) {
        printf("FAIL (indices mismatch)\n");
        pass = false;
    }
    if (result.data.size() != mask.nblocks * mask.R * mask.C) {
        printf("FAIL (data size=%zu, expected %zu)\n",
               result.data.size(), mask.nblocks * mask.R * mask.C);
        pass = false;
    }

    if (pass) {
        printf("PASS (%zu blocks)\n", result.nblocks);
    }
    return pass ? 0 : 1;
}

int run_device_test(
    SDDMMHostCodeFunctionPtr host_func,
    const std::string& host_func_name,
    bsr_matrix<bfloat16>& mask,
    dense_matrix<bfloat16>& c,
    dense_matrix<bfloat16>& d,
    const std::string& test_name,
    IDevice* device) {

    uint32_t M = mask.H;
    uint32_t N = mask.W;
    uint32_t K = c.W;
    uint32_t R = mask.R;
    uint32_t C_block = mask.C;

    // CPU reference (before tilizing inputs)
    bsr_matrix<bfloat16> expected = mask.sddmm(c, d);

    // Tilize mask: each BSR block of R x C_block independently
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

    // Device run
    bsr_matrix<bfloat16> output;
    host_func(mask, c, d, output, M, N, K, R, C_block, 1, device);

    // Untilize output: each BSR block of R x C_block independently
    {
        std::vector<bfloat16> untilized;
        untilized.reserve(output.data.size());
        size_t block_elems = R * C_block;
        for (size_t b = 0; b < output.nblocks; b++) {
            auto begin = output.data.begin() + b * block_elems;
            auto end = begin + block_elems;
            std::vector<bfloat16> block_data(begin, end);
            block_data = untilize_nfaces(block_data, R, C_block);
            untilized.insert(untilized.end(), block_data.begin(), block_data.end());
        }
        output.data = std::move(untilized);
    }

    // Compare via PCC on dense representations
    dense_matrix<bfloat16> output_dense = output.to_dense();
    dense_matrix<bfloat16> expected_dense = expected.to_dense();

    float pcc = check_bfloat16_vector_pcc(output_dense.data, expected_dense.data);

    if (pcc > 0.99f || (output_dense.data.size() == 0 && expected_dense.data.size() == 0)) {
        printf("PASS (PCC=%.6f, %zu blocks)\n", pcc, output.nblocks);
        return 0;
    } else {
        printf("FAIL (PCC=%.6f, %zu blocks)\n", pcc, output.nblocks);
        return 1;
    }
}
