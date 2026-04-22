
#include <cstdio>
#include <string>
#include "../inc/include_me.hpp"
#include "../inc/test_suite.hpp"
#include "../inc/profiling_suite.hpp"
#include "../inc/host_code.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

using namespace bsr_test_suite;
using namespace bsr_host_code;
using namespace profiling_suite;

#define ESC "\033["
#define BLACK_BKG "106"
#define GREEN_TXT "118"
#define RED_TXT "196"
#define RESET "\033[m"

// Print to the *original* console, regardless of where stdout is redirected
void console_printf(const char* fmt, ...) {
    static int console_fd = -1;
    if (console_fd == -1) {
        console_fd = ::open("/dev/tty", O_WRONLY | O_CLOEXEC);
    }
    if (console_fd == -1) return;

    va_list ap;
    va_start(ap, fmt);
    ::vdprintf(console_fd, fmt, ap);
    va_end(ap);
}

struct TestResult {
    std::string test_name;
    float pearson;
    bool all_close;
};

TestResult run_test(
    HostCodeFunctionPtr host_func,
    bsr_matrix<bfloat16>& a,
    dense_matrix<bfloat16>& b,
    std::string& test_name,
    bool emit_output = false) {

    constexpr int device_id = 0;
    IDevice* device = CreateDevice(device_id);

    // matmul params setup
    uint32_t M = a.H;
    uint32_t N = b.W;
    uint32_t K = a.W;
    // block params setup
    uint32_t R = a.R;
    uint32_t C = a.C;
    uint32_t nblocks = a.nblocks;
    uint32_t block_matrix_height = M / R;

    uint32_t Rt = R / TILE_HEIGHT;
    uint32_t Ct = C / TILE_WIDTH;

    // initialize output_data
    dense_matrix<float> tmp(M, N, 0.0f);
    dense_matrix<bfloat16> output = tmp.bfloat16_cast();

    a.pretty_print();

    // run sequential spmm
    dense_matrix<bfloat16> golden = a.omp_spmm_bf16(b);

    // tilize input data
    a.data = tilize_nfaces(a.data, R, C);
    b.data = tilize_nfaces(b.data, K, N);

    host_func(a, b, output, false, nblocks, M, N, K, R, C, 1, device);

    if (emit_output) {
        std::string local_path = "/home/user/tt-metal/tt_metal/programming_examples/rahmy/SC26_submission/block_spmm/" + test_name;

        std::filesystem::create_directory(local_path);
        std::string output_file = local_path + "/output.txt";
        std::ofstream out(output_file);
        if (!out.is_open()) {
            TT_THROW("Failed to open output file: {}", output_file);
        }
        for (size_t i = 0; i < output.data.size(); i++) {
            out << output.data[i].to_float() << "\n";
        }
        out.close();

        log_info(tt::LogVerif, "Output written to {}", output_file);
        std::string golden_file = local_path + "/golden.txt";
        std::ofstream golden_out(golden_file);
        if (!golden_out.is_open()) {
            TT_THROW("Failed to open golden file: {}", golden_file);
        }

        golden.data = tilize_nfaces(golden.data, M, N);
        for (size_t i = 0; i < golden.data.size(); i++) {
            golden_out << golden.data[i].to_float() << "\n";
        }
        golden.data = untilize_nfaces(golden.data, M, N);
        golden_out.close();

        std::string bsr_file = local_path + "/bsr.txt";
        std::ofstream bsr_out(bsr_file);
        if (!bsr_out.is_open()) {
            TT_THROW("Failed to open bsr file: {}", bsr_file);
        }
        for (size_t i = 0; i < a.data.size(); i++) {
            bsr_out << a.data[i].to_float() << "\n";
        }
        bsr_out.close();

        std::string dense_file = local_path + "/dense.txt";
        std::ofstream dense_out(dense_file);
        if (!dense_out.is_open()) {
            TT_THROW("Failed to open dense file: {}", dense_file);
        }
        for (size_t i = 0; i < a.data.size(); i++) {
            dense_out << b.data[i].to_float() << "\n";
        }
        dense_out.close();
    }

    // untile output data
    output.data = untilize_nfaces(output.data, M, N);

    float pearson = check_bfloat16_vector_pcc(golden.data, output.data);

    bool all_close = golden.all_close_bfloat16(output);

    CloseDevice(device);

    return TestResult{test_name, pearson, all_close};
}

void add_and_run_test(
        HostCodeFunctionPtr host_func,
        TestFunctionPtr test_case,
        vector<TestResult> &results,
        bool emit_output = false) {
    auto [a, b, test_name] = test_case();
    results.push_back(run_test(host_func, a, b, test_name, emit_output));
}

bool print_and_assess_results(std::vector<TestResult> &test_results, std::string& host_code_function_name){
    console_printf("---------------------------------------------------------------------------------\n");
    console_printf("--- Test results ----------------------------------------------------------------\n");
    console_printf("---------------------------------------------------------------------------------\n");
    console_printf("--- Host code function: ");
    console_printf(host_code_function_name.c_str());
    console_printf("\n");
    console_printf("---------------------------------------------------------------------------------\n");

    std::string spacing = "  ";
    bool all_pass = true;
    char buf[12];
    uint32_t count = 0;
    for (auto &p : test_results) {
        bool pass = true;
        if (p.pearson < 0.99){
            pass = false;
            all_pass = false;
        }

        if (count >= 10 && count < 100)
            spacing = " ";
        if (count >= 100)
            spacing = "";

        std::string result = pass ? "PASS " : "FAIL ";
        sprintf(buf, "w/ PCC=%.2f", p.pearson);
        result += std::string(buf);
        console_printf("Test #");
        console_printf(std::to_string(count).c_str());
        console_printf(": ");
        console_printf(spacing.c_str());
        console_printf(result.c_str());
        console_printf(" ");
        console_printf(std::to_string(count).c_str());
        console_printf(" ");
        console_printf(spacing.c_str());
        console_printf(p.test_name.c_str());
        console_printf("\n");
        count++;
    }

    std::string result = all_pass ? "ALL PASS" : "SOME FAIL";

    console_printf("---------------------------------------------------------------------------------\n");
    console_printf(result.c_str());
    console_printf("\n");
    console_printf("---------------------------------------------------------------------------------\n");

    return all_pass;
}

void test_suite(HostCodeFunctionPtr host_function_ptr, std::string host_function_name, TestFunctionPtr* registry, size_t num_tests){
    console_printf("---------------------------------------------------------------------------------\n");
    console_printf("--- Test results ----------------------------------------------------------------\n");
    console_printf("---------------------------------------------------------------------------------\n");
    console_printf("--- Host code function: ");
    console_printf(host_function_name.c_str());
    console_printf("\n");
    console_printf("---------------------------------------------------------------------------------\n");
    console_printf("Testing ");
    console_printf(std::to_string(num_tests).c_str());
    console_printf(" tests\n");

    std::vector<TestResult> test_results;
    std::string spacing = "  ";
    char buf[12];
    bool all_pass = true;
    uint32_t count_pass = 0;
    for (size_t i = 0; i < num_tests; i++) {
        if (i >= 10 && i < 100)
            spacing = " ";
        if (i >= 100)
            spacing = "";
        add_and_run_test(host_function_ptr, registry[i], test_results);
        auto res = test_results[i];
        bool pass = res.pearson >= 0.99;
        count_pass += pass;
        if (!pass){
            all_pass = false;
        }
        std::string result = pass ? "PASS " : "FAIL ";

        sprintf(buf, "w/ PCC=%.2f", res.pearson);
        result += std::string(buf);
        console_printf("Test #");
        console_printf(std::to_string(i).c_str());
        console_printf(": ");
        console_printf(spacing.c_str());
        console_printf(result.c_str());
        console_printf(" ");
        console_printf(std::to_string(i).c_str());
        console_printf(" ");
        console_printf(spacing.c_str());
        console_printf(res.test_name.c_str());
        console_printf("\n");
    }
    std::string result = all_pass ? "ALL PASS" : "SOME FAIL";
    std::string count_result = std::to_string(count_pass) + "/" + std::to_string(num_tests) + " tests passed!\n";
    console_printf("---------------------------------------------------------------------------------\n");
    console_printf(result.c_str());
    console_printf("\n");
    console_printf(count_result.c_str());
    console_printf("---------------------------------------------------------------------------------\n");
    console_printf("--- Host code function: ");
    console_printf(host_function_name.c_str());
    console_printf("\n");
    console_printf("---------------------------------------------------------------------------------\n");
}

void run_verbose_test(HostCodeFunctionPtr host_func, std::string host_func_name, int test_num, TestFunctionPtr* registry){
    auto [a, b, test_name] = registry[test_num]();
    TestResult res = run_test(host_func, a, b, test_name, true);

    console_printf("--------------------------------------------------------\n");
    console_printf("--- Single Test results --------------------------------\n");
    console_printf("--------------------------------------------------------\n");
    console_printf("--- Host Code function: ");
    console_printf(host_func_name.c_str());
    console_printf("\n");
    console_printf("--------------------------------------------------------\n");

    bool pass = true;
    if (res.pearson < 0.99){
        pass = false;
    }

    char buf[13];
    std::string result = pass ? "PASS " : "FAIL ";
    sprintf(buf, "w/ PCC=%.2f", res.pearson);
    result += std::string(buf);
    console_printf("Test #");
    console_printf(std::to_string(test_num).c_str());
    console_printf(": ");
    console_printf(result.c_str());
    console_printf(" ");
    console_printf(std::to_string(test_num).c_str());
    console_printf(" ");
    console_printf(res.test_name.c_str());
    console_printf("\n");
    console_printf("--------------------------------------------------------\n");
    console_printf("--------------------------------------------------------\n");
    console_printf("--------------------------------------------------------\n");
}

int main(int argc, char** argv) {
    bool test_all = true;
    int host_code_index = 0;
    if (argc > 1) {
        test_all = std::string(argv[1]) == "all";
    }
    if (argc > 2) {
        host_code_index = std::stoi(argv[2]);
    }

    // Registry selection: use SC26 profiling_suite Registries, or fall back to TestRegistry
    int registry_number = argc > 3 ? std::stoi(argv[3]) : -1;
    TestFunctionPtr *registry = nullptr;
    size_t num_tests = 0;
    if (registry_number >= 0 && registry_number < NUM_REGISTRIES) {
        registry = reinterpret_cast<TestFunctionPtr*>(Registries[registry_number]);
        num_tests = RegistrySizes[registry_number];
    } else {
        registry = TestRegistry;
        num_tests = sizeof(TestRegistry) / sizeof(TestRegistry[0]);
    }

    // Host code selection from HostCodeRegistryVerbose
    auto [host_func, host_func_name] = HostCodeRegistryVerbose[host_code_index];

    if (test_all) {
        int saved_stdout = ::dup(STDOUT_FILENO);
        if (saved_stdout == -1) {
            std::perror("dup");
            return 1;
        }

        int log_fd = ::open("std.out.log", O_CREAT | O_WRONLY | O_TRUNC, 0644);
        if (log_fd == -1) {
            std::perror("open");
            return 1;
        }
        if (::dup2(log_fd, STDOUT_FILENO) == -1) {
            std::perror("dup2");
            return 1;
        }
        ::close(log_fd);

        test_suite(host_func, host_func_name, registry, num_tests);
    }
    else {
        int test_num = argc > 1 ? std::stoi(argv[1]) : -1;
        if (test_num == -1) {
            console_printf("No test specified. Returning.\n");
            return 0;
        }
        run_verbose_test(host_func, host_func_name, test_num, registry);
        console_printf("Leaving the test program\n");
    }
}
