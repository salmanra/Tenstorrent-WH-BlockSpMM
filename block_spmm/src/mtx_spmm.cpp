
// mtx_spmm: run a real SuiteSparse (.mtx) matrix through the BlockSpMM host variants.
//
//   mtx_spmm --selftest                                    host-only reader checks on fixtures
//   mtx_spmm <file.mtx> [variant=0..5] [R=32] [C=32] [N=1024] [--dump]
//
// variant indexes HostCodeRegistryVerbose (0=DM, 1=SnF, 2=CDA, 3-5=no_lb).
// --dump writes the constructed BSR data/indices/indptr vectors to <stem>_bsr_*.txt.

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#include "../inc/include_me.hpp"
#include "../inc/host_code.hpp"
// include_me.hpp must precede this: the two bsr_matrix.hpp copies share one
// include guard, and we want the block_spmm copy.
#include "../../sparse_common/mtx_reader.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

using namespace bsr_host_code;

namespace {

std::string fixture_dir() {
    const char* home = std::getenv("TT_METAL_HOME");
    if (!home) {
        TT_THROW("TT_METAL_HOME must be set");
    }
    return std::string(home) +
        "/tt_metal/programming_examples/Tenstorrent-WH-BlockSpMM/block_spmm/tests/mtx";
}

bool check(bool cond, const std::string& what) {
    std::printf("  %s %s\n", cond ? "PASS" : "FAIL", what.c_str());
    return cond;
}

// Hand-computed expectations for block_spmm/tests/mtx fixtures at R=C=32.
int selftest() {
    bool ok = true;
    const std::string dir = fixture_dir();

    {
        // tiny_general.mtx: 3x3 real general, 5 lines but a duplicated (1,1) entry.
        // Unique nnz: (0,0)=2.5, (0,2)=-1.5, (1,0)=1.0, (2,1)=4.0
        bsr_matrix<float> a = bsr_mtx::read_mtx_as_bsr(dir + "/tiny_general.mtx", 32, 32);
        ok &= check(a.H == 32 && a.W == 32, "tiny_general pads 3x3 to 32x32");
        ok &= check(a.nblocks == 1, "tiny_general has 1 block");
        ok &= check(a.indptr == std::vector<int>({0, 1}), "tiny_general indptr");
        ok &= check(a.indices == std::vector<int>({0}), "tiny_general indices");
        ok &= check(
            a.data.size() == 1024 && a.data[0] == 2.5f && a.data[2] == -1.5f && a.data[32] == 1.0f &&
                a.data[65] == 4.0f,
            "tiny_general block data (duplicate summed)");
        dense_matrix<float> d = a.to_dense();
        ok &= check(
            d.data[0 * 32 + 0] == 2.5f && d.data[0 * 32 + 2] == -1.5f && d.data[1 * 32 + 0] == 1.0f &&
                d.data[2 * 32 + 1] == 4.0f && d.data[5 * 32 + 5] == 0.0f,
            "tiny_general to_dense");
    }
    {
        // tiny_symmetric.mtx: integer symmetric -> (0,0)=2, (1,0)=3, (0,1)=3, (2,2)=4
        bsr_matrix<float> a = bsr_mtx::read_mtx_as_bsr(dir + "/tiny_symmetric.mtx", 32, 32);
        dense_matrix<float> d = a.to_dense();
        ok &= check(a.nblocks == 1, "tiny_symmetric has 1 block");
        ok &= check(
            d.data[0] == 2.0f && d.data[1 * 32 + 0] == 3.0f && d.data[0 * 32 + 1] == 3.0f && d.data[2 * 32 + 2] == 4.0f,
            "tiny_symmetric mirrored entries");
    }
    {
        // tiny_pattern.mtx: pattern symmetric -> entries carry implicit value 1.0
        bsr_matrix<float> a = bsr_mtx::read_mtx_as_bsr(dir + "/tiny_pattern.mtx", 32, 32);
        dense_matrix<float> d = a.to_dense();
        ok &= check(a.nblocks == 1, "tiny_pattern has 1 block");
        ok &= check(
            d.data[0] == 1.0f && d.data[1 * 32 + 0] == 1.0f && d.data[0 * 32 + 1] == 1.0f,
            "tiny_pattern implicit 1.0 + mirror");
    }
    {
        // tiny_array.mtx: array format must be rejected
        bool threw = false;
        try {
            bsr_mtx::read_mtx(dir + "/tiny_array.mtx");
        } catch (const std::runtime_error& e) {
            threw = std::string(e.what()).find("coordinate") != std::string::npos;
        }
        ok &= check(threw, "tiny_array rejected (array format)");
    }

    std::printf("%s\n", ok ? "SELFTEST ALL PASS" : "SELFTEST SOME FAIL");
    return ok ? 0 : 1;
}

size_t round_up(size_t v, size_t mult) { return (v + mult - 1) / mult * mult; }

int run_mtx(
    const std::string& path, int variant, size_t R, size_t C, size_t N_req, bool dump) {
    bsr_mtx::coo_matrix coo = bsr_mtx::read_mtx(path);
    bsr_matrix<float> a_f = bsr_mtx::coo_to_bsr(coo, R, C);
    bsr_matrix<bfloat16> a = a_f.bfloat16_cast();

    const uint32_t M = a.H, K = a.W;
    // N padded to a multiple of 12 tiles: keeps Nt's prime factors <= 3 so
    // get_Npc_from_BSR_block_size never pins Npc_min above the L1 limit.
    const uint32_t N = round_up(N_req, 12 * TILE_WIDTH);

    const size_t stored = a.nblocks * R * C;
    const double blowup = coo.entries.empty() ? 0.0 : (double)stored / (double)coo.entries.size();
    std::printf(
        "Matrix %s: %zux%zu nnz=%zu -> BSR %zux%zu padded %ux%u, nblocks=%zu, stored=%zu (%.2fx nnz)\n",
        path.c_str(), coo.rows, coo.cols, coo.entries.size(), (size_t)R, (size_t)C, M, K,
        a.nblocks, stored, blowup);
    if (N != N_req) {
        std::printf("N padded %zu -> %u\n", N_req, N);
    }

    if (dump) {
        std::string stem = path.substr(path.find_last_of("/\\") + 1);
        stem = stem.substr(0, stem.find_last_of('.'));
        std::ofstream d(stem + "_bsr_data.txt"), ip(stem + "_bsr_indptr.txt"), ix(stem + "_bsr_indices.txt");
        for (auto v : a.data) d << v << "\n";
        for (auto v : a.indptr) ip << v << "\n";
        for (auto v : a.indices) ix << v << "\n";
        std::printf("Dumped %s_bsr_{data,indptr,indices}.txt\n", stem.c_str());
    }

    auto [host_func, host_func_name] = HostCodeRegistryVerbose[variant];
    std::printf("Variant %d: %s\n", variant, host_func_name.c_str());

    const tt::ChipId device_id = bspmm_compat::select_blackhole_device_id();
    IDevice* device = bspmm_compat::create_blackhole_device_slow_dispatch();
    TT_FATAL(
        device->arch() == tt::ARCH::BLACKHOLE,
        "BlockSpMM bring-up requires a Blackhole device; selected device {} has arch {}",
        device_id,
        static_cast<int>(device->arch()));

    dense_matrix<bfloat16> b(K, N, RAND);
    dense_matrix<float> tmp(M, N, 0.0f);
    dense_matrix<bfloat16> output = tmp.bfloat16_cast();

    log_info(tt::LogVerif, "Computing host golden SpMM (M={} N={} K={} blocks={})", M, N, K, a.nblocks);
    dense_matrix<bfloat16> golden = a.omp_spmm_bf16(b);
    log_info(tt::LogVerif, "Host golden SpMM complete");

    a.data = tilize_nfaces(a.data, R, C);
    b.data = tilize_nfaces(b.data, K, N);

    host_func(a, b, output, false, a.nblocks, M, N, K, R, C, 1, device);

    output.data = untilize_nfaces(output.data, M, N);
    float pearson = check_bfloat16_vector_pcc(golden.data, output.data);

    bspmm_compat::close_device_slow_dispatch(device);

    bool pass = pearson >= 0.99f;
    std::printf("%s PCC=%.4f %s\n", pass ? "PASS" : "FAIL", pearson, host_func_name.c_str());
    return pass ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--selftest") {
        return selftest();
    }
    if (argc < 2) {
        std::fprintf(
            stderr,
            "usage: %s --selftest | <file.mtx> [variant=0..5] [R=32] [C=32] [N=1024] [--dump]\n",
            argv[0]);
        return 1;
    }
    std::string path = argv[1];
    int variant = argc > 2 ? std::stoi(argv[2]) : 2;
    size_t R = argc > 3 ? std::stoul(argv[3]) : 32;
    size_t C = argc > 4 ? std::stoul(argv[4]) : 32;
    size_t N = argc > 5 ? std::stoul(argv[5]) : 1024;
    bool dump = false;
    for (int i = 2; i < argc; i++) {
        if (std::string(argv[i]) == "--dump") {
            dump = true;
        }
    }
    if (variant < 0 || variant >= 6) {
        std::fprintf(stderr, "variant must be 0..5\n");
        return 1;
    }
    try {
        return run_mtx(path, variant, R, C, N, dump);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
