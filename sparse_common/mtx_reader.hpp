#ifndef MTX_READER_HPP
#define MTX_READER_HPP

// Matrix Market (.mtx) reader producing the project's BSR layout.
// SuiteSparse ships COO; the SpMM kernels consume BSR with block dims that are
// multiples of the 32x32 tile, so conversion pads M/K with implicit zeros and
// materializes a full R*C block for any block containing at least one entry
// (explicit zeros included — block presence is structural, like CSR nnz).

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bsr_matrix.hpp"

namespace bsr_mtx {

struct coo_entry {
    size_t row;
    size_t col;
    double val;
};

struct coo_matrix {
    size_t rows = 0;
    size_t cols = 0;
    std::vector<coo_entry> entries;  // deduplicated, sorted by (row, col), 0-indexed
};

inline std::string _lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

inline coo_matrix read_mtx(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("read_mtx: cannot open " + path);
    }

    std::string line;
    if (!std::getline(in, line)) {
        throw std::runtime_error("read_mtx: empty file " + path);
    }
    {
        std::istringstream banner(_lower(line));
        std::string magic, object, format, field, symmetry;
        banner >> magic >> object >> format >> field >> symmetry;
        if (magic != "%%matrixmarket" || object != "matrix") {
            throw std::runtime_error("read_mtx: not a Matrix Market file: " + path);
        }
        if (format != "coordinate") {
            throw std::runtime_error("read_mtx: only 'coordinate' format is supported (got '" + format + "'): " + path);
        }
        if (field != "real" && field != "integer" && field != "pattern") {
            throw std::runtime_error("read_mtx: unsupported field '" + field + "' (no complex support): " + path);
        }
        if (symmetry != "general" && symmetry != "symmetric" && symmetry != "skew-symmetric" && symmetry != "hermitian") {
            throw std::runtime_error("read_mtx: unsupported symmetry '" + symmetry + "': " + path);
        }

        bool is_pattern = field == "pattern";
        bool mirror = symmetry != "general";
        double mirror_sign = symmetry == "skew-symmetric" ? -1.0 : 1.0;

        // Dims line is the first non-comment line after the banner.
        size_t rows = 0, cols = 0, declared_nnz = 0;
        while (std::getline(in, line)) {
            if (!line.empty() && line[0] == '%') {
                continue;
            }
            std::istringstream dims(line);
            if (!(dims >> rows >> cols >> declared_nnz)) {
                throw std::runtime_error("read_mtx: bad dims line '" + line + "': " + path);
            }
            break;
        }
        if (rows == 0 || cols == 0) {
            throw std::runtime_error("read_mtx: missing dims line: " + path);
        }

        // Sum duplicate coordinates; the map also yields sorted (row, col) order.
        std::map<std::pair<size_t, size_t>, double> acc;
        size_t parsed = 0;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '%') {
                continue;
            }
            std::istringstream e(line);
            int64_t i, j;
            double v = 1.0;
            if (!(e >> i >> j)) {
                continue;
            }
            if (!is_pattern && !(e >> v)) {
                throw std::runtime_error("read_mtx: missing value in entry '" + line + "': " + path);
            }
            if (i < 1 || j < 1 || (size_t)i > rows || (size_t)j > cols) {
                throw std::runtime_error("read_mtx: entry out of range '" + line + "': " + path);
            }
            acc[{i - 1, j - 1}] += v;
            if (mirror && i != j) {
                acc[{j - 1, i - 1}] += mirror_sign * v;
            }
            parsed++;
        }
        if (parsed != declared_nnz) {
            throw std::runtime_error(
                "read_mtx: expected " + std::to_string(declared_nnz) + " entries, parsed " + std::to_string(parsed) +
                ": " + path);
        }

        coo_matrix coo;
        coo.rows = rows;
        coo.cols = cols;
        coo.entries.reserve(acc.size());
        for (auto& [coord, v] : acc) {
            coo.entries.push_back({coord.first, coord.second, v});
        }
        return coo;
    }
}

inline bsr_matrix<float> coo_to_bsr(const coo_matrix& coo, size_t R, size_t C) {
    if (R == 0 || C == 0 || R % 32 != 0 || C % 32 != 0) {
        throw std::invalid_argument("coo_to_bsr: R and C must be positive multiples of 32");
    }
    const size_t M_pad = ((coo.rows + R - 1) / R) * R;
    const size_t W_pad = ((coo.cols + C - 1) / C) * C;
    const size_t bmh = M_pad / R;
    const size_t bmw = W_pad / C;

    // Group entries by block; map iteration yields blocks in (bi, bj) lexicographic order.
    std::map<std::pair<size_t, size_t>, std::vector<coo_entry>> blocks;
    for (const auto& e : coo.entries) {
        blocks[{e.row / R, e.col / C}].push_back(e);
    }

    std::vector<int> indptr(bmh + 1, 0);
    std::vector<int> indices;
    std::vector<float> data;
    indices.reserve(blocks.size());
    data.reserve(blocks.size() * R * C);

    for (auto& [bcoord, elems] : blocks) {
        const size_t bi = bcoord.first, bj = bcoord.second;
        if (bj >= bmw) {
            throw std::runtime_error("coo_to_bsr: internal block col overflow");
        }
        indptr[bi + 1]++;
        indices.push_back((int)bj);
        const size_t base = data.size();
        data.resize(base + R * C, 0.0f);
        for (const auto& e : elems) {
            data[base + (e.row % R) * C + (e.col % C)] = (float)e.val;
        }
    }
    for (size_t i = 1; i < indptr.size(); i++) {
        indptr[i] += indptr[i - 1];
    }

    return bsr_matrix<float>(
        std::move(data), std::move(indptr), std::move(indices), M_pad, W_pad, R, C, blocks.size());
}

inline bsr_matrix<float> read_mtx_as_bsr(const std::string& path, size_t R, size_t C) {
    return coo_to_bsr(read_mtx(path), R, C);
}

}  // namespace bsr_mtx

#endif  // MTX_READER_HPP
