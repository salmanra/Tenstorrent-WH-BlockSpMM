#ifndef BSR_MATRIX_HPP
#define BSR_MATRIX_HPP

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <random>
#include <cmath>
#include <chrono>
#include "bmm_op.hpp"
#include "tt-metalium/constants.hpp"

#define FILL_ROW 1
#define FILL_COL 2
#define FILL_DIAG 3 // will require the size to be perfect
#define FILL_TRIL 4 // block lower triangular (requires square block grid)
#define FILL_MULTI_DIAG 5 // fills diagonals expanding outward from the main diagonal
#define TILE_SIZE 32
#define RAND_DENOM 2 << 10 // trying to control the range...
#define SIGNED_RAND_MAX RAND_MAX / 2

#pragma omp parallel num_threads(48)

enum content_type{
    RAND,
    UNIFORM,
    ARANGE,
    ID
};

// TODO: if we wanted, we could put this in a namespace,
//        then we could define gemm() and spmm() to be
//        non-member functions in the namespace.

template <typename T>
class dense_matrix {
public:


    std::vector<T> data;
    size_t H;
    size_t W;

    dense_matrix() : H(0), W(0) {}

    dense_matrix(int rows, int cols, content_type content) : H(rows), W(cols) {
        data.resize(rows * cols);
        uint32_t k = 0;
        switch (content) {
            case RAND:
                std::generate(data.begin(), data.end(), []() { return static_cast<T>(SIGNED_RAND_MAX - rand()) / static_cast<T>(RAND_DENOM); });
                break;
            case ID:
                for (auto it = data.begin(); it != data.end(); it++){
                    *it = static_cast<T>(((k / W) == (k % W)) ? 1.0f : 0.0f);
                    k++;
                }
                break;
            case UNIFORM:
                for (auto it = data.begin(); it != data.end(); it++){
                    *it = static_cast<T>(1.0f);
                }
                break;
            case ARANGE:
                for (auto it = data.begin(); it != data.end(); it++){
                    *it = static_cast<T>(static_cast<float>(k++));
                }
                break;
        }
        // if (random)
        //     std::generate(data.begin(), data.end(), []() { return static_cast<T>(SIGNED_RAND_MAX - rand()) / static_cast<T>(RAND_DENOM); });
        // else {
        //     // std::fill(data.begin(), data.end(), 1);
        //     uint32_t k = 0;
        //     for (auto it = data.begin(); it != data.end(); it++){
        //         *it = static_cast<T>(k++);
        //     }
        // }
    }

    dense_matrix(int rows, int cols) : H(rows), W(cols) {
        data.resize(rows * cols);
        std::fill(data.begin(), data.end(), 0);
    }

    dense_matrix(int rows, int cols, T val) : H(rows), W(cols) {
        data.resize(rows * cols);
        std::fill(data.begin(), data.end(), val);
    }

    // Constructor from a flat vector
    dense_matrix(const std::vector<T> &data, int rows, int cols) : data(data), H(rows), W(cols) {
        assert(data.size() == rows * cols);
    }

    // Constructor from a flat vector with no size check
    dense_matrix(const dense_matrix<T> &other) : data(other.data), H(other.H), W(other.W) {}

    // Assignment operator
    dense_matrix<T>& operator=(const dense_matrix<T> &other) {
        if (this != &other) {
            data = other.data;
            H = other.H;
            W = other.W;
        }
        return *this;
    }

    // Move constructor
    dense_matrix(dense_matrix<T> &&other) noexcept : data(std::move(other.data)), H(other.H), W(other.W) {
        other.H = 0;
        other.W = 0;
    }

    // Move to bfloat16
    dense_matrix<bfloat16> bfloat16_cast() {
        std::vector<bfloat16> bfloat16_data(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            bfloat16_data[i] = bfloat16(data[i]);
        }
        return dense_matrix<bfloat16>(bfloat16_data, H, W);
    }

    void print(std::ostream& os = std::cout) {
        os << "Dense Matrix:" << std::endl;
        for (size_t i = 0; i < H; ++i) {
            for (size_t j = 0; j < W; ++j) {
                os << data[i * W + j] << " ";
            }
            os << std::endl;
        }
    }

    void pretty_print(std::ostream& os = std::cout) {
        os << "Dense Matrix:" << std::endl;
        os << "Size (H x W): " << H << " x " << W << std::endl;
    }

    dense_matrix<T> gemm(const dense_matrix<T> &other) {
        assert(W == other.H);
        dense_matrix<T> result(H, other.W);
        for (size_t i = 0; i < H; ++i) {
            for (size_t j = 0; j < other.W; ++j) {
                T sum = 0;
                for (size_t k = 0; k < W; ++k) {
                    sum += data[i * W + k] * other.data[k * other.W + j];
                }
                result.data[i * other.W + j] = sum;
            }
        }
        return result;
    }

    dense_matrix<T> gemm_bfloat16(const dense_matrix<T> &other) {
        assert(W == other.H);
        dense_matrix<T> result(H, other.W);
        for (size_t i = 0; i < H; ++i) {
            for (size_t j = 0; j < other.W; ++j) {
                float sum = 0;
                for (size_t k = 0; k < W; ++k) {
                    sum += data[i * W + k].to_float() * other.data[k * other.W + j].to_float();
                }
                result.data[i * other.W + j] = T(sum);
            }
        }
        return result;
    }
    bool all_close(const dense_matrix<T> &other, T tol = 1e-3) {
        assert(H == other.H && W == other.W);
        for (size_t i = 0; i < H; ++i) {
            for (size_t j = 0; j < W; ++j) {
                if (std::abs(data[i * W + j] - other.data[i * W + j]) > tol) {
                    return false;
                }
            }
        }
        return true;
    }

    bool all_close_bfloat16(const dense_matrix<bfloat16> &other, float tol = 1) {
        assert(H == other.H && W == other.W);
        for (size_t i = 0; i < H; ++i) {
            for (size_t j = 0; j < W; ++j) {

                if (std::abs(data[i * W + j].to_float() - other.data[i * W + j].to_float()) > tol) {
                    return false;
                }
            }
        }
        return true;
    }
};

// Is there a world where this class can handle TT host code H2D/D2H? And does that look at all like Policies?
// It's a fun idea but not really what this needs.
// The actual lesson is that partial instantiation enables the bfloat16/native c++ float type split
//      to happen within a single class instead of the weirdness I have right now.
template <typename T>
class bsr_matrix {
public: // everything is public for now
    std::vector<int> indptr;
    std::vector<int> indices;
    std::vector<T> data;

    size_t H;
    size_t W;
    size_t nblocks;
    size_t R;
    size_t C;

    static std::vector<std::pair<size_t, size_t>> multi_diag_positions(
            size_t blocked_h, size_t blocked_w, size_t nblocks) {
        std::vector<std::pair<size_t, size_t>> positions;
        positions.reserve(nblocks);
        // Expand outward from main diagonal: d=0, +1, -1, +2, -2, ...
        for (int d = 0; positions.size() < nblocks; ) {
            if (d >= 0) {
                // super-diagonal d
                for (size_t i = 0; i < blocked_h && i + d < blocked_w && positions.size() < nblocks; i++) {
                    positions.emplace_back(i, i + d);
                }
            } else {
                // sub-diagonal |d|
                size_t ad = static_cast<size_t>(-d);
                for (size_t j = 0; j < blocked_w && j + ad < blocked_h && positions.size() < nblocks; j++) {
                    positions.emplace_back(j + ad, j);
                }
            }
            // Advance: 0 -> 1 -> -1 -> 2 -> -2 -> ...
            if (d <= 0) d = -d + 1; else d = -d;
            // Safety: if we've exhausted all positions
            if (static_cast<size_t>(std::abs(d)) >= std::max(blocked_h, blocked_w)) break;
        }
        std::sort(positions.begin(), positions.end());
        return positions;
    }

public:
    bsr_matrix() : H(0), W(0), nblocks(0), R(0), C(0) {}
    bsr_matrix(size_t rows, size_t cols, size_t block_rows, size_t block_cols, size_t num_blocks, int fill_type = FILL_ROW, content_type content = RAND) :
        H(rows), W(cols), R(block_rows), C(block_cols), nblocks(num_blocks) {
        assert(H * W >= nblocks * R * C);
        assert(R > 0);
        assert(C > 0);
        assert(H % R == 0);
        assert(W % C == 0);
        assert(H >= R);
        assert(W >= C);


        size_t blocked_matrix_height = H / R;
        size_t blocked_matrix_width = W / C;

        indptr.resize(blocked_matrix_height + 1);
        indices.reserve(nblocks);
        data.reserve(nblocks * R * C);
        std::fill(indptr.begin(), indptr.end(), 0);
        // if Fill_row, fill the matrix with blocks in a row-wise manner
        // if Fill_col, fill the matrix with blocks in a column-wise manner
        if (fill_type == FILL_ROW) {
            for (size_t i = 0; i < blocked_matrix_height; i++) {
                for (size_t j = 0; j < blocked_matrix_width; j++) {
                    if (i * blocked_matrix_width + j < nblocks) {
                        indptr[i + 1]++;
                        indices.push_back(j);
                        for (size_t k = 0; k < R * C; k++) {
                            float temp = ((k / C) == (k % C)) ? 1.0 : 0.0;
                            T val = static_cast<T>(temp);
                            switch (content) {
                                case RAND: 
                                    data.push_back(static_cast<T>(SIGNED_RAND_MAX - rand()) / static_cast<T>(RAND_DENOM));
                                    break;
                                case UNIFORM:
                                    data.push_back(static_cast<T>(1.0f));
                                    break;
                                case ID:
                                    data.push_back(val);
                                    break;
                                case ARANGE:
                                    data.push_back(static_cast<T>(static_cast<float>(k)));
                                    break;
                            }
                        }
                    }
                }
            }
        } else if (fill_type == FILL_COL) {
            for (size_t j = 0; j < blocked_matrix_width; j++) {
                for (size_t i = 0; i < blocked_matrix_height; i++) {
                    if (i + (j * blocked_matrix_height) < nblocks) {
                        indptr[i + 1]++;
                        indices.push_back(j);
                        for (size_t k = 0; k < R * C; k++) {
                            float temp = ((k / C) == (k % C)) ? 1.0 : 0.0;
                            T val = static_cast<T>(temp);
                            switch (content) {
                                case RAND: 
                                    data.push_back(static_cast<T>(SIGNED_RAND_MAX - rand()) / static_cast<T>(RAND_DENOM));
                                    break;
                                case UNIFORM:
                                    data.push_back(static_cast<T>(1.0f));
                                    break;
                                case ID:
                                    data.push_back(val);
                                    break;
                                case ARANGE:
                                    data.push_back(static_cast<T>(static_cast<float>(k)));
                                    break;
                            }
                        }
                    }
                }
            }
        } else if (fill_type == FILL_DIAG){
            // TODO: add logic to fill multiple diagonals. Maybe just fill from the center out?
            //          And we can use nblocks to mean num_diags instead maybe...
            assert(nblocks <= std::min(blocked_matrix_height, blocked_matrix_width));
            for (size_t i = 0; i < nblocks; i++) {
                indptr[i + 1]++;
                indices.push_back(i);
                for (size_t k = 0; k < R * C; k++) {
                    float temp = ((k / C) == (k % C)) ? 1.0 : 0.0;
                    T val = static_cast<T>(temp);
                    switch (content) {
                        case RAND:
                            data.push_back(static_cast<T>(SIGNED_RAND_MAX - rand()) / static_cast<T>(RAND_DENOM));
                            break;
                        case UNIFORM:
                            data.push_back(static_cast<T>(1.0f));
                            break;
                        case ID:
                            data.push_back(val);
                            break;
                        case ARANGE:
                            data.push_back(static_cast<T>(static_cast<float>(k)));
                            break;
                    }
                }
            }

        } else if (fill_type == FILL_TRIL) {
            assert(blocked_matrix_height == blocked_matrix_width);
            nblocks = blocked_matrix_height * (blocked_matrix_height + 1) / 2;
            indices.reserve(nblocks);
            data.reserve(nblocks * R * C);
            for (size_t i = 0; i < blocked_matrix_height; i++) {
                for (size_t j = 0; j <= i; j++) {
                    indptr[i + 1]++;
                    indices.push_back(j);
                    for (size_t k = 0; k < R * C; k++) {
                        float temp = ((k / C) == (k % C)) ? 1.0 : 0.0;
                        T val = static_cast<T>(temp);
                        switch (content) {
                            case RAND:
                                data.push_back(static_cast<T>(SIGNED_RAND_MAX - rand()) / static_cast<T>(RAND_DENOM));
                                break;
                            case UNIFORM:
                                data.push_back(static_cast<T>(1.0f));
                                break;
                            case ID:
                                data.push_back(val);
                                break;
                            case ARANGE:
                                data.push_back(static_cast<T>(static_cast<float>(k)));
                                break;
                        }
                    }
                }
            }

        } else if (fill_type == FILL_MULTI_DIAG) {
            auto positions = multi_diag_positions(blocked_matrix_height, blocked_matrix_width, nblocks);
            nblocks = positions.size();  // may be fewer if matrix is small
            for (auto& [row, col] : positions) {
                indptr[row + 1]++;
                indices.push_back(col);
                for (size_t k = 0; k < R * C; k++) {
                    float temp = ((k / C) == (k % C)) ? 1.0 : 0.0;
                    T val = static_cast<T>(temp);
                    switch (content) {
                        case RAND:
                            data.push_back(static_cast<T>(SIGNED_RAND_MAX - rand()) / static_cast<T>(RAND_DENOM));
                            break;
                        case UNIFORM:
                            data.push_back(static_cast<T>(1.0f));
                            break;
                        case ID:
                            data.push_back(val);
                            break;
                        case ARANGE:
                            data.push_back(static_cast<T>(static_cast<float>(k)));
                            break;
                    }
                }
            }

        } else {
            throw std::invalid_argument("Invalid fill type");
        }
        for (size_t i = 1; i < indptr.size(); i++) {
            indptr[i] += indptr[i - 1];
        }
        indptr.resize(blocked_matrix_height + 1);
    }

    bsr_matrix(size_t rows, size_t cols, size_t block_rows, size_t block_cols, size_t num_blocks, int fill_type = FILL_ROW) :
        H(rows), W(cols), R(block_rows), C(block_cols), nblocks(num_blocks) {
        assert(H * W >= nblocks * R * C);
        assert(R > 0);
        assert(C > 0);
        assert(H % R == 0);
        assert(W % C == 0);
        assert(H >= R);
        assert(W >= C);

        size_t blocked_matrix_height = H / R;
        size_t blocked_matrix_width = W / C;

        indptr.resize(blocked_matrix_height + 1);
        indices.reserve(nblocks);
        data.reserve(nblocks * R * C);
        std::fill(indptr.begin(), indptr.end(), 0);

        // if Fill_row, fill the matrix with blocks in a row-wise manner
        // if Fill_col, fill the matrix with blocks in a column-wise manner
        if (fill_type == FILL_ROW) {
            for (size_t i = 0; i < blocked_matrix_height; i++) {
                for (size_t j = 0; j < blocked_matrix_width; j++) {
                    if (i * blocked_matrix_width + j < nblocks) {
                        indptr[i + 1]++;
                        indices.push_back(j);
                        for (size_t k = 0; k < R * C; k++) {
                            data.push_back(static_cast<T>(SIGNED_RAND_MAX - rand()) / static_cast<T>(RAND_DENOM));
                        }
                    }
                }
            }
        } else if (fill_type == FILL_COL) {
            for (size_t j = 0; j < blocked_matrix_width; j++) {
                for (size_t i = 0; i < blocked_matrix_height; i++) {
                    if (i * blocked_matrix_width + j < nblocks) {
                        indptr[i + 1]++;
                        indices.push_back(i);
                        for (size_t k = 0; k < R * C; k++) {
                            data.push_back(static_cast<T>(SIGNED_RAND_MAX - rand()) / static_cast<T>(RAND_DENOM));
                        }
                    }
                }
            }
        } else if (fill_type == FILL_TRIL) {
            assert(blocked_matrix_height == blocked_matrix_width);
            nblocks = blocked_matrix_height * (blocked_matrix_height + 1) / 2;
            indices.reserve(nblocks);
            data.reserve(nblocks * R * C);
            for (size_t i = 0; i < blocked_matrix_height; i++) {
                for (size_t j = 0; j <= i; j++) {
                    indptr[i + 1]++;
                    indices.push_back(j);
                    for (size_t k = 0; k < R * C; k++) {
                        data.push_back(static_cast<T>(SIGNED_RAND_MAX - rand()) / static_cast<T>(RAND_DENOM));
                    }
                }
            }
        } else if (fill_type == FILL_MULTI_DIAG) {
            auto positions = multi_diag_positions(blocked_matrix_height, blocked_matrix_width, nblocks);
            nblocks = positions.size();
            for (auto& [row, col] : positions) {
                indptr[row + 1]++;
                indices.push_back(col);
                for (size_t k = 0; k < R * C; k++) {
                    data.push_back(static_cast<T>(SIGNED_RAND_MAX - rand()) / static_cast<T>(RAND_DENOM));
                }
            }
        } else {
            throw std::invalid_argument("Invalid fill type");
        }
    }


    // construct a copy of the dense matrix in this bsr matrix
    // this is the first function that caters to TT hardware by
    // picking block size a multiple of tile size
    // REQUIRES: other has been tilized by the TT utils
    bsr_matrix(dense_matrix<T>& other, uint32_t right_outer_dim = 0) :
        H(other.H),
        W(other.W)
        {
        // how do we choose block sizes?
        // largest square multiple of 32 that tilizes the dense matrix?
        // 32x32 for everything?
        // 128x128 for everything, falling back to smaller sizes if that doesn't fit?
        // 64x64 for everything, falling back to smaller sizes. That's it because in0_block_w = 2.
        // TODO: change this to choose block sizes equal to whatever TT would pick for its dense block sizes,
        //          using the sparse fit-in-SRAM function.
        //          maybe we maintain our own version of bmm_op.hpp after all...
        // TODO: the way we are sizing right now is a bit of a cheat, but it's good for now.
        // TODO: continue the cheat by passing number of cores to constructor


        uint32_t Nt = right_outer_dim > 0 ? right_outer_dim / TILE_WIDTH : std::max(H, W) / TILE_WIDTH;
        auto matmul_params = bmm_op_utils::get_large_matmul_params(H / TILE_HEIGHT, Nt, 8, 8, 2);
        uint32_t per_core_M = std::get<0>(matmul_params);
        uint32_t per_core_N = std::get<1>(matmul_params);
        uint32_t out_subblock_h = std::get<2>(matmul_params);
        uint32_t out_subblock_w = std::get<3>(matmul_params);


        // try to let Ct = 2, but accept it otherwise
        R = per_core_M * TILE_HEIGHT;
        C = std::min(per_core_N * TILE_WIDTH, 2 * TILE_WIDTH);
        // if (W % 64 == 0)
        //     C = 64;
        // if (H % 64 == 0)
        //     R = 64; /
        // R = 32;
        // C = 32;

        size_t blocked_matrix_height = H / R;
        size_t blocked_matrix_width = W / C;
        nblocks = blocked_matrix_height * blocked_matrix_width;

        indptr.resize(blocked_matrix_height + 1);
        indices.reserve(nblocks);
        data.resize(nblocks * R * C);

        size_t count = 0;
        for (int i = 0 ; i < nblocks; i++){
            size_t row = i / blocked_matrix_width;
            size_t col = i % blocked_matrix_width;
            indptr[row + 1]++;
            indices.push_back(col);
            for (int j = 0; j < R * C; j++){
                data.push_back(other.data[count++]);
            }
        }

        for (size_t i = 1; i < indptr.size(); i++) {
            indptr[i] += indptr[i - 1];
        }
        indptr.resize(blocked_matrix_height + 1);

        std::vector<T> tilized_input;
        tilized_input.reserve(other.data.size());

        const auto write_face = [](std::vector<T>& tilized_input,
                                const std::vector<T>& input,
                                uint32_t face_height,
                                uint32_t face_width,
                                uint32_t face_idx,
                                uint32_t n) -> void {
            for (uint32_t i = 0; i < face_height; i++) {
                for (uint32_t j = 0; j < face_width; j++) {
                    tilized_input.push_back(input[face_idx + j]);
                }
                face_idx += n;
            }
        };

        uint32_t TILE_HEIGHT = R;
        uint32_t TILE_WIDTH = C;
        uint32_t FACE_HEIGHT = R;
        uint32_t FACE_WIDTH = C;
        uint32_t row_tiles = H / TILE_HEIGHT;
        uint32_t col_tiles = W / TILE_WIDTH;
        uint32_t row_of_tiles_num_elements = TILE_HEIGHT * W;
        uint32_t tile_start = 0;
        for (uint32_t row_tile = 0; row_tile < row_tiles; row_tile++) {
            uint32_t row_tile_start = tile_start;
            for (uint32_t col_tile = 0; col_tile < col_tiles; col_tile++) {
                uint32_t face0_id = row_tile_start;
                uint32_t face1_id = face0_id + FACE_WIDTH;
                uint32_t face2_id = face0_id + W * FACE_HEIGHT;
                uint32_t face3_id = face2_id + FACE_WIDTH;

                write_face(tilized_input, other.data, FACE_HEIGHT, FACE_WIDTH, face0_id, W);
                // write_face(tilized_input, other.data, FACE_HEIGHT, FACE_WIDTH, face1_id, W);
                // write_face(tilized_input, other.data, FACE_HEIGHT, FACE_WIDTH, face2_id, W);
                // write_face(tilized_input, other.data, FACE_HEIGHT, FACE_WIDTH, face3_id, W);
                row_tile_start += TILE_WIDTH;
            }
            tile_start += row_of_tiles_num_elements;
        }


        data = std::move(tilized_input);
    }

    // construct from custom data members. Most flexible constructor.
    bsr_matrix(
        std::vector<T> data,
        std::vector<int> indptr,
        std::vector<int> indices,
        size_t rows,
        size_t cols,
        size_t block_rows,
        size_t block_cols,
        size_t num_blocks) :
        data(std::move(data)),
        indptr(std::move(indptr)),
        indices(std::move(indices)),
        H(rows),
        W(cols),
        R(block_rows),
        C(block_cols),
        nblocks(num_blocks) {
        assert(H * W >= nblocks * R * C);
        assert(R > 0);
        assert(C > 0);
        assert(H % R == 0);
        assert(W % C == 0);
        assert(H >= R);
        assert(W >= C);
    }

    bsr_matrix(int rows, int cols, size_t block_rows, size_t block_cols, size_t num_blocks, content_type content) {
        H = rows;
        W = cols;
        R = block_rows;
        C = block_cols;
        nblocks = num_blocks;

        assert(H*W >= nblocks * R * C);
        assert(R > 0);
        assert(C > 0);
        assert(H % R == 0);
        assert(W % C == 0);
        assert(H >= R);
        assert(W >= C);

        size_t blocked_matrix_height = H / R;
        size_t blocked_matrix_width = W / C;

        indptr.resize(blocked_matrix_height + 1);
        indices.reserve(nblocks);
        data.reserve(nblocks * R * C);
        std::fill(indptr.begin(), indptr.end(), 0);

        std::vector<int> block_indices(blocked_matrix_height * blocked_matrix_width);
        std::fill(block_indices.begin(), block_indices.begin() + nblocks, 1);
        std::fill(block_indices.begin() + nblocks, block_indices.end(), 0);

        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        shuffle(block_indices.begin(), block_indices.end(), std::default_random_engine(seed));

        for (size_t i = 0; i < block_indices.size(); i++) {
            if (block_indices[i] == 1) {
                size_t row = i / blocked_matrix_width;
                size_t col = i % blocked_matrix_width;
                indptr[row + 1]++;
                indices.push_back(col);
                for (size_t k = 0; k < R * C; k++) {
                    float temp = ((k / C) == (k % C)) ? 1.0 : 0.0;
                    T val = static_cast<T>(temp);
                    switch (content) {
                        case RAND: 
                            data.push_back(static_cast<T>(SIGNED_RAND_MAX - rand()) / static_cast<T>(RAND_DENOM));
                            break;
                        case UNIFORM:
                            data.push_back(static_cast<T>(1.0f));
                            break;
                        case ID:
                            data.push_back(val);
                            break;
                        case ARANGE:
                            data.push_back(static_cast<T>(static_cast<float>(k)));
                            break;
                    }
                }
            }
        }
        for (size_t i = 1; i < indptr.size(); i++) {
            indptr[i] += indptr[i - 1];
        }
        indptr.resize(blocked_matrix_height + 1);
    }

    // Moves inptr, indices, and data to the new bsr_matrix
    bsr_matrix<bfloat16> bfloat16_cast() {
        std::vector<bfloat16> bfloat16_data(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            bfloat16_data[i] = bfloat16(data[i]);
        }
        return bsr_matrix<bfloat16>(bfloat16_data, indptr, indices, H, W, R, C, nblocks);
    }

    // partial instantiation: this function will get compiled only on instances of dense matrices
    //  on which it is called, at which point it will fail if the type T does not have a member to_float()
    dense_matrix<T> spmm_bfloat16(dense_matrix<T> &B) {
        assert(W == B.H);
        dense_matrix<T> output(H, B.W);

        // FORALL parallelism on the block rows
        for (size_t i = 0; i < indptr.size() - 1; i++) {                        // runtime args
            for (size_t r = 0; r < R; r += TILE_SIZE) {                         // comptime args
                for (size_t p = 0; p < B.W; p += TILE_SIZE) {                   // comptime args
                    std::vector<float> output_tile(TILE_SIZE * TILE_SIZE, 0);   // DST register
                    for (size_t idx = indptr[i]; idx < indptr[i + 1]; idx++) {  // reading raw data from a CB
                        size_t j = indices[idx];                                // reading raw data from a CB
                        auto iter_start = data.begin() + idx * R * C;           // src0_addr, determined from args
                        auto iter_B_start = B.data.begin() + j * C * B.W;       // src1_addr, determined from args
                        for (size_t c = 0; c < C; c += TILE_SIZE) {             // comptime args
                            // begin matmul_tiles API call
                            for (size_t rr = r; rr < std::min(r + TILE_SIZE, R); rr++) {
                                for (size_t pp = p; pp < std::min(p + TILE_SIZE, B.W); pp++) {
                                    float sum = 0;
                                    for (size_t cc = c; cc < std::min(C, c + TILE_SIZE); cc++) {
                                        T a_val = *(iter_start + rr * C + cc);
                                        T b_val = *(iter_B_start + cc * B.W + pp);
                                        sum += a_val.to_float() * b_val.to_float();
                                    }
                                    output_tile[(rr - r) * TILE_SIZE + pp - p] += sum;
                                }
                            }
                            // end matmul_tiles API call
                        }
                    }
                    // write tile to DRAM starting at tile i*R + r, p (output is dense, no more blocks)
                    // On TT:
                    // 1. pack DST reg to output CB
                    // 2. writer kernel pops from output CB
                    // 3. writer kernel NoC's to DRAM
                    for (size_t rr = r; rr < std::min(R, r + TILE_SIZE); rr++) {
                        for (size_t pp = p; pp < std::min(B.W, p + TILE_SIZE); pp++) {
                            *(output.data.begin() + (i * R + rr) * output.W + pp) =
                                T(output_tile[(rr - r) * TILE_SIZE + pp - p]);
                        }
                    }
                }
            }
        }
        return output;
    }

    // Naive CPU SpGEMM: this (BSR) * other (BSR) -> BSR
    // Requires: this->W == other.H and this->C == other.R
    // Uses float accumulation internally for bfloat16 compatibility.
    bsr_matrix<T> spgemm(const bsr_matrix<T> &B) {
        assert(W == B.H);
        assert(C == B.R);

        size_t out_brow = H / R;
        size_t out_bcol = B.W / B.C;

        // Accumulate blocks in float for numerical stability and bfloat16 compat
        std::vector<bool> block_present(out_brow * out_bcol, false);
        std::vector<std::vector<float>> block_accum(out_brow * out_bcol);

        // For each block row i of A
        for (size_t i = 0; i < indptr.size() - 1; i++) {
            // For each nonzero block A[i, k]
            for (size_t a_idx = indptr[i]; a_idx < indptr[i + 1]; a_idx++) {
                size_t k = indices[a_idx];
                auto a_block = data.begin() + a_idx * R * C;

                // For each nonzero block B[k, j]
                for (size_t b_idx = B.indptr[k]; b_idx < B.indptr[k + 1]; b_idx++) {
                    size_t j = B.indices[b_idx];
                    auto b_block = B.data.begin() + b_idx * B.R * B.C;

                    size_t flat = i * out_bcol + j;
                    if (!block_present[flat]) {
                        block_present[flat] = true;
                        block_accum[flat].resize(R * B.C, 0.0f);
                    }
                    auto &acc = block_accum[flat];

                    // Dense block multiply: acc += A[i,k] * B[k,j]
                    for (size_t r = 0; r < R; r++) {
                        for (size_t p = 0; p < B.C; p++) {
                            float sum = 0.0f;
                            for (size_t c = 0; c < C; c++) {
                                sum += static_cast<float>(*(a_block + r * C + c)) * static_cast<float>(*(b_block + c * B.C + p));
                            }
                            acc[r * B.C + p] += sum;
                        }
                    }
                }
            }
        }

        // Build BSR arrays from accumulated blocks (row-major order)
        std::vector<int> out_indptr(out_brow + 1, 0);
        std::vector<int> out_indices;
        std::vector<T> out_data;
        size_t out_nblocks = 0;

        for (size_t i = 0; i < out_brow; i++) {
            for (size_t j = 0; j < out_bcol; j++) {
                size_t flat = i * out_bcol + j;
                if (block_present[flat]) {
                    out_indptr[i + 1]++;
                    out_indices.push_back(j);
                    for (auto& val : block_accum[flat]) {
                        out_data.push_back(T(val));
                    }
                    out_nblocks++;
                }
            }
        }
        for (size_t i = 1; i <= out_brow; i++) {
            out_indptr[i] += out_indptr[i - 1];
        }

        return bsr_matrix<T>(std::move(out_data), std::move(out_indptr), std::move(out_indices),
                              H, B.W, R, B.C, out_nblocks);
    }

    // Naive CPU SDDMM: A = this ⊙ (C × D)
    // this: BSR M×N (sampling mask), C: dense M×K, D: dense K×N
    // Output: BSR M×N with same sparsity pattern as this
    // Uses float accumulation internally for bfloat16 compatibility.
    bsr_matrix<T> sddmm(const dense_matrix<T>& C_mat, const dense_matrix<T>& D_mat) const {
        assert(H == C_mat.H);       // M dimension matches
        assert(W == D_mat.W);       // N dimension matches
        assert(C_mat.W == D_mat.H); // K dimension matches

        size_t K_dim = C_mat.W;

        // Output has identical sparsity structure to this (the sampling mask)
        std::vector<int> out_indptr(indptr);
        std::vector<int> out_indices(indices);
        std::vector<T> out_data;
        out_data.reserve(nblocks * R * C);

        // For each block row i
        for (size_t i = 0; i < indptr.size() - 1; i++) {
            // For each nonzero block (i, j) in the mask
            for (size_t blk_idx = indptr[i]; blk_idx < indptr[i + 1]; blk_idx++) {
                size_t j = indices[blk_idx];
                auto mask_block = data.begin() + blk_idx * R * C;

                // Compute dense block of (C × D) at block position (i, j)
                // Block rows [i*R, (i+1)*R) of C times block cols [j*C_block, (j+1)*C_block) of D
                // Then element-wise multiply with mask block
                for (size_t r = 0; r < R; r++) {
                    for (size_t c = 0; c < C; c++) {
                        // Dot product: C[i*R+r, :] · D[:, j*C+c]
                        float dot = 0.0f;
                        for (size_t k = 0; k < K_dim; k++) {
                            dot += static_cast<float>(C_mat.data[(i * R + r) * K_dim + k])
                                 * static_cast<float>(D_mat.data[k * D_mat.W + (j * C + c)]);
                        }
                        // Hadamard with mask: output = mask_val * dot
                        float mask_val = static_cast<float>(*(mask_block + r * C + c));
                        out_data.push_back(T(mask_val * dot));
                    }
                }
            }
        }

        return bsr_matrix<T>(std::move(out_data), std::move(out_indptr), std::move(out_indices),
                              H, W, R, C, nblocks);
    }

    dense_matrix<T> spmm(dense_matrix<T> &B) {
        assert(W == B.H);
        dense_matrix<T> output(H, B.W);
        for (size_t i = 0; i < indptr.size() - 1; i++) {
            for (size_t idx = indptr[i]; idx < indptr[i + 1]; idx++) {
                size_t j = indices[idx];
                auto iter_start = data.begin() + idx * R * C;
                auto iter_B_start = B.data.begin() + j * C * B.W;
                for (size_t r = 0; r < R; r++) {
                    for (size_t p = 0; p < B.W; p++) {
                        T sum = 0;
                        for (size_t c = 0; c < C; c++) {
                            T a_val = *(iter_start + r * C + c);
                            T b_val = *(iter_B_start + c * B.W + p);
                            sum += a_val * b_val;
                        }
                        *(output.data.begin() + (i * R + r) * output.W + p) += sum;
                    }
                }
            }
        }
        return output;
    }

    dense_matrix<T> omp_spmm(dense_matrix<T> &B) {
        assert(W == B.H);
        dense_matrix<T> output(H, B.W);
        #pragma omp parallel for
        for (size_t i = 0; i < indptr.size() - 1; i++) {
            for (size_t idx = indptr[i]; idx < indptr[i + 1]; idx++) {
                size_t j = indices[idx];
                auto iter_start = data.begin() + idx * R * C;
                auto iter_B_start = B.data.begin() + j * C * B.W;
                for (size_t r = 0; r < R; r++) {
                    for (size_t p = 0; p < B.W; p++) {
                        T sum = 0;
                        #pragma omp reduction(+:sum)
                        for (size_t c = 0; c < C; c++) {
                            T a_val = *(iter_start + r * C + c);
                            T b_val = *(iter_B_start + c * B.W + p);
                            sum += a_val * b_val;
                        }
                        *(output.data.begin() + (i * R + r) * output.W + p) += sum;
                    }
                }
            }
        }
        return output;
    }

    dense_matrix<T> omp_spmm_bf16(dense_matrix<T> &B) {
        assert(W == B.H);
        dense_matrix<T> output(H, B.W);

        // FORALL parallelism on the block rows
        #pragma omp parallel for
        for (size_t i = 0; i < indptr.size() - 1; i++) {                        // runtime args
            for (size_t r = 0; r < R; r += TILE_SIZE) {                         // comptime args
                for (size_t p = 0; p < B.W; p += TILE_SIZE) {                   // comptime args
                    std::vector<float> output_tile(TILE_SIZE * TILE_SIZE, 0);   // DST register
                    for (size_t idx = indptr[i]; idx < indptr[i + 1]; idx++) {  // reading raw data from a CB
                        size_t j = indices[idx];                                // reading raw data from a CB
                        auto iter_start = data.begin() + idx * R * C;           // src0_addr, determined from args
                        auto iter_B_start = B.data.begin() + j * C * B.W;       // src1_addr, determined from args
                        for (size_t c = 0; c < C; c += TILE_SIZE) {             // comptime args
                            // begin matmul_tiles API call
                            for (size_t rr = r; rr < std::min(r + TILE_SIZE, R); rr++) {
                                for (size_t pp = p; pp < std::min(p + TILE_SIZE, B.W); pp++) {
                                    float sum = 0;
                                    #pragma omp reduction(+:sum)
                                    for (size_t cc = c; cc < std::min(C, c + TILE_SIZE); cc++) {
                                        T a_val = *(iter_start + rr * C + cc);
                                        T b_val = *(iter_B_start + cc * B.W + pp);
                                        sum += a_val.to_float() * b_val.to_float();
                                    }
                                    output_tile[(rr - r) * TILE_SIZE + pp - p] += sum;
                                }
                            }
                            // end matmul_tiles API call
                        }
                    }
                    // write tile to DRAM starting at tile i*R + r, p (output is dense, no more blocks)
                    // On TT:
                    // 1. pack DST reg to output CB
                    // 2. writer kernel pops from output CB
                    // 3. writer kernel NoC's to DRAM
                    for (size_t rr = r; rr < std::min(R, r + TILE_SIZE); rr++) {
                        for (size_t pp = p; pp < std::min(B.W, p + TILE_SIZE); pp++) {
                            *(output.data.begin() + (i * R + rr) * output.W + pp) =
                                T(output_tile[(rr - r) * TILE_SIZE + pp - p]);
                        }
                    }
                }
            }
        }
        return output;
    }

    dense_matrix<T> tiled_spmm_CPU(dense_matrix<T> &B) {
        assert(W == B.H);
        dense_matrix<T> output(H, B.W);
        // FORALL parallelism on the block rows
        for (size_t i = 0; i < indptr.size() - 1; i++) { // host
            for (size_t idx = indptr[i]; idx < indptr[i + 1]; idx++) { // host
                size_t j = indices[idx]; // host

                // now. read the code below as though it were the reader kernel on TT,
                // reader_bmm_8bank_output_tiles_partitioned.cpp
                // R --> M, Mt <-- R / TILE_SIZE
                // C --> K, Kt <-- C / TILE_SIZE
                // QUIRK: Kt = Ct, BUT KtNt = KtNt. Kt is used to index into A (of shape RxC)
                //     while KtNt is used to index into B (of shape CxN, or KxN)
                // B.W --> N, Nt <-- B.W / TILE_SIZE
                auto iter_start = data.begin() + idx * R * C;     // src0_addr
                auto iter_B_start = B.data.begin() + j * C * B.W; // src1_addr
                for (size_t r = 0; r < R; r += TILE_SIZE) {
                    for (size_t p = 0; p < B.W; p += TILE_SIZE) {
                        for (size_t rr = r; rr < std::min(R, r + TILE_SIZE); rr++) {
                            for (size_t pp = p; pp < std::min(B.W, p + TILE_SIZE); pp++) {
                                T sum = 0;
                                for (size_t c = 0; c < C; c += TILE_SIZE) {
                                    for (size_t cc = c; cc < std::min(C, c + TILE_SIZE); cc++) {
                                        T a_val = *(iter_start + rr * C + cc);
                                        T b_val = *(iter_B_start + cc * B.W + pp);
                                        sum += a_val * b_val;
                                    }
                                }
                                *(output.data.begin() + (i * R + rr) * output.W + pp) += sum;
                            }
                        }
                    }
                }
            }
        }
        return output;
    }


    // DEAR READER, this is a tile-wise parallel implementation of SpMM on BSR matrices.
    // It is not optimized for CPU and in fact has several egregious inefficiencies from that perspective.
    // It is written to demonstrate the tile-wise parallel implementation on TT hardware.
    // (In fact, this version is not as numerically stable on CPU as the others).
    dense_matrix<T> tiled_spmm(dense_matrix<T> &B){
        /*
        Some Tenstorrent terminology:
        - tile: a contiguous block of 2048 bytes, usually 32x32 elements of a bfloat16 matrix.
                The APIs into the FPU and SFPU operate on tiles. The NoC architecture is optmized
                to send pages of sizes between 1-4MB, so the tile size is strategically set
                to benefit from both the compute engines and the dataflow architecture.
        - comptime args: the JIT compile time args to compute and dataflow kernels.
                         These can be set at the host's compile time or the host's runtime.

        */
        assert(W == B.H);
        dense_matrix<T> output(H, B.W);

        // FORALL parallelism on the block rows
        for (size_t i = 0; i < indptr.size() - 1; i++) { // runtime args
            for (size_t r = 0; r < R; r += TILE_SIZE) { // comptime args
                for (size_t p = 0; p < B.W; p += TILE_SIZE) { // comptime args
                    std::vector<T> output_tile(TILE_SIZE * TILE_SIZE, 0); // DST register
                    for (size_t idx = indptr[i]; idx < indptr[i + 1]; idx++) { // reading raw data from a CB
                        size_t j = indices[idx];                               // reading raw data from a CB
                        auto iter_start = data.begin() + idx * R * C;     // src0_addr, determined from args
                        auto iter_B_start = B.data.begin() + j * C * B.W; // src1_addr, determined from args
                        for (size_t c = 0; c < C; c += TILE_SIZE) { // comptime args
                            // begin matmul_tiles API call
                            for (size_t rr = r; rr < std::min(r + TILE_SIZE, R); rr++) {
                                for (size_t pp = p; pp < std::min(p + TILE_SIZE, B.W); pp++) {
                                    T sum = 0;
                                    for (size_t cc = c; cc < std::min(C, c + TILE_SIZE); cc++) {
                                        T a_val = *(iter_start + rr * C + cc);
                                        T b_val = *(iter_B_start + cc * B.W + pp);
                                        sum += a_val * b_val;
                                    }
                                    output_tile[(rr - r) * TILE_SIZE + pp - p] += sum;
                                }
                            }
                            // end matmul_tiles API call
                        }
                    }
                    // write tile to DRAM starting at tile i*R + r, p (output is dense, no more blocks)
                    // On TT:
                    // 1. pack DST reg to output CB
                    // 2. writer kernel pops from output CB
                    // 3. writer kernel NoC's to DRAM
                    for (size_t rr = r; rr < std::min(R, r + TILE_SIZE); rr++) {
                        for (size_t pp = p; pp < std::min(B.W, p + TILE_SIZE); pp++) {
                            *(output.data.begin() + (i * R + rr) * output.W + pp) = output_tile[(rr - r) * TILE_SIZE + pp - p];
                        }
                    }
                }
            }
        }
        return output;
    }

    dense_matrix<T> to_dense() {
        dense_matrix<T> output(H, W);
        for (size_t i = 0; i < indptr.size() - 1; i++) {
            for (size_t idx = indptr[i]; idx < indptr[i + 1]; idx++) {
                size_t j = indices[idx];
                auto iter_start = data.begin() + idx * R * C;
                for (size_t r = 0; r < R; r++) {
                    for (size_t c = 0; c < C; c++) {
                        output.data[(i * R + r) * W + (j * C + c)] = *(iter_start + r * C + c);
                    }
                }
            }
        }
        return output;
    }

    void print(std::ostream& os = std::cout) {
        os << "BSR Matrix:" << std::endl;
        os << "Size: " << H << " x " << W << std::endl;
        os << "Block Size: " << R << " x " << C << std::endl;
        os << "Number of blocks: " << nblocks << std::endl;
        os << "Indptr:" << std::endl;
        for (size_t i = 0; i < indptr.size(); ++i) {
            os << indptr[i] << " ";
        }
        os << std::endl;
        os << "Indices:" << std::endl;
        for (size_t i = 0; i < indices.size(); ++i) {
            os << indices[i] << " ";
        }
        os << std::endl;
        os << "Data:" << std::endl;
        for (size_t i = 0; i < data.size(); ++i) {
            os << data[i] << " ";
        }
        os << std::endl;
    }

    void pretty_print(std::ostream& os = std::cout) {
        os << "BSR Matrix:" << std::endl;
        os << "Size (H x W): " << H << " x " << W << std::endl;
        os << "Block Size (R x C): " << R << " x " << C << std::endl;
        os << "Number of blocks: " << nblocks << std::endl;
        os << "Indptr:" << std::endl;
        for (size_t i = 0; i < indptr.size(); ++i) {
            os << indptr[i] << " ";
        }
        os << std::endl;
        os << "Indices:" << std::endl;
        for (size_t i = 0; i < indices.size(); ++i) {
            os << indices[i] << " ";
        }
        os << std::endl;
        os << "Data:" << std::endl;
        for (size_t i = 0; i < H / R; i++) {
            for (size_t j = 0; j < W / C; j++) {
                char nz = '_';
                for (size_t idx = indptr[i]; idx < indptr[i + 1]; idx++) {
                    if (indices[idx] == j) {
                        nz = '*';
                        break;
                    }
                }
                os << nz << " ";
            }
            os << std::endl;
        }
        os << std::endl;
    }

};

#endif  // BSR_MATRIX_HPP
