#pragma once
#include <fcntl.h>    // open
#include <unistd.h>   // dup, dup2, close, STDOUT_FILENO, vdprintf
#include <cstdarg>    // va_list, va_start, va_end
#include <cstdio>     // printf
#include <cstdint>
#include <memory>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/bfloat16.hpp>
// #include <tt-metalium/test_tiles.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "bmm_op.hpp"
#include <tt-metalium/tilize_utils.hpp>



#include <unordered_map>
#include <variant>
#include <vector>
#include <tuple>
#include <filesystem> // for emitting test output

#include "bsr_matrix.hpp"
#include "tt-metalium/core_coord.hpp"
