#pragma once
#include <tools/profiler/kernel_profiler.hpp>

// Compile-time togglable profiling zones for SpMM kernels.
//
// Usage:  SPMM_PROFILE_ZONE(PROFILE_READ_IN0, "zone name", [&]() { ... });
//
// `enabled` must be a preprocessor token that expands to 1 or 0.
// The token-paste selects between a variant with DeviceZoneScopedN (1)
// and one without (0), keeping _Pragma out of if-constexpr.
//
// Zone flags are #ifndef-defaulted to 1 in each kernel file and can be
// overridden to 0 via CreateKernel's .defines map (fed by env vars).

// ── Dataflow kernel variants ──
#define _SPMM_PROFILE_ZONE_1(name, fn)  { DeviceZoneScopedN(name); (fn)(); }
#define _SPMM_PROFILE_ZONE_0(name, fn)  { (fn)(); }

// ── Compute kernel variants ──
#define _SPMM_PROFILE_ZONE_COMPUTE_1(name, fn)  { UNPACK(DeviceZoneScopedN(name)); (fn)(); }
#define _SPMM_PROFILE_ZONE_COMPUTE_0(name, fn)  { (fn)(); }

// ── Token-paste dispatch ──
#define _SPMM_PZ_CAT2(a, b) a##b
#define _SPMM_PZ_CAT(a, b)  _SPMM_PZ_CAT2(a, b)

#define SPMM_PROFILE_ZONE(enabled, name, fn) \
    _SPMM_PZ_CAT(_SPMM_PROFILE_ZONE_, enabled)(name, fn)

#define SPMM_PROFILE_ZONE_COMPUTE(enabled, name, fn) \
    _SPMM_PZ_CAT(_SPMM_PROFILE_ZONE_COMPUTE_, enabled)(name, fn)
