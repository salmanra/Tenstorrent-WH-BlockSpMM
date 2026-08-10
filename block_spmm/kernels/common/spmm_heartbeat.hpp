#pragma once

#include "api/debug/dprint.h"

#ifndef BSPMM_HEARTBEAT
#define BSPMM_HEARTBEAT 0
#endif

#if BSPMM_HEARTBEAT
#define BSPMM_HB_DATA(...) DPRINT(__VA_ARGS__)
#define BSPMM_HB_MATH(...) DPRINT_MATH(__VA_ARGS__)
#define BSPMM_HB_WP(x) WAYPOINT(x)
#else
#define BSPMM_HB_DATA(...) \
    do {                   \
    } while (0)
#define BSPMM_HB_MATH(...) \
    do {                   \
    } while (0)
#define BSPMM_HB_WP(x) \
    do {               \
    } while (0)
#endif
