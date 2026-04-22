#!/usr/bin/env bash
#
# Create symlinks at $TT_METAL_HOME pointing at the Tracy host-side tools
# built by tt-metal's `build_metal.sh --enable-profiler`.
#
# The profile and export binaries invoke these tools via std::system() with
# relative paths (./capture-release, ./csvexport-release), so they must be
# reachable from the CWD in which the reproduction scripts run — which, by
# convention, is $TT_METAL_HOME (the reproduction scripts cd there on entry).
#
# Usage:
#   TT_METAL_HOME=/path/to/tt-metal bash setup_tracy_symlinks.sh

set -euo pipefail

: "${TT_METAL_HOME:?TT_METAL_HOME must be set to your tt-metal checkout path}"

if [[ ! -d "$TT_METAL_HOME" ]]; then
    echo "TT_METAL_HOME does not point at a directory: $TT_METAL_HOME" >&2
    exit 1
fi

CAPTURE_SRC="tt_metal/third_party/tracy/capture/build/unix/capture-release"
EXPORT_SRC="tt_metal/third_party/tracy/csvexport/build/unix/csvexport-release"

fail=0
for entry in \
    "capture-release:${CAPTURE_SRC}" \
    "csvexport-release:${EXPORT_SRC}"
do
    link="${entry%%:*}"
    target="${entry#*:}"
    abs_target="$TT_METAL_HOME/$target"

    if [[ ! -x "$abs_target" ]]; then
        echo "  [fail]    $link — source not built: $target" >&2
        echo "            Run './build_metal.sh --enable-profiler --build-programming-examples' from \$TT_METAL_HOME first." >&2
        fail=1
        continue
    fi

    ln -sf "$target" "$TT_METAL_HOME/$link"
    echo "  [linked]  $link -> $target"
done

if [[ $fail -ne 0 ]]; then
    exit 1
fi

echo "Tracy symlinks ready at \$TT_METAL_HOME."
