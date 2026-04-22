# Host-side integration assets

Files in this directory are **applied to the host `tt-metal` checkout**, not to
the submodule itself. They adjust the host project just enough to register
this submodule with its build system.

## `programming_examples.patch`

A one-line unified diff that adds

```cmake
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/Tenstorrent-WH-BlockSpMM)
```

to `tt_metal/programming_examples/CMakeLists.txt`.

### Baseline

`tt-metal` at tag `v0.63.0`. Other tt-metal versions may have different
context lines around the insertion point; if the patch fails to apply,
re-generate it or edit the target file by hand.

### Applying

From the root of your tt-metal checkout (i.e. from `$TT_METAL_HOME`):

```bash
patch -p1 < tt_metal/programming_examples/Tenstorrent-WH-BlockSpMM/host_integration/programming_examples.patch
```

### Reverting

From the same location:

```bash
patch -p1 -R < tt_metal/programming_examples/Tenstorrent-WH-BlockSpMM/host_integration/programming_examples.patch
```
