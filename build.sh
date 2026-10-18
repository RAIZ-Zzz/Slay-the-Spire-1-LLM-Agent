#!/bin/bash
# Build the sts_lightspeed Python module on Linux (e.g. the TC2 cluster). Linux twin of build.ps1.
#   ./build.sh            release -> sts_lightspeed/build/
#   ./build.sh debug      symbols + bounds checks + UBSan -> sts_lightspeed/build-debug/
#                         (point paths.sim_build_dir in sts1/config.json at it to use it)
# PY selects the Python to build for (default: python on PATH); the module only loads in that Python.
#
# Non-obvious flags:
#   -include algorithm                    upstream relies on transitive <algorithm>
#   -static-libstdc++ -static-libgcc      jobs may run without the gcc module that provides a new libstdc++
#   debug: -D_GLIBCXX_ASSERTIONS -fsanitize=undefined
#                                         Linux exposed UB that Windows hid (uninitialized Player::cc)
set -e
cd "$(dirname "$0")/sts_lightspeed"
PY=${PY:-$(command -v python)}
FLAGS="-O3 -include algorithm"
LINK="-static-libstdc++ -static-libgcc"
DIR=build
if [ "$1" = debug ]; then
    FLAGS="-O1 -g -fno-omit-frame-pointer -include algorithm -D_GLIBCXX_ASSERTIONS -fsanitize=undefined"
    LINK="$LINK -static-libubsan -fsanitize=undefined"
    DIR=build-debug
fi
cmake -S . -B $DIR -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CXX_FLAGS_RELEASE=$FLAGS" \
    "-DCMAKE_MODULE_LINKER_FLAGS=$LINK" -DPYBIND11_FINDPYTHON=ON "-DPython_EXECUTABLE=$PY"
cmake --build $DIR --target slaythespire -j "${JOBS:-8}"
