# Build sts_lightspeed (C++ apps + Python module) with the portable GCC in tools\mingw64.
# Output: sts_lightspeed\build\slaythespire.cp314-win_amd64.pyd, main.exe, test.exe
#
# Non-obvious flags:
#   -include algorithm             upstream relies on transitive <algorithm>; GCC 16 no longer provides it
#   CMAKE_POLICY_VERSION_MINIMUM   json/pybind11 submodules declare cmake_minimum_required < 3.5, which CMake 4 rejects
#   -static                        bake libstdc++/libgcc/winpthread into the .pyd; Python 3.8+ ignores PATH for DLLs
#   pybind11 submodule is pinned to v3.1.0 (upstream's v2.7.1 cannot build against Python >= 3.11)
$root = $PSScriptRoot
$env:PATH = "$root\tools\mingw64\bin;$env:PATH"
$env:TMP = "$root\tools\tmp"; $env:TEMP = $env:TMP
New-Item -ItemType Directory -Force $env:TMP | Out-Null
$py = (Get-Command python).Source -replace '\\', '/'
Set-Location "$root\sts_lightspeed"
cmake -S . -B build -G Ninja `
    "-DCMAKE_BUILD_TYPE=Release" `
    "-DCMAKE_CXX_FLAGS_RELEASE=-O3 -include algorithm" `
    "-DCMAKE_POLICY_VERSION_MINIMUM=3.5" `
    "-DCMAKE_SHARED_LINKER_FLAGS=-static" "-DCMAKE_MODULE_LINKER_FLAGS=-static" "-DCMAKE_EXE_LINKER_FLAGS=-static" `
    "-DPYBIND11_FINDPYTHON=ON" "-DPython_EXECUTABLE=$py"
if ($LASTEXITCODE -eq 0) { cmake --build build -j $env:NUMBER_OF_PROCESSORS }
