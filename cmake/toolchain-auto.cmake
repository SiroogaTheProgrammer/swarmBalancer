# Toolchain auto-detection for Windows, included before project().
#
# Why: on this class of machine (Windows on ARM64, no Visual Studio) three
# things routinely go wrong when CMake is left to its own devices:
#   * it picks `C:\Program Files\LLVM\clang++` (MSVC-targeting) which cannot
#     link without the Windows SDK -> "could not open kernel32.lib";
#   * llvm-mingw / ninja were installed with winget but are not on the PATH
#     of the shell (or of VS Code) that runs cmake;
#   * the shell runs emulated and reports the CPU as AMD64, so "native" is
#     ambiguous. Python may be an x64 build that needs an x64 DLL.
# This file resolves all three: it reads the true machine architecture from
# the registry, finds llvm-mingw + ninja on PATH or in the winget package
# directory, and sets the compiler for the requested SWARM_TARGET_ARCH.
#
# Nothing here runs if the user already chose a compiler (CMAKE_CXX_COMPILER,
# CC/CXX, or a toolchain file) or is not on Windows.

set(SWARM_TARGET_ARCH "native" CACHE STRING
    "CPU to build for on Windows: native (the real machine CPU), arm64, x64, x86, python (whatever the Python interpreter is)")
set_property(CACHE SWARM_TARGET_ARCH PROPERTY STRINGS native arm64 x64 x86 python)

if(NOT CMAKE_HOST_WIN32)
  return()
endif()
if(NOT DEFINED SWARM_RESOLVED_ARCH AND (CMAKE_CXX_COMPILER OR CMAKE_C_COMPILER OR CMAKE_TOOLCHAIN_FILE
                                        OR DEFINED ENV{CXX} OR DEFINED ENV{CC}))
  return()  # the user picked a toolchain; respect it
endif()

# ---- 1. the real machine architecture ---------------------------------------------------------
# CMAKE_HOST_SYSTEM_PROCESSOR comes from PROCESSOR_ARCHITECTURE, which an x64-emulated parent shell
# passes down as AMD64. The registry value is written at OS install time and not redirected.
set(_swarm_host_arch "")
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
  cmake_host_system_information(RESULT _swarm_reg_arch
      QUERY WINDOWS_REGISTRY "HKLM/SYSTEM/CurrentControlSet/Control/Session Manager/Environment"
      VALUE "PROCESSOR_ARCHITECTURE" ERROR_VARIABLE _swarm_reg_err)
  if(NOT _swarm_reg_err)
    set(_swarm_host_arch "${_swarm_reg_arch}")
  endif()
endif()
if(NOT _swarm_host_arch)
  set(_swarm_host_arch "$ENV{PROCESSOR_ARCHITEW6432}")
  if(NOT _swarm_host_arch)
    set(_swarm_host_arch "$ENV{PROCESSOR_ARCHITECTURE}")
  endif()
endif()
string(TOLOWER "${_swarm_host_arch}" _swarm_host_arch)
if(_swarm_host_arch MATCHES "arm64|aarch64")
  set(_swarm_host_arch arm64)
elseif(_swarm_host_arch MATCHES "amd64|x86_64|x64")
  set(_swarm_host_arch x64)
else()
  set(_swarm_host_arch x86)
endif()

# ---- 2. which architecture to build -----------------------------------------------------------
set(_swarm_arch "${SWARM_TARGET_ARCH}")
if(_swarm_arch STREQUAL "native")
  set(_swarm_arch "${_swarm_host_arch}")
elseif(_swarm_arch STREQUAL "python")
  # ctypes can only load a DLL of the interpreter's own architecture; ask the interpreter.
  find_program(_swarm_python NAMES python python3 NO_CACHE)
  if(_swarm_python)
    execute_process(COMMAND "${_swarm_python}" -c "import sysconfig; print(sysconfig.get_platform())"
                    OUTPUT_VARIABLE _swarm_pyplat OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  endif()
  if(_swarm_pyplat MATCHES "arm64")
    set(_swarm_arch arm64)
  elseif(_swarm_pyplat MATCHES "amd64")
    set(_swarm_arch x64)
  elseif(_swarm_pyplat STREQUAL "win32")
    set(_swarm_arch x86)
  else()
    message(WARNING "SWARM_TARGET_ARCH=python but no python interpreter found; building for ${_swarm_host_arch}")
    set(_swarm_arch "${_swarm_host_arch}")
  endif()
endif()

if(_swarm_arch STREQUAL "arm64")
  set(_swarm_triple aarch64-w64-mingw32)
elseif(_swarm_arch STREQUAL "x64")
  set(_swarm_triple x86_64-w64-mingw32)
elseif(_swarm_arch STREQUAL "x86")
  set(_swarm_triple i686-w64-mingw32)
else()
  message(FATAL_ERROR "SWARM_TARGET_ARCH=${SWARM_TARGET_ARCH} is not one of native, arm64, x64, x86, python")
endif()

# A build directory is tied to one compiler; changing the architecture needs a fresh one.
if(DEFINED SWARM_RESOLVED_ARCH)
  if(NOT SWARM_RESOLVED_ARCH STREQUAL _swarm_arch)
    message(FATAL_ERROR "${CMAKE_BINARY_DIR} was configured for ${SWARM_RESOLVED_ARCH}; "
                        "to build for ${_swarm_arch} delete it or use another build directory (-B).")
  endif()
  return()  # already configured; the compilers are in the cache
endif()

# ---- 3. find llvm-mingw (PATH first, then where winget puts it) ---------------------------------
file(GLOB _swarm_mingw_dirs
     "$ENV{LOCALAPPDATA}/Microsoft/WinGet/Packages/MartinStorsjo.LLVM-MinGW*/llvm-mingw-*/bin"
     "$ENV{ProgramFiles}/llvm-mingw*/bin" "C:/llvm-mingw*/bin" "$ENV{USERPROFILE}/llvm-mingw*/bin")
find_program(_swarm_cxx NAMES ${_swarm_triple}-clang++ HINTS ${_swarm_mingw_dirs} NO_CACHE)
find_program(_swarm_cc NAMES ${_swarm_triple}-clang HINTS ${_swarm_mingw_dirs} NO_CACHE)

if(NOT _swarm_cxx OR NOT _swarm_cc)
  message(FATAL_ERROR
    "No llvm-mingw compiler for ${_swarm_arch} (${_swarm_triple}-clang++) found on PATH or under "
    "%LOCALAPPDATA%\\Microsoft\\WinGet\\Packages.\n"
    "Install it (no admin rights needed):\n"
    "    winget install --id MartinStorsjo.LLVM-MinGW.UCRT --scope user --silent --accept-package-agreements --accept-source-agreements\n"
    "then run:  python dev.py build\n"
    "(or set CMAKE_CXX_COMPILER yourself to use another toolchain)")
endif()

set(CMAKE_C_COMPILER "${_swarm_cc}" CACHE FILEPATH "C compiler" FORCE)
set(CMAKE_CXX_COMPILER "${_swarm_cxx}" CACHE FILEPATH "C++ compiler" FORCE)

# ---- 4. ninja (only if a Ninja generator was asked for and it is not on PATH) -------------------
if(CMAKE_GENERATOR MATCHES "Ninja" AND NOT CMAKE_MAKE_PROGRAM)
  file(GLOB _swarm_ninja_dirs "$ENV{LOCALAPPDATA}/Microsoft/WinGet/Packages/Ninja-build.Ninja*"
                              "$ENV{LOCALAPPDATA}/Microsoft/WinGet/Links")
  find_program(CMAKE_MAKE_PROGRAM NAMES ninja HINTS ${_swarm_ninja_dirs})
  if(NOT CMAKE_MAKE_PROGRAM)
    message(FATAL_ERROR "ninja not found. Install it with:\n"
      "    winget install --id Ninja-build.Ninja --scope user --silent --accept-package-agreements --accept-source-agreements")
  endif()
endif()

# Cross-compiling (e.g. an x64 DLL for an emulated Python on an ARM64 PC): never tune for the host CPU.
if(NOT _swarm_arch STREQUAL _swarm_host_arch)
  set(SWARM_NATIVE_ARCH OFF CACHE BOOL "Tune for the host CPU" FORCE)
endif()

set(SWARM_RESOLVED_ARCH "${_swarm_arch}" CACHE INTERNAL "architecture this build directory targets")
message(STATUS "swarm toolchain: machine is ${_swarm_host_arch}, building for ${_swarm_arch} with ${_swarm_cxx}")
