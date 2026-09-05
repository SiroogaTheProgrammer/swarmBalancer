# The brain: a static core library (kernels + layers + model loader), a shared
# library exposing the C ABI for Python/ctypes, plus tools and tests.

add_library(swarm_core STATIC
  src/matmul.cpp
  src/nn.cpp
  src/model.cpp
)
target_include_directories(swarm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(swarm_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(swarm_core PUBLIC -Wall -Wextra -Wpedantic)
  # -O3 for auto-vectorisation; fp-contract lets the compiler fuse a*b+c into FMA.
  target_compile_options(swarm_core PUBLIC $<$<CONFIG:Release,RelWithDebInfo>:-O3 -ffp-contract=fast -fno-math-errno>)
  if(SWARM_NATIVE_ARCH)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(-march=native SWARM_HAS_MARCH_NATIVE)
    if(SWARM_HAS_MARCH_NATIVE)
      target_compile_options(swarm_core PUBLIC -march=native)
    else()
      check_cxx_compiler_flag(-mcpu=native SWARM_HAS_MCPU_NATIVE)
      if(SWARM_HAS_MCPU_NATIVE)
        target_compile_options(swarm_core PUBLIC -mcpu=native)
      endif()
    endif()
  endif()
elseif(MSVC)
  target_compile_options(swarm_core PUBLIC /W4 /EHsc $<$<CONFIG:Release>:/O2 /fp:fast>)
endif()

# ---- shared library with the C ABI (loaded by python/swarm/brain/native.py) ----
add_library(swarm_brain SHARED src/c_api.cpp)
target_link_libraries(swarm_brain PRIVATE swarm_core)
target_compile_definitions(swarm_brain PRIVATE SWARM_BUILDING_DLL)
set_target_properties(swarm_brain PROPERTIES PREFIX "" OUTPUT_NAME "swarm_brain" CXX_VISIBILITY_PRESET hidden)
if(MINGW)
  # Self-contained DLL: no dependency on libc++.dll / libunwind.dll next to python.exe.
  target_link_options(swarm_brain PRIVATE -static)
endif()

# ---- tools ----
if(SWARM_BUILD_TOOLS)
  add_executable(bench_matmul tools/bench_matmul.cpp)
  target_link_libraries(bench_matmul PRIVATE swarm_core)

  add_executable(run_model tools/run_model.cpp)
  target_link_libraries(run_model PRIVATE swarm_core)

  if(MINGW)
    target_link_options(bench_matmul PRIVATE -static)
    target_link_options(run_model PRIVATE -static)
  endif()
endif()

# ---- tests (dependency-free: plain asserts, registered with CTest) ----
if(SWARM_BUILD_TESTS)
  foreach(t IN ITEMS test_matmul test_nn)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE swarm_core)
    if(MINGW)
      target_link_options(${t} PRIVATE -static)
    endif()
    add_test(NAME ${t} COMMAND ${t})
  endforeach()
endif()
