cmake_minimum_required(VERSION 3.20)
project(swarmBalancer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(SWARM_BUILD_TESTS "Build C++ unit tests" ON)
option(SWARM_BUILD_TOOLS "Build benchmark / CLI tools" ON)
option(SWARM_NATIVE_ARCH "Tune for the host CPU (-march=native / -mcpu=native). Turn OFF when cross-compiling for a target board." ON)

# Put every runtime artifact (dll/so/exe) in one flat directory so the Python
# ctypes loader (python/swarm/brain/native.py) can find the library easily.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(SWARM_BUILD_TESTS)
  enable_testing()
endif()

add_subdirectory(cpp)
