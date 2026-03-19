cmake_minimum_required(VERSION 3.17)
project(PASE VERSION 1.0.0 LANGUAGES CXX)

option(PASE_ENABLE_CUDA "Build with NVIDIA CUDA GPU sort" OFF)

enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
set(GTEST_USE_OWN_TR1_TUPLE 1)
FetchContent_MakeAvailable(googletest)

# Google Benchmark
FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.8.3
)
FetchContent_MakeAvailable(benchmark)

# PASE library
add_subdirectory(src)

# Tests
add_subdirectory(tests)

# Benchmarks
add_subdirectory(bench)
