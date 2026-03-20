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

# Google Benchmark (library only by default; upstream self-tests can fail on some hosts, e.g. macOS CV output)
option(
  PASE_BUILD_BENCHMARK_UPSTREAM_TESTS
  "Register Google Benchmark library self-tests in ctest (not PASE coverage)"
  OFF)
if(PASE_BUILD_BENCHMARK_UPSTREAM_TESTS)
  set(BENCHMARK_ENABLE_TESTING ON CACHE BOOL "benchmark self-tests" FORCE)
else()
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "benchmark self-tests" FORCE)
endif()
FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.8.3
)
FetchContent_MakeAvailable(benchmark)

# nlohmann/json (Phase 4+ config)
FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(json)

# PASE library
add_subdirectory(src)

# Tests
add_subdirectory(tests)

# Benchmarks
add_subdirectory(bench)
