option(PASE_ENABLE_SIMD_INTRINSICS
  "Use AVX2 (x86_64) / NEON (AArch64) intrinsics in simd_profiler.cpp (scalar fallback if off)"
  ON)

set(PASE_SOURCES
  profiler.cpp
  simd_profiler.cpp
  dispatcher.cpp
  cost_model.cpp
  strategies.cpp
  config_loader.cpp
  runtime.cpp
  plan.cpp
  feedback.cpp
  threshold_tuner.cpp
  pase.cpp
  cpu/insertion.cpp
  cpu/introsort.cpp
  cpu/introsort_simd.cpp
  cpu/run_merge.cpp
  cpu/quicksort_3way.cpp
  ml_tuner.cpp
  distributed/distributed_sort.cpp
)

option(PASE_ENABLE_ML_TUNING "Load optional ML threshold JSON at runtime" OFF)
option(PASE_ENABLE_MPI "Build optional MPI distributed sort" OFF)

option(PASE_GPU_SORT_USE_CUB
  "Use CUB DeviceRadixSort for int keys (experimental; default is Thrust)" OFF)

if(PASE_ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75" CACHE STRING "CUDA GPU architectures")
  endif()
  find_package(CUDAToolkit REQUIRED)
  list(APPEND PASE_SOURCES gpu/gpu_sort.cu)
else()
  list(APPEND PASE_SOURCES gpu/gpu_stub.cpp)
endif()

add_library(pase ${PASE_SOURCES})

option(PASE_ENABLE_BUILT_INTROSORT_INT
  "Use PASE median-of-three Hoare introsort for int (Phase 5); default std::sort"
  OFF)
if(PASE_ENABLE_BUILT_INTROSORT_INT)
  target_compile_definitions(pase PRIVATE PASE_SIMD_SORT_BUILT_INTROSORT=1)
endif()

if(PASE_ENABLE_ML_TUNING)
  target_compile_definitions(pase PRIVATE PASE_WITH_ML_TUNING=1)
endif()

if(PASE_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  target_compile_definitions(pase PUBLIC PASE_WITH_MPI=1)
  target_link_libraries(pase PRIVATE MPI::MPI_CXX)
endif()

target_include_directories(pase PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}
)

target_link_libraries(pase PUBLIC nlohmann_json::nlohmann_json)

target_compile_features(pase PUBLIC cxx_std_17)

if(PASE_ENABLE_SIMD_INTRINSICS)
  if(MSVC)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|arm64|aarch64")
      target_compile_definitions(pase PRIVATE PASE_SIMD_NEON=1)
    else()
      set_source_files_properties(simd_profiler.cpp PROPERTIES COMPILE_OPTIONS
        "/arch:AVX2")
      target_compile_definitions(pase PRIVATE PASE_SIMD_AVX2=1)
    endif()
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
    target_compile_definitions(pase PRIVATE PASE_SIMD_NEON=1)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64")
    set_source_files_properties(simd_profiler.cpp
      PROPERTIES COMPILE_OPTIONS "-mavx2")
    target_compile_definitions(pase PRIVATE PASE_SIMD_AVX2=1)
  endif()
endif()

if(PASE_ENABLE_CUDA)
  target_compile_definitions(pase PUBLIC PASE_WITH_CUDA=1)
  target_link_libraries(pase PRIVATE CUDA::cudart)
  if(PASE_GPU_SORT_USE_CUB)
    # CUB is header-only and ships with the CUDA toolkit includes.
    target_compile_definitions(pase PUBLIC PASE_GPU_SORT_USE_CUB=1)
  endif()
endif()
