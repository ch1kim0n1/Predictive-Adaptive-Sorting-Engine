set(PASE_SOURCES
  profiler.cpp
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
  cpu/run_merge.cpp
  cpu/quicksort_3way.cpp
)

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

target_include_directories(pase PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}
)

target_link_libraries(pase PUBLIC nlohmann_json::nlohmann_json)

target_compile_features(pase PUBLIC cxx_std_17)

if(PASE_ENABLE_CUDA)
  target_compile_definitions(pase PUBLIC PASE_WITH_CUDA=1)
  target_link_libraries(pase PRIVATE CUDA::cudart)
  if(PASE_GPU_SORT_USE_CUB)
    # CUB is header-only and ships with the CUDA toolkit includes.
    target_compile_definitions(pase PUBLIC PASE_GPU_SORT_USE_CUB=1)
  endif()
endif()
