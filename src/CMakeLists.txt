add_library(pase
  profiler.cpp
  dispatcher.cpp
  cost_model.cpp
  feedback.cpp
  pase.cpp
  cpu/insertion.cpp
  cpu/introsort.cpp
  cpu/run_merge.cpp
  cpu/quicksort_3way.cpp
)

target_include_directories(pase PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}
)

target_compile_features(pase PUBLIC cxx_std_17)
