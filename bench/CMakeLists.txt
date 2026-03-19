add_executable(bench_main bench_main.cpp gen_datasets.cpp)
target_include_directories(bench_main PRIVATE ${CMAKE_SOURCE_DIR}/bench)
target_link_libraries(bench_main pase benchmark::benchmark)
