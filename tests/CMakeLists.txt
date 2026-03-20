enable_testing()

add_executable(test_correctness test_correctness.cpp)
target_link_libraries(test_correctness pase GTest::gtest_main)
add_test(NAME Correctness COMMAND test_correctness)

add_executable(test_profiler test_profiler.cpp)
target_link_libraries(test_profiler pase GTest::gtest_main)
add_test(NAME Profiler COMMAND test_profiler)

add_executable(test_cost_model test_cost_model.cpp)
target_link_libraries(test_cost_model pase GTest::gtest_main)
add_test(NAME CostModel COMMAND test_cost_model)

add_executable(test_gpu test_gpu.cpp)
target_link_libraries(test_gpu pase GTest::gtest_main)
add_test(NAME GpuSort COMMAND test_gpu)

add_executable(test_config_loader test_config_loader.cpp)
target_link_libraries(test_config_loader pase GTest::gtest_main)
add_test(NAME ConfigLoader COMMAND test_config_loader)

add_executable(test_performance_regression test_performance_regression.cpp)
target_link_libraries(test_performance_regression pase GTest::gtest_main)
add_test(NAME PerformanceRegression COMMAND test_performance_regression)

target_include_directories(test_correctness PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_profiler PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_cost_model PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_gpu PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_config_loader PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_performance_regression PRIVATE ${CMAKE_SOURCE_DIR}/include)
