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

target_include_directories(test_correctness PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_profiler PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_cost_model PRIVATE ${CMAKE_SOURCE_DIR}/include)
