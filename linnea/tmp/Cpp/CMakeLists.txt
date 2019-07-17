cmake_minimum_required(VERSION 3.0)
project(tmp)

find_package(LibTests REQUIRED)

add_executable(runner_t1 runner_t1.cpp)
target_link_libraries(runner_t1 PRIVATE libtests)
set_target_properties(runner_t1 PROPERTIES CXX_STANDARD 14)

add_executable(runner_t24 runner_t24.cpp)
target_link_libraries(runner_t24 PRIVATE libtests)
set_target_properties(runner_t24 PROPERTIES CXX_STANDARD 14)

