cmake_minimum_required(VERSION 3.9)

project(bench VERSION 1.0.0 DESCRIPTION "binary operations benchmark")

add_executable(bench bench.cpp)

target_link_libraries (bench PUBLIC bops)

set_target_properties(bench PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF)

add_executable(bench_dense_f32 bench_dense_f32.cpp)

target_include_directories(bench_dense_f32 PRIVATE /usr/local/Cellar/openblas/0.3.6_1/include)
target_link_directories(bench_dense_f32 PRIVATE /usr/local/Cellar/openblas/0.3.6_1/lib)
target_link_libraries (bench_dense_f32 PRIVATE openblas)

set_target_properties(bench_dense_f32 PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF)
