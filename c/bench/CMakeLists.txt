cmake_minimum_required(VERSION 3.9)

project(bench VERSION 1.0.0 DESCRIPTION "binary operations benchmark")

add_executable(${PROJECT_NAME}
  bench.cpp)

target_link_libraries (${PROJECT_NAME} LINK_PUBLIC bops)

set_target_properties(${PROJECT_NAME} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF)
