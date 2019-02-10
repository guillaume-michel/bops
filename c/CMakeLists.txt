cmake_minimum_required(VERSION 3.9)

project(top_project VERSION 1.0.0 DESCRIPTION "Top CMakeLists.txt")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror -Wpedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -march=native")

add_subdirectory(lib)
add_subdirectory(bench)
add_subdirectory(train)
