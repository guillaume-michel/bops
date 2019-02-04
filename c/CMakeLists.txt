cmake_minimum_required(VERSION 3.9)

project(bench VERSION 1.0.0 DESCRIPTION "binary operations benchmark")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror -Wpedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -march=native")

include(GNUInstallDirs)

add_subdirectory(lib)

add_executable(${PROJECT_NAME}
  bench.cpp)

target_link_libraries (${PROJECT_NAME} LINK_PUBLIC bops)

set_target_properties(${PROJECT_NAME} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF)
