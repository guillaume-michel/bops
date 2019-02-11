cmake_minimum_required(VERSION 3.9)

project(train VERSION 1.0.0 DESCRIPTION "binary operations train program")

add_executable(${PROJECT_NAME}
  train.cpp
)

target_link_libraries (${PROJECT_NAME} LINK_PUBLIC bops mnist)

set_target_properties(${PROJECT_NAME} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF)
