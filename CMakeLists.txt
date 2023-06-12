cmake_minimum_required(VERSION 3.20)

project(
  tracking-daeo-solver
  VERSION 0.1
  LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# dco/c++
find_package(NAG_dco_cpp REQUIRED)

if(NOT DEFINED ENV{NAG_KUSARI_FILE})
  message(FATAL_ERROR "Cannot use dco/c++ without license file.")
endif()

# compiler options
add_compile_options(-Wall)

add_executable(test_ex src/main.cpp)
target_link_libraries(test_ex PUBLIC NAG::dco_cpp)
