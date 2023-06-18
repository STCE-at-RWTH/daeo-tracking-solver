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

if(_DCO_LICENSE) # AND NOT DEFINED ENV{NAG_KUSARI_FILE})
  message(STATUS "Cannot use dco/c++ without license file. Be sure to set the 'NAG_KUSARI_FILE' environment variable.")
endif()

find_package(fmt)

# compiler options
add_compile_options(-Wall)

add_compile_definitions(
  DCO_STD_COMPATABILITY
  DCO_AUTO_SUPPORT
  DCO_INTEROPERABILITY_BOOST_INTERVAL
  DCO_DISABLE_AVX2_WARNING
)

set(bnb_solver_headers src/solver/bnb_settings.hpp src/solver/local_optima_bnb.hpp)
add_executable(prototype src/idea_prototypes.cpp ${bnb_solver_headers})

get_target_property(NAG_dco_cpp_INCLUDE_DIRS NAG::dco_cpp INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(prototype PUBLIC include ${NAG_dco_cpp_INCLUDE_DIRS})
target_link_libraries(prototype PRIVATE NAG::dco_cpp)
