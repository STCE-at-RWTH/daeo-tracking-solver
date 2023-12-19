cmake_minimum_required(VERSION 3.20)

project(
  tracking-daeo-solver
  VERSION 0.1
  LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# project structure settings
set(GFX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/gfx)
set(INPUT_DATA_PATH ${CMAKE_CURRENT_SOURCE_DIR}/data/in)
set(OUTPUT_DATA_PATH ${CMAKE_CURRENT_SOURCE_DIR}/data/out)
foreach(p IN ITEMS ${GFX_PATH} ${INPUT_DATA_PATH} ${OUTPUT_DATA_PATH})
  make_directory(${p})
endforeach()

# dco/c++
find_package(NAG_dco_cpp REQUIRED)
if(_DCO_LICENSE) # AND NOT DEFINED ENV{NAG_KUSARI_FILE})
  message(STATUS "Cannot use dco/c++ without license file. Be sure to set the 'NAG_KUSARI_FILE' environment variable.")
endif()

# fmt
find_package(fmt REQUIRED)

# compiler options
# Warn all
add_compile_options(-Wall)

# dco be like
add_compile_definitions(
  DCO_STD_COMPATABILITY
  DCO_AUTO_SUPPORT
  DCO_INTEROPERABILITY_BOOST_INTERVAL
  DCO_DISABLE_AVX2_WARNING
  DCO_EXT_NO_EIGEN
)

add_compile_definitions(
  DATA_INPUT_DIR="${INPUT_DATA_PATH}"
  DATA_OUTPUT_DIR="${OUTPUT_DATA_PATH}"
  GFX_PATH="${GFX_PATH}"
)

# get boost interval from the submodule
include_directories(interval/include)

set(objective_headers src/solver/objective.hpp)
set(bnb_solver_headers src/solver/settings.hpp src/solver/local_optima_solver.hpp)
set(daeo_solver_headers src/solver/daeo_solver.hpp)

get_target_property(NAG_dco_cpp_INCLUDE_DIRS NAG::dco_cpp INTERFACE_INCLUDE_DIRECTORIES)

add_executable(prototype src/idea_prototypes.cpp 
  ${objective_headers} 
  ${bnb_solver_headers} 
  ${daeo_solver_headers})
target_include_directories(prototype PUBLIC src/include ${NAG_dco_cpp_INCLUDE_DIRS})
target_link_libraries(prototype PRIVATE NAG::dco_cpp fmt::fmt)

add_executable(main src/main.cpp ${bnb_solver_headers})
target_include_directories(main PUBLIC src/include ${NAG_dco_cpp_INCLUDE_DIRS})
target_link_libraries(main PRIVATE NAG::dco_cpp fmt::fmt)