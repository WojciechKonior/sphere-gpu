cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(sphere_gpu LANGUAGES CXX CUDA)
enable_testing()
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin/nvcc")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests/googletest)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_executable(
    "Sphere" "src/Sphere.cu"
    "include/World.h" "src/World.cu"
    "include/Field.h" "src/Field.cu"
    "include/Source.h" "src/Source.cu"
    "include/Species.h" "src/Species.cu"
    "include/PotentialSolver.h" "src/PotentialSolver.cu"
    "include/Collisions.h" "src/Collisions.cu"
    "include/Thruster.h" "src/Thruster.cu"
    "include/Output.h" "src/Output.cu"
)

set_property(TARGET Sphere PROPERTY CUDA_SEPARABLE_COMPILATION ON)