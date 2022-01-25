cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(sphere_gpu LANGUAGES CXX CUDA)
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin/nvcc")

# add_library(collisions "include/Collisions.h" "src/Collisions.cpp")
# add_library(field "include/Field.h" "src/Field.cpp")
# add_library(output "include/Output.h" "src/Output.cpp")
# add_library(potential-solver "include/PotentialSolver.h" "src/PotentialSolver.cpp")
# add_library(source "include/Source.h" "src/Source.cpp")
# add_library(species "include/Species.h" "src/Species.cpp")
# add_library(thruster "include/Thruster.h" "src/Thruster.cpp")
# add_library(world "include/World.h" "src/World.cpp")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_executable(sphere src/Sphere.cu)
set_property(TARGET sphere PROPERTY CUDA_SEPARABLE_COMPILATION ON)
# target_link_libraries(cuda PRIVATE particles)

# target_link_libraries(sphere PRIVATE collisions field output potential-solver source species thruster world)

