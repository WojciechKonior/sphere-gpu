cmake_minimum_required(VERSION 3.14)
project(my_project LANGUAGES CXX CUDA)

# GoogleTest requires at least C++11
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "RELEASE")

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

# Tests
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
  tests
  tests/demo_test.cc
  tests/demo_test.cu
)

target_link_libraries(
  tests
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(tests)