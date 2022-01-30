cmake_minimum_required(VERSION 3.14)
project(my_project LANGUAGES CXX CUDA)

# GoogleTest requires at least C++11
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "RELEASE")
set(
  HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/include/World.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/Field.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/Source.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/Species.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/PotentialSolver.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/Collisions.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/Thruster.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/Output.h
)
set(
  SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/World.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Field.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Source.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Species.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/PotentialSolver.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Collisions.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Thruster.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Output.cu
)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_executable(
  Sphere
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Sphere.cu
  ${HEADERS}
  ${SOURCES}
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

set(
  TESTS
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/demo_test.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/demo_test.cu
)

add_executable(
  tests
  ${TESTS}
  ${HEADERS}
  ${SOURCES}
)

target_link_libraries(
  tests
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(tests)