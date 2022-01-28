cmake_minimum_required(VERSION 3.8)
set(This ExampleTests)
set(Sources ExampleTests.cpp)

add_executable(${This} ${Sources})

target_link_libraries(${This} PUBLIC 
    gtest_main 
    "G:/My Drive/doktorat/SPHERE_GPU/build/Debug/Sphere.lib"
)

add_test(
    NAME ${This}
    COMMAND ${This}
)