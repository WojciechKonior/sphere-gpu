#include <gtest/gtest.h>
#include <World.h>

TEST(addKernel, AddingTwoArraysTest) {
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    
    for(int i = 0; i<arraySize; i++){
      EXPECT_EQ(a[i]+b[i], c[i]);
    }
}

TEST(WorldHeader, world_say_helloMethodTest) {
    EXPECT_EQ(world_say_hello(), "World say hello");
}