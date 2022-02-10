#include <gtest/gtest.h>
#include <World.h>

TEST(dataClass, dataClassTest)
{
    const int arraySize = 5;
    const int a[arraySize] = {1, 2, 3, 4, 5};
    const int b[arraySize] = {10, 20, 30, 40, 50};
    int c[arraySize] = {0};

    std::shared_ptr<Data> dataptr(new Data(arraySize));
    std::shared_ptr<Summator> sum(new Summator(dataptr));
    cudaError_t cudaStatus = sum->addWithCuda(c, a, b, arraySize);

    EXPECT_EQ(dataptr->cudaStatus, cudaSuccess);
    for (int i = 0; i < arraySize; i++)
    {
        EXPECT_EQ(a[i] + b[i], c[i]);
    }
}

int main()
{
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}