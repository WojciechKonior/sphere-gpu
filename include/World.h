#pragma once
#include <iostream>

__host__ void world_say_hello();
__global__ void addKernel(int *c, const int *a, const int *b);