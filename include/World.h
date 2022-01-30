#ifndef _WORLD_H
#define _WORLD_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdio.h>

__host__ std::string world_say_hello();
__global__ void addKernel(int *c, const int *a, const int *b);
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

#endif /* _WORLD_H */