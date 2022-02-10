#ifndef _WORLD_H
#define _WORLD_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdio.h>
#include <memory>
// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>

__global__ void addKernel(int *c, const int *a, const int *b);

__device__ class Point {
    int __x, __y;
public:
    __device__ Point(int x, int y): __x(x), __y(y) {}
    __device__ int getX() { return this->__x; }
    __device__ int getY() { return this->__y; }
};

class Particle{
public:
    float x, y, z;
    float vx, vy, vz;
};

class Data 
{
public:
    int *dev_a;
    int *dev_b;
    int *dev_c;
    Particle *dev_parts;
    unsigned int size;
    cudaError_t cudaStatus;

    Data(unsigned int size);
    ~Data();
    void copyFromHostToDevice(const int *host_data, int *dev_data);
    void copyFromDeviceToHost(const int *dev_data, int *host_data);
    void copyFromHostToDevice(const Particle *host_part);
    void copyFromDeviceToHost(Particle *host_part);
};

class Summator {
    std::shared_ptr<Data> dataptr;

public:
    Summator(std::shared_ptr<Data> shptr);
    void initializeParticles();
    cudaError_t propagateParticles(unsigned int size);
    cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
};

#endif /* _WORLD_H */