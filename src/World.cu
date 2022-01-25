#include <World.h>

__host__ void world_say_hello(){
	std::cout<<"World say hello"<<std::endl;
}

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}