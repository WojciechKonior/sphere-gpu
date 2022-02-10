#include <World.h>

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;

    Point *p = new Point(1, 2);
    c[i] = p->getX() + p->getY() - 3 + a[i] + b[i];

    delete p;
}

__global__ void propagateParticlesKernel(Particle *particles)
{
    int i = threadIdx.x;
    particles[i].x = 1;
    particles[i].y = 2;
    particles[i].z = 3;
    particles[i].vx = 4;
    particles[i].vy = 5;
    particles[i].vz = 6;
}

Data::Data(unsigned int size)
{
    this->size = size;
    fprintf(stderr, "Data Constructor!!!\n");

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void **)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
    }

    cudaStatus = cudaMalloc((void **)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
    }

    cudaStatus = cudaMalloc((void **)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
    }

    cudaStatus = cudaMalloc((void **)&dev_parts, size * sizeof(Particle));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
    }
}

Data::~Data()
{
    fprintf(stderr, "Data Destructor!!!\n");
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_parts);
}

void Data::copyFromHostToDevice(const int *host_data, int *dev_data)
{
    cudaStatus = cudaMemcpy(dev_data, host_data, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
    }
}

void Data::copyFromDeviceToHost(const int *dev_data, int *host_data)
{
    cudaStatus = cudaMemcpy(host_data, dev_data, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
    }
}

void Data::copyFromHostToDevice(const Particle *host_part)
{
    cudaStatus = cudaMemcpy(dev_parts, host_part, size * sizeof(Particle), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
    }
}

void Data::copyFromDeviceToHost(Particle *host_part)
{
    cudaStatus = cudaMemcpy(host_part, dev_parts, size * sizeof(Particle), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
    }
}

Summator::Summator(std::shared_ptr<Data> shptr)
{
    dataptr = shptr;
    this->initializeParticles();
}

void Summator::initializeParticles()
{
    // const unsigned int s = const unsigned int(dataptr->size);
    Particle particles[5];
    for (int i = 0; i < 5; i++)
    {
        particles[i].x = 0;
        particles[i].y = 0;
        particles[i].z = 0;
        particles[i].vx = 0;
        particles[i].vy = 0;
        particles[i].vz = 0;
    }

    dataptr->copyFromHostToDevice(particles);
}

cudaError_t Summator::propagateParticles(Particle *particles, unsigned int size)
{
    propagateParticlesKernel<<<1, size>>>(dataptr->dev_parts);

    // Check for any errors launching the kernel
    dataptr->cudaStatus = cudaGetLastError();
    if (dataptr->cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(dataptr->cudaStatus));
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
    dataptr->cudaStatus = cudaDeviceSynchronize();
    if (dataptr->cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", dataptr->cudaStatus);
    }

    // Copy output vector from GPU buffer to host memory.
    dataptr->copyFromDeviceToHost(particles);

    return dataptr->cudaStatus;
}

cudaError_t Summator::addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    // Copy input vectors from host memory to GPU buffers.
    dataptr->copyFromHostToDevice(a, dataptr->dev_a);
    dataptr->copyFromHostToDevice(b, dataptr->dev_b);

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dataptr->dev_c, dataptr->dev_a, dataptr->dev_b);

    // Check for any errors launching the kernel
    dataptr->cudaStatus = cudaGetLastError();
    if (dataptr->cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(dataptr->cudaStatus));
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
    dataptr->cudaStatus = cudaDeviceSynchronize();
    if (dataptr->cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", dataptr->cudaStatus);
    }

    // Copy output vector from GPU buffer to host memory.
    dataptr->copyFromDeviceToHost(dataptr->dev_c, c);

    return dataptr->cudaStatus;
}