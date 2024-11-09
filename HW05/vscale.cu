#include "vscale.cuh"
#include <cuda.h>

__global__ void vscale(const float *a, float *b, unsigned int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
        b[index] = a[index] * b[index];
}