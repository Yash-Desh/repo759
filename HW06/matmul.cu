#include "matmul.cuh"
#include <cuda.h>
#include <cmath>

__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n)
{
    int index_x = threadIdx.x + blockIdx.x * blockDim.x;
    int index_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (index_x < n && index_y < n)
    {
        float temp = 0;
        for (unsigned int j = 0; j < n; j++)
        {
            temp += A[index_x * n + j] * B[j * n + index_y];
        }
        C[index_x * n + index_y] = temp;
    }
}
void matmul(const float* A, const float* B, float *C, size_t n, unsigned int threads_per_block)
{

    dim3 dimGrid((n + sqrt(threads_per_block) - 1) / sqrt(threads_per_block), (n + sqrt(threads_per_block) - 1) / sqrt(threads_per_block));
    dim3 dimBlock(sqrt(threads_per_block), sqrt(threads_per_block));

    matmul_kernel<<<dimGrid, dimBlock>>>(A, B, C, n);
    cudaDeviceSynchronize();
}
