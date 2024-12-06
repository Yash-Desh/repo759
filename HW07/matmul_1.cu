#include "matmul.cuh"
#include <cstdio>
#include <cmath>
#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>

__global__ void matmul_1_kernel(const int*A, const int*B, int*C, unsigned int n)
{
    extern __shared__ int shared_mem[];
    int* As = shared_mem;
    int* Bs = &shared_mem[blockDim.x*blockDim.x];
    int Csub = 0;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    for(int tile_idx = 0; tile_idx < (int)n; tile_idx+=blockDim.x)
    {
	if(row < n && (tile_idx + threadIdx.x) < n)
        As[threadIdx.y*blockDim.x + threadIdx.x] = A[(row*n + (threadIdx.x+tile_idx))];
	else
	As[threadIdx.y*blockDim.x + threadIdx.x] = 0;

	if(col < n && (tile_idx + threadIdx.y) < n)
        Bs[threadIdx.y*blockDim.x + threadIdx.x] = B[(threadIdx.y+tile_idx)*n + col];
	else
	Bs[threadIdx.y*blockDim.x + threadIdx.x] = 0;

        __syncthreads();

        for(int k = 0; k < blockDim.x; k++)
        {
            Csub += As[threadIdx.y*blockDim.x + k]*Bs[k*blockDim.x + threadIdx.x];
        }

    }

    __syncthreads();
    if(row < n && col < n)
    C[row*n + col] = Csub;
}
__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n,unsigned int block_dim)
{
    dim3 dimGrid((n+block_dim-1)/block_dim,(n+block_dim-1)/block_dim);
    dim3 dimBlock(block_dim,block_dim);
    matmul_1_kernel<<<dimGrid,dimBlock,(2*block_dim*block_dim)*sizeof(int)>>>(A,B,C,n);
}


__global__ void matmul_2_kernel(const float*A, const float*B, float*C, unsigned int n)
{
    extern __shared__ float shared_mem1[];
    float* As = shared_mem1;
    float* Bs = &shared_mem1[blockDim.x*blockDim.x];
    float Csub = 0;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    for(int tile_idx = 0; tile_idx < (int)n; tile_idx+=blockDim.x)
    {
	if(row < n && (tile_idx + threadIdx.x) < n)
        As[threadIdx.y*blockDim.x + threadIdx.x] = A[(row*n + (threadIdx.x+tile_idx))];
	else
	As[threadIdx.y*blockDim.x + threadIdx.x] = 0;

	if(col < n && (tile_idx + threadIdx.y) < n)
        Bs[threadIdx.y*blockDim.x + threadIdx.x] = B[(threadIdx.y+tile_idx)*n + col];
	else
	Bs[threadIdx.y*blockDim.x + threadIdx.x] = 0;

        __syncthreads();

        for(int k = 0; k < blockDim.x; k++)
        {
            Csub += As[threadIdx.y*blockDim.x + k]*Bs[k*blockDim.x + threadIdx.x];
        }

    }

    __syncthreads();
    if(row < n && col < n)
    C[row*n + col] = Csub;
}
__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n,unsigned int block_dim)
{
    dim3 dimGrid((n+block_dim-1)/block_dim,(n+block_dim-1)/block_dim);
    dim3 dimBlock(block_dim,block_dim);
    matmul_2_kernel<<<dimGrid,dimBlock,(2*block_dim*block_dim)*sizeof(float)>>>(A,B,C,n);
}


__global__ void matmul_3_kernel(const double*A, const double*B, double*C, unsigned int n)
{
    extern __shared__ double shared_mem2[];
    double* As = shared_mem2;
    double* Bs = &shared_mem2[blockDim.x*blockDim.x];
    double Csub = 0;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    for(int tile_idx = 0; tile_idx < (int)n; tile_idx+=blockDim.x)
    {
	if(row < n && (tile_idx + threadIdx.x) < n)
        As[threadIdx.y*blockDim.x + threadIdx.x] = A[(row*n + (threadIdx.x+tile_idx))];
	else
	As[threadIdx.y*blockDim.x + threadIdx.x] = 0;

	if(col < n && (tile_idx + threadIdx.y) < n)
        Bs[threadIdx.y*blockDim.x + threadIdx.x] = B[(threadIdx.y+tile_idx)*n + col];
	else
	Bs[threadIdx.y*blockDim.x + threadIdx.x] = 0;

        __syncthreads();

        for(int k = 0; k < blockDim.x; k++)
        {
            Csub += As[threadIdx.y*blockDim.x + k]*Bs[k*blockDim.x + threadIdx.x];
        }

    }

    __syncthreads();
    if(row < n && col < n)
    C[row*n + col] = Csub;
}
__host__ void matmul_3(const double *A, const double *B, double *C, unsigned int n,unsigned int block_dim)
{
    dim3 dimGrid((n+block_dim-1)/block_dim,(n+block_dim-1)/block_dim);
    dim3 dimBlock(block_dim,block_dim);
    matmul_3_kernel<<<dimGrid,dimBlock,(2*block_dim*block_dim)*sizeof(double)>>>(A,B,C,n);
}
