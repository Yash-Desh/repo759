#include <cuda.h>
#include "matmul.cuh"

template <typename T>
__global__ void matmul_kernel(const T* A, const T* B, T* C, unsigned int n, unsigned int block_dim)
{
    // // Block index
    // int bx = blockIdx.x; 
    // int by = blockIdx.y; 
    
    // // Thread index
    // int tx = threadIdx.x; 
    // int ty = threadIdx.y;


    // // ----- indexes to select a tile within the nxn sized arrays of A & B -----

    // // Index of the first sub-matrix of A processed by the block
    // int aBegin = n * block_dim * by;
    
    // // Index of the last sub-matrix of A processed by the block
    // int aEnd = aBegin + n - 1;
    
    // // Step size used to iterate through the sub-matrices of A
    // int aStep = block_dim;
    
    // // Index of the first sub-matrix of B processed by the block
    // int bBegin = block_dim * bx;
    
    // // Step size used to iterate through the sub-matrices of B
    // int bStep = block_dim * n;
    
    // The element of the block sub-matrix that is computed
    // by the thread
    T Csub = 0;

    // ------------------------------------------------------

    // Shared memory for the sub-matrices (tiles) of A and B
    __shared__ T As[block_dim * block_dim];
    __shared__ T Bs[block_dim * block_dim];

    int x = (n+block_dim-1)/block_dim;

    for(int i=0; i<x; i++)
    {
        int tile_index_A = blockIdx.y + i*block_dim;
        int tile_index_B = blockIdx.x + i*block_dim;

        if(block_dim*blockIdx.y+threadIdx.y >= n || block_dim*blockIdx.x+threadIdx.x)
            continue;

        As[ty*block_dim * tx] = A[tile_index_A + ty];
        Bs[ty*block_dim * tx] = B[tile_index_B + tx];


        for (int k = 0; k < block_dim; k++)
        {
            Csub += As[ty*block_dim + k] * Bs[k*block_dim + tx];
        }
    }

    C[(block_dim*blockIdx.y+threadIdx.y)*n + (block_dim*blockIdx.x+threadIdx)] = Csub;
}

void matmul_1(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim)
{
    dim3 dimGrid((n+block_dim-1)/block_dim, (n+block_dim-1)/block_dim);
    dim3 dimBlock(block_dim, block_dim);

    matmul_kernel<int><<<dimGrid, dimBlock>>>(A, B, C, n);
    cudaDeviceSynchronize();
}

void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim)
{
    dim3 dimGrid((n+block_dim-1)/block_dim, (n+block_dim-1)/block_dim);
    dim3 dimBlock(block_dim, block_dim);

    matmul_kernel<float><<<dimGrid, dimBlock>>>(A, B, C, n);
    cudaDeviceSynchronize();
}

void matmul_3(const double *A, const double *B, double *C, unsigned int n, unsigned int block_dim)
{
    dim3 dimGrid((n+block_dim-1)/block_dim, (n+block_dim-1)/block_dim);
    dim3 dimBlock(block_dim, block_dim);

    matmul_kernel<double><<<dimGrid, dimBlock>>>(A, B, C, n);
    cudaDeviceSynchronize();
}
