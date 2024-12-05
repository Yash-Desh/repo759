#include <cuda.h>
#include "matmul.cuh"


__global__ void matmul_kernel_1(const int* A, const int * B, int * C, unsigned int n, unsigned int block_dim)
{
    // // Block index
    // int bx = blockIdx.x; 
    // int by = blockIdx.y; 
    
    // // Thread index
    int tx = threadIdx.x; 
    int ty = threadIdx.y;


    
    int  Csub = 0;

    // ------------------------------------------------------

    // Shared memory for the sub-matrices (tiles) of A and B
    extern __shared__ int shared_tile_1[];
    int* As = shared_tile_1;
    int* Bs = &shared_tile_1[block_dim*block_dim];

    int x = (n+block_dim-1)/block_dim;

    for(int i=0; i<x; i++)
    {
        int tile_index_A = blockIdx.y + i*block_dim;
        int tile_index_B = blockIdx.x + i*block_dim;

        if((block_dim*blockIdx.y+threadIdx.y) >= n || (block_dim*blockIdx.x+threadIdx.x) >= n)
            continue;

        As[ty*block_dim + tx] = A[tile_index_A + ty];
        Bs[ty*block_dim + tx] = B[tile_index_B + tx];

	__syncthreads();
        for (int k = 0; k < block_dim; k++)
        {
            Csub += As[ty*block_dim + k] * Bs[k*block_dim + tx];
        }
	__syncthreads();
    }

    C[(block_dim*blockIdx.y+threadIdx.y)*n + (block_dim*blockIdx.x+threadIdx.x)] = Csub;
}

void matmul_1(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim)
{
    dim3 dimGrid((n+block_dim-1)/block_dim, (n+block_dim-1)/block_dim);
    dim3 dimBlock(block_dim, block_dim);

    matmul_kernel_1<<<dimGrid, dimBlock, 2*block_dim*block_dim*sizeof(int)>>>(A, B, C, n, block_dim);
    cudaDeviceSynchronize();
}



__global__ void matmul_kernel_2(const float* A, const float * B, float* C, unsigned int n, unsigned int block_dim)
{
    // // Block index
    // int bx = blockIdx.x; 
    // int by = blockIdx.y; 
    
    // // Thread index
    int tx = threadIdx.x; 
    int ty = threadIdx.y;


    
    float  Csub = 0;

    // ------------------------------------------------------

    // Shared memory for the sub-matrices (tiles) of A and B
    extern __shared__ float shared_tile_2[];
    float* As = shared_tile_2;
    float* Bs = &shared_tile_2[block_dim*block_dim];

    int x = (n+block_dim-1)/block_dim;

    for(int i=0; i<x; i++)
    {
        int tile_index_A = blockIdx.y + i*block_dim;
        int tile_index_B = blockIdx.x + i*block_dim;

        if((block_dim*blockIdx.y+threadIdx.y) >= n || (block_dim*blockIdx.x+threadIdx.x) >= n)
            continue;

        As[ty*block_dim + tx] = A[tile_index_A + ty];
        Bs[ty*block_dim + tx] = B[tile_index_B + tx];

	__syncthreads();
        for (int k = 0; k < block_dim; k++)
        {
            Csub += As[ty*block_dim + k] * Bs[k*block_dim + tx];
        }
	__syncthreads();
    }

    C[(block_dim*blockIdx.y+threadIdx.y)*n + (block_dim*blockIdx.x+threadIdx.x)] = Csub;
}



void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim)
{
    dim3 dimGrid((n+block_dim-1)/block_dim, (n+block_dim-1)/block_dim);
    dim3 dimBlock(block_dim, block_dim);

    matmul_kernel_2<<<dimGrid, dimBlock, 2*block_dim*block_dim*sizeof(float)>>>(A, B, C, n, block_dim);
    cudaDeviceSynchronize();
}


__global__ void matmul_kernel_3(const double* A, const double * B, double* C, unsigned int n, unsigned int block_dim)
{
    // // Block index
    // int bx = blockIdx.x; 
    // int by = blockIdx.y; 
    
    // // Thread index
    int tx = threadIdx.x; 
    int ty = threadIdx.y;


    
    double  Csub = 0;

    // ------------------------------------------------------

    // Shared memory for the sub-matrices (tiles) of A and B
    extern __shared__ double shared_tile_3[];
    double* As = shared_tile_3;
    double* Bs = &shared_tile_3[block_dim*block_dim];

    int x = (n+block_dim-1)/block_dim;

    for(int i=0; i<x; i++)
    {
        int tile_index_A = blockIdx.y + i*block_dim;
        int tile_index_B = blockIdx.x + i*block_dim;

        if((block_dim*blockIdx.y+threadIdx.y) >= n || (block_dim*blockIdx.x+threadIdx.x) >= n)
            continue;

        As[ty*block_dim + tx] = A[tile_index_A + ty];
        Bs[ty*block_dim + tx] = B[tile_index_B + tx];

	__syncthreads();
        for (int k = 0; k < block_dim; k++)
        {
            Csub += As[ty*block_dim + k] * Bs[k*block_dim + tx];
        }
	__syncthreads();
    }

    C[(block_dim*blockIdx.y+threadIdx.y)*n + (block_dim*blockIdx.x+threadIdx.x)] = Csub;
}



void matmul_3(const double *A, const double *B, double *C, unsigned int n, unsigned int block_dim)
{
    dim3 dimGrid((n+block_dim-1)/block_dim, (n+block_dim-1)/block_dim);
    dim3 dimBlock(block_dim, block_dim);

    matmul_kernel_3<<<dimGrid, dimBlock, 2*block_dim*block_dim*sizeof(double)>>>(A, B, C, n, block_dim);
    cudaDeviceSynchronize();
}
