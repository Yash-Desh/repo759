#include <cuda.h>
#include "reduce.cuh"

__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n)
{
    // perform first level of reduction upon reading from
    // global memory and writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    extern __shared__ float sdata[];

    if(blockDim.x * blockIdx.x + threadIdx.x < N)
    {
        sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
        __syncthreads();

        for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
        {
            if (tid < s) 
            {
            sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        g_odata[blockIdx.x] = sdata[0];
    }


    
}

__host__ void reduce(float **input, float **output, unsigned int N, unsigned int threads_per_block)
{
    // N = length of input array
    // input of the next kernel call is the 

    while(N!=1)
    {
        num_blocks = (N+threads_per_block-1)/threads_per_block;
        reduce_kernel<<<num_blocks, threads_per_block, threads_per_block*sizeof(float)>>>(*input, *output, N);
        *input = *output;
        N = (N+threads_per_block-1)/threads_per_block;
    }
    
    cudaDeviceSynchronize();
}
