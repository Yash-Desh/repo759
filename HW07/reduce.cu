#include <cuda.h>
#include "reduce.cuh"

__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n)
{
    // perform first level of reduction upon reading from
    // global memory and writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    extern __shared__ float sdata[];

    // if(blockDim.x * blockIdx.x + threadIdx.x < n)
    // {
    //     sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
    //     __syncthreads();

    //     for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
    //     {
    //         if (tid < s) 
    //         {
    //         sdata[tid] += sdata[tid + s];
    //         }
    //         __syncthreads();
    //     }
    //     g_odata[blockIdx.x] = sdata[0];
    // }
    if(i+blockDim.x < n)
   	 sdata[threadIdx.x] = g_idata[i] + g_idata[i+blockDim.x];
    else if(i < n)
	 sdata[threadIdx.x] = g_idata[i];
    else
	 sdata[threadIdx.x] = 0;
    __syncthreads();
    for(unsigned int s = blockDim.x/2; s > 0; s>>=1)
    {
	
        if(threadIdx.x < s)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0) g_odata[blockIdx.x] = sdata[0];

    
}

__host__ void reduce(float **input, float **output, unsigned int N, unsigned int threads_per_block)
{
    // N = length of input array
    // input of the next kernel call is the 

    while(N!=1)
    {
        int num_blocks = (N+threads_per_block-1)/threads_per_block;
        reduce_kernel<<<num_blocks, threads_per_block, threads_per_block*sizeof(float)>>>(*input, *output, N);
        cudaDeviceSynchronize();
        *input = *output;
        N = (N+threads_per_block-1)/threads_per_block;
    }
    *input = *output;
    
}
