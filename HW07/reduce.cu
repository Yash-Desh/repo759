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


// #include "reduce.cuh"

// __global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int N) {
//     extern __shared__ float sdata[];

//     unsigned int tid = threadIdx.x;
//     unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

//     // First add during load
//     sdata[tid] = (idx < N ? g_idata[idx] : 0.0f) + 
//                  ((idx + blockDim.x) < N ? g_idata[idx + blockDim.x] : 0.0f);
//     __syncthreads();

//     // Perform reduction in shared memory
//     for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
//         if (tid < s) {
//             sdata[tid] += sdata[tid + s];
//         }
//         __syncthreads();
//     }

//     // Write the block's result to global memory
//     if (tid == 0) {
//         g_odata[blockIdx.x] = sdata[0];
//     }
// }

// void reduce(float **input, float **output, unsigned int N, unsigned int threads_per_block) {
//     unsigned int num_blocks;
//     float *temp;

//     while (N > 1) {
//         num_blocks = (N + threads_per_block - 1) / threads_per_block;

//         reduce_kernel<<<num_blocks, threads_per_block, threads_per_block * sizeof(float)>>>(*input, *output, N);
//         cudaDeviceSynchronize();

//         // Swap input and output pointers
//         temp = *input;
//         *input = *output;
//         *output = temp;

//         N = num_blocks; // New size is the number of blocks
//     }
// }
