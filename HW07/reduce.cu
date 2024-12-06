#include "reduce.cuh"

__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int N) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // First add during load
    sdata[tid] = (idx < N ? g_idata[idx] : 0.0f) + 
                 ((idx + blockDim.x) < N ? g_idata[idx + blockDim.x] : 0.0f);
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the block's result to global memory
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

void reduce(float **input, float **output, unsigned int N, unsigned int threads_per_block) {
    unsigned int num_blocks;
    float *temp;

    while (N > 1) {
        num_blocks = (N + threads_per_block - 1) / threads_per_block;

        reduce_kernel<<<num_blocks, threads_per_block, threads_per_block * sizeof(float)>>>(*input, *output, N);
        cudaDeviceSynchronize();

        // Swap input and output pointers
        temp = *input;
        *input = *output;
        *output = temp;

        N = num_blocks; // New size is the number of blocks
    }
}
