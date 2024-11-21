#include "stencil.cuh"
#include <cuda.h>
#include <cmath>

__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R) {
    // // ShMem allocation determined at run time
    extern __shared__ float shared_memory[];

    // Pointers within shared memory
    float* shared_image = shared_memory;                     
    float* shared_mask = shared_memory + blockDim.x + 2 * R; 
    float* shared_output = shared_memory + blockDim.x + 4 * R + 1;

    // Global thread ID and local thread ID
    unsigned int local_thread_id = threadIdx.x;
    unsigned int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;


    // ##############################################################################################

    // Load into shared memory the mask
    // if (local_thread_id < 2 * R + 1) {
    //     shared_mask[local_thread_id] = mask[local_thread_id];
    // }

    // // put image shared memory
    // int shared_start_idx = blockIdx.x * blockDim.x - R;
    // int shared_end_idx = shared_start_idx + blockDim.x + 2 * R;

    // // Assume that image[i] = 1 when i < 0 or i > n − 1
    // if (shared_start_idx + local_thread_id >= 0 && shared_start_idx + local_thread_id < n) 
    // {
    //     shared_image[local_thread_id] = image[shared_start_idx + local_thread_id];
    // } 
    // else 
    // {
    //     shared_image[local_thread_id] = 1.0f; 
    // }

    if(local_thread_id < R)
    {
	    int left_idx = global_thread_id - R;
	      shared_image[threadIdx.x] = (left_idx >= 0 )?(image[left_idx]):1.0;
	    // printf("Left halo loaded with value: %f\n", image_s[threadIdx.x]);
    }
    if(threadIdx.x < R)
    {
	    int right_idx = global_thread_id + blockDim.x;
	    shared_image[blockDim.x + R + threadIdx.x] = (right_idx < n) ? (image[right_idx]):1.0;
	    // printf("Right halo loaded with value: %f\n",image_s[blockDim.x + R + threadIdx.x]);
    }
    if(global_thread_id < n)
      shared_image[R+threadIdx.x] = image[global_thread_id];
    if(local_thread_id <= 2*R)
      shared_image[threadIdx.x] = mask[threadIdx.x];

    // ##############################################################################################



    // make sure all threads have completed loading into shared memory
    __syncthreads();

    // convolution within valid range
    if (global_thread_id < n) 
    {
        float outcome = 0.0f;
        for (int i = -(int)R; i <= (int)R; ++i) 
        {
            outcome += shared_image[local_thread_id + R + i] * shared_mask[i+R];
        }
        shared_output[local_thread_id] = outcome;
        // output[global_thread_id] = shared_output[local_thread_id];
    }
    if(global_thread_id < n) 
      output[global_thread_id] = shared_output[threadIdx.x];
}



__host__ void stencil(const float* image,
    const float* mask,
    float* output,
    unsigned int n,
    unsigned int R,
    unsigned int threads_per_block) 
{
    
 

    // number of required blocks
    unsigned int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    
    // Shared memory includes:
    // - threads_per_block + 2 * R elements for shared_image
    // - 2 * R + 1 elements for shared_mask
    size_t shared_memory_size = (2*R + threads_per_block + (2*R+1) + threads_per_block)*sizeof(float);

    
    stencil_kernel<<<blocks_per_grid, threads_per_block, shared_memory_size>>>(image, mask, output, n, R);

    
    cudaDeviceSynchronize();
}

