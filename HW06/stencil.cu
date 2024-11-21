#include "stencil.cuh"
#include <cuda.h>
#include <cmath>
#include <cuda_runtime.h>

__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R) {
    // // ShMem allocation determined at run time
    extern __shared__ float shared_memory[];

    // Pointers within shared memory
    float* shared_image = shared_memory;                     
    float* shared_mask = &shared_memory[blockDim.x+2*R]; 
    float* shared_output = &shared_memory[blockDim.x + 4*R + 1];

    // Global thread ID and local thread ID
    int local_thread_id = threadIdx.x;
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;


    // ##############################################################################################

    if(local_thread_id < R)
    {
	    int left_halo_index = global_thread_id - R;
	      shared_image[threadIdx.x] = (left_halo_index >= 0 )?(image[left_halo_index]):1.0;
	    // printf("Left halo loaded with value: %f\n", image_s[threadIdx.x]);
    }
    if(threadIdx.x < R)
    {
	    int right_halo_index = global_thread_id + blockDim.x;
	    shared_image[blockDim.x + R + threadIdx.x] = (right_halo_index < n) ? (image[right_halo_index]):1.0;
	    // printf("Right halo loaded with value: %f\n",image_s[blockDim.x + R + threadIdx.x]);
    }
    if(global_thread_id < n)
      shared_image[R+threadIdx.x] = image[global_thread_id];
    if(local_thread_id <= 2*R)
      shared_mask[threadIdx.x] = mask[threadIdx.x];

    // ##############################################################################################



    // make sure all threads have completed loading into shared memory
    __syncthreads();

    // convolution within valid range
    if (global_thread_id < n) 
    {
        float outcome = 0.0;
        for (int i = -(int)R; i <= (int)R; i++) 
        {
            outcome += shared_image[local_thread_id + R + i] * shared_mask[i+R];
        }
        shared_output[local_thread_id] = outcome;
        // output[global_thread_id] = shared_output[local_thread_id];
    }
    __syncthreads();
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
    // unsigned int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    
    // Shared memory includes:
    // - threads_per_block + 2 * R elements for shared_image
    // - 2 * R + 1 elements for shared_mask
    // size_t shared_memory_size = (2*R + threads_per_block + (2*R+1) + threads_per_block)*sizeof(float);

    
    stencil_kernel<<<(n + threads_per_block - 1) / threads_per_block, threads_per_block, (2*R + threads_per_block + (2*R+1) + threads_per_block)*sizeof(float)>>>(image, mask, output, n, R);

    
   // cudaDeviceSynchronize();
}

