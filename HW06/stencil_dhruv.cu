#include "stencil.cuh"
#include <cstdio>
#include <cmath>
#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>
__host__ void stencil(const float* image,const float* mask, float* output, unsigned int n, unsigned int R, unsigned int threads_per_block)
    {
       stencil_kernel<<<(n+threads_per_block-1)/threads_per_block,threads_per_block,(2*R + threads_per_block + (2*R+1) + threads_per_block)*sizeof(float)>>>(image,mask,output,n,R);
       cudaError_t err = cudaGetLastError();
       if(err!=cudaSuccess){
	       std::cerr<<"Kernel failed "<<cudaGetErrorString(err);
       }
    }

__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R)
{
     // printf("R = %d\n",R);
      extern __shared__ float shared_mem[];
      float* image_s = shared_mem;
      float* mask_s = &shared_mem[blockDim.x+2*R];
      float* output_s = &shared_mem[blockDim.x + 4*R + 1];
      int index = blockDim.x*blockIdx.x + threadIdx.x;
      if(threadIdx.x < R)
      {
	      int left_idx = index - R;
	      image_s[threadIdx.x] = (left_idx >= 0 )?(image[left_idx]):1.0;
	     // printf("Left halo loaded with value: %f\n", image_s[threadIdx.x]);
      }
      if(threadIdx.x < R)
      {
	      int right_idx = index + blockDim.x;
	      image_s[blockDim.x + R + threadIdx.x] = (right_idx < n) ? (image[right_idx]):1.0;
	     // printf("Right halo loaded with value: %f\n",image_s[blockDim.x + R + threadIdx.x]);
      }
      if(index < n)
      image_s[R+threadIdx.x] = image[index];
      if(threadIdx.x <= 2*R)
      mask_s[threadIdx.x] = mask[threadIdx.x];
     // printf("Thread got elemnet:%f",image_s[threadIdx.x]);
      __syncthreads();


      if(index < n)
      {
        float sum = 0.0;
        for(int j = -(int)R; j <= (int)R; j++)
        {
        
		//printf("Threadid: %d, R = %d, j = %d",threadIdx.x,R,j);
	    	sum += image_s[R+threadIdx.x+j]*mask_s[j+R];
		//printf("Calculated sum:%f",sum);
        }
	output_s[threadIdx.x] = sum;
    
      }
      __syncthreads();
     if(index < n) 
      output[index] = output_s[threadIdx.x];
}
