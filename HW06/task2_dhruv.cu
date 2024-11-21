#include <iostream>
#include <cuda.h>
#include <random>
#include "stencil.cuh"

int main(int argc, char*argv[])
{
    cudaEvent_t start;
    cudaEvent_t stop;
    float ms;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int n = std::atoi(argv[1]);
    int R = std::atoi(argv[2]);
    int NUM_THREADS_PER_BLOCK = std::atoi(argv[3]);
    std::random_device entropy_source;
    std::mt19937_64 generator(entropy_source());
    std::uniform_real_distribution<float> dist(-1.0,1.0);

    float*image = (float*)malloc(n*(sizeof(float)));
    float*mask = (float*)malloc((2*R+1)*(sizeof(float)));
    float*output = (float*)malloc(n*(sizeof(float)));
    float*d_image,*d_mask,*d_output;
    for(int i = 0; i < n; i++)
    {
        image[i] = dist(generator);
    }
    for(int i = 0; i < 2*R+1; i++)
    {
        mask[i] = dist(generator);
    }
   /* std::cout <<"Image: ";
    for(int i = 0; i < n; i++)
    {
        std::cout<<image[i]<<" ";
    }
    std::cout << std::endl;
    std::cout << "Mask: ";
    for(int i = 0; i < 2*R+1; i++)
    {
        std::cout<<mask[i]<<" ";
    }
    std::cout << std::endl;*/

    cudaMalloc((void**)&d_image,sizeof(float)*n);
    cudaMalloc((void**)&d_output,sizeof(float)*n);
    cudaMalloc((void**)&d_mask,sizeof(float)*(2*R+1));

    cudaMemcpy(d_image,image,sizeof(float)*n,cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask,mask,sizeof(float)*(2*R+1),cudaMemcpyHostToDevice);
    cudaMemcpy(d_output,output,sizeof(float)*n,cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    stencil(d_image,d_mask,d_output,n,R,NUM_THREADS_PER_BLOCK);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaMemcpy(output,d_output,sizeof(float)*n,cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&ms, start, stop);

   /* std::cout<<"Output ";
    for(int i = 0; i < n; i++)
    {
        std::cout<<output[i]<<" ";
    }*/
    std::cout<<output[n-1];
    std::cout<<std::endl; 
    std::cout<<"Time ELapsed "<<ms;
    std::cout<<std::endl;

    cudaFree(d_image);
    cudaFree(d_output);
    cudaFree(d_mask);
    free(image);
    free(output);
    free(mask);

}
