#include <cuda.h>
#include <cstdio>
#include <random>
#include "vscale.cuh"
#include <iostream>

int main(int argc, char *argv[])
{
    const int NUM_THREADS_PER_BLOCK = 512;
    cudaEvent_t start;
    cudaEvent_t stop;
    float ms;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int N = std::stoi(argv[1]);
    std::random_device entropy_source;
    std::mt19937_64 generator(entropy_source()); 
    std::uniform_real_distribution<float> dist1(-10.0,10.0);
    std::uniform_real_distribution<float> dist2(0.0,1.0);
    
    float*a = (float*)malloc(N*(sizeof(float)));
    float*b = (float*)malloc(N*(sizeof(float)));
    float*d_a,*d_b;


    for(int i = 0; i < N; i++)
    {
        a[i] = dist1(generator);
        b[i] = dist2(generator);
    }

    cudaMalloc((void**)&d_a,sizeof(float) * N);
    cudaMalloc((void**)&d_b,sizeof(float) * N);
    
    cudaMemcpy(d_a,a,sizeof(float)*N,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,b,sizeof(float)*N,cudaMemcpyHostToDevice);
    
    cudaEventRecord(start);
    vscale<<<(N+NUM_THREADS_PER_BLOCK-1)/NUM_THREADS_PER_BLOCK,NUM_THREADS_PER_BLOCK>>>(d_a,d_b,N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    cudaMemcpy(b,d_b,sizeof(float)*N,cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&ms, start, stop);
  
    std::cout<<N;
    std::cout<<std::endl; 
    std::cout<<ms;
    std::cout << std::endl;
    std::cout<<b[0];
    std::cout << std::endl;
    std::cout<<b[N-1];
    std::cout << std::endl;
    std::cout << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    free(a);
    free(b);
    
}