#include <iostream>
#include <cuda.h>
#include <random>
#include "reduce.cuh"

int main(int argc, char*argv[])
{
    cudaEvent_t start;
    cudaEvent_t stop;
    float ms;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int n = std::stoi(argv[1]);
    int threads_per_block = std::stoi(argv[2]);
    std::random_device entropy_source;
    std::mt19937_64 generator(entropy_source());
    std::uniform_real_distribution <float> dist1(-1,1);

    float*input = (float*)malloc(n*sizeof(float));
    for(int i = 0; i < n; i++)
    {
        input[i] = dist1(generator);
    }
    float*d_input,*d_output;
    cudaMalloc((void**)&d_input,sizeof(float) * n);
    cudaMemcpy(d_input,input,sizeof(float)*n,cudaMemcpyHostToDevice);

    unsigned int first_block_size = (n + threads_per_block - 1)/threads_per_block;
    cudaMalloc((void**)&d_output,sizeof(float)*first_block_size);

    cudaEventRecord(start);
    reduce(&d_input,&d_output,n,threads_per_block);
    cudaEventRecord(stop);

    cudaMemcpy(input,d_input,sizeof(float),cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    std::cout<<"Sum: "<<input[0];
    std::cout<<std::endl;
    std::cout<<"Time ELapsed "<<ms;
    std::cout<<std::endl;
    std::cout<<std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    free(input);

}
