#include <cuda.h>
#include "reduce.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <random>           // To generate random numbers

using namespace std;

int main(int argc, char *argv[])
{
    // command line arguments
    unsigned int N = std::stoi(argv[1]);                // arrray dimension
    unsigned int threads_per_block = std::stoi(argv[2]);        // block dimension
    
    // declarations for calculating time
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms=0;  

    // random number generation
    std::random_device entropy_source;
    std::mt19937_64 generator(entropy_source());
    std::uniform_real_distribution<float> dist(-1.0, 1.0);

    int num_blocks = (N+threads_per_block-1)/threads_per_block;

    float *h_input = (int *)malloc(N*sizeof(float));
    float *h_output = (int *)malloc(num_blocks*sizeof(float));

    for(int i=0; i<N; i++)
    {
        h_input[i] = dist(generator);
    }

    float *d_input; 
    float *d_output
    cudaMalloc((void **)&d_input, N *sizeof(float));
    cudaMalloc((void **)&d_output, num_blocks *sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // #############################################
    cudaEventRecord(start);
    reduce(d_input, float **output, N, threads_per_block);
    cudaEventRecord(stop);
    // #############################################
    
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_input, d_input, N * sizeof(float), cudaMemcpyDeviceToHost);

    cout<<"hello world"<<endl;
    std::cout << h_input[0] << std::endl;
    std::cout << ms << std::endl;

    return 0;
}