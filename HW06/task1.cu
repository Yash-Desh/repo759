#include <iostream>
#include <cuda.h>
#include "matmul.cuh"

#include <random>           // To generate random numbers
using namespace std;


int main(int argc, char *argv[])
{
    // command line arguments
    unsigned int n = std::stoi(argv[1]);                        // n-dimension
    unsigned int threads_per_block = std::stoi(argv[2]);        // threads per block
    
    // declarations for calculating time
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;                                                   // time in ms

    // random number generation
    std::random_device entropy_source;
    std::mt19937_64 generator(entropy_source());
    std::uniform_real_distribution<float> dist1(-1.0, 1.0);
    std::uniform_real_distribution<float> dist2(-1.0, 1.0);


    // host arrays 
    float *A= (float*)malloc(n*n*(sizeof(float)));
    float *B= (float*)malloc(n*n*(sizeof(float)));
    float *C= (float*)malloc(n*n*(sizeof(float)));

    for (size_t i = 0; i < n; i++)
    {
        A[i] = dist1(generator);
        B[i] = dist2(generator);
    }

    // device arrays
    float *d_A, *d_B, *d_C;

    // allocate memory on the device (GPU)
    cudaMalloc((void **)&d_A, n * n *sizeof(float));
    cudaMalloc((void **)&d_B, n * n* sizeof(float));
    cudaMalloc((void **)&d_C, n * n* sizeof(float));

    // copy data into device blocks
    cudaMemcpy(d_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * n * sizeof(float), cudaMemcpyHostToDevice);


    // #############################################
    cudaEventRecord(start);
    matmul(d_A, d_B, d_C, n, threads_per_block);
    cudaEventRecord(stop);
    // #############################################
    
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Durations are converted to milliseconds already thanks to std::chrono::duration_cast
    std::cout << C[n*n - 1] << std::endl;
    std::cout << ms << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // deallocate memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);


    return 0;
    
}