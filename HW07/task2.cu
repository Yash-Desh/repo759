#include <cuda.h>
#include "reduce.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <random> // To generate random numbers

// using namespace std;

int main(int argc, char *argv[]) {
    int N = std::stoi(argv[1]); // Array dimension
    unsigned int threads_per_block = std::stoi(argv[2]); // Block dimension

    // Time calculation variables
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0;

    // Random number generation
    std::random_device entropy_source;
    std::mt19937_64 generator(entropy_source());
    std::uniform_real_distribution<float> dist(-1.0, 1.0);

    int num_blocks = (N + threads_per_block - 1) / threads_per_block;

    float *h_input = (float *)malloc(N * sizeof(float));
    // float *h_output = (float *)malloc(num_blocks * sizeof(float));

    for (int i = 0; i < N; i++) {
       //  h_input[i] = dist(generator);
	    h_input[i] = 1;
    }

    float *d_input;
    float *d_output;
    cudaMalloc((void **)&d_input, N * sizeof(float));
    cudaMalloc((void **)&d_output, num_blocks * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // #############################################
    cudaEventRecord(start);
    reduce(&d_input, &d_output, N, threads_per_block);
    cudaEventRecord(stop);
    // #############################################

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_input, d_input, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "Reduction result: " << h_input[0] << std::endl;
    std::cout << "Elapsed time (ms): " << ms << std::endl;

    // Cleanup
    free(h_input);
    // free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

