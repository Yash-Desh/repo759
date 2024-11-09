#include <iostream>
#include <cuda.h>
#include "vscale.cuh"

#include <random> // To generate random numbers

int main(int argc, char *argv[])
{
    std::random_device entropy_source;
    std::mt19937_64 generator(entropy_source());
    std::uniform_real_distribution<float> dist1(-10.0, 10.0);
    std::uniform_real_distribution<float> dist2(0.0, 1.0);

    // n is given as command line argument
    unsigned int n = std::atoi(argv[1]);

    // array to store n*n random float values from -10.0 to 10.0 for image
    float *a = new float[n];
    if (!a)
    {
        std::cout << "Memory allocation failed for array a\n";
    }

    // array to store the n*n random float values from 0.0 to 1.0 for mask
    float *b = new float[n];
    if (!b)
    {
        std::cout << "Memory allocation failed for array b\n";
    }

    // Write a random value to each slot in N
    for (size_t i = 0; i < n; i++)
    {
        a[i] = dist1(generator);
    }

    for (size_t i = 0; i < n; i++)
    {
        b[i] = dist2(generator);
    }

    float *d_a, *d_b;
    size_t size = n * sizeof(float);

    // allocate memory on the device (GPU)
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // threads per block
    int M = 512;

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int number_of_block = (n + M - 1) / M;
    cudaEventRecord(start);
    vscale<<<number_of_block, M>>>(a, b, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);

     // Get the elapsed time in milliseconds
     float ms;
     cudaEventElapsedTime(&ms, start, stop);

    // Durations are converted to milliseconds already thanks to std::chrono::duration_cast
    std::cout << ms << std::endl;
    std::cout << b[0] << std::endl;
    std::cout << b[n - 1] << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // deallocate memory
    cudaFree(d_a);
    cudaFree(d_b);
    delete[] a;
    delete[] b;
}
