#include <iostream>
#include <cuda.h>
#include "vscale.cuh"

#include <random>  // To generate random numbers

// The std::chrono namespace provides timer functions in C++
#include <chrono>

// std::ratio provides easy conversions between metric units
#include <ratio>
using std::chrono::duration;
using std::chrono::high_resolution_clock;

int main(int argc, char* argv[])
{
    // declarations needed for timer
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    std::random_device entropy_source;

	// 64-bit variant of mt19937 isn't necessary here, but it's just an example
	std::mt19937_64 generator(entropy_source()); 

    std::uniform_real_distribution<float> dist1(-10.0, 10.0);
	std::uniform_real_distribution<float> dist2(0.0, 1.0);

    
    // n is given as command line argument
    unsigned int n = atoi(argv[1]);
    

    // array to store n*n random float values from -10.0 to 10.0 for image
    float *a = new float[n];
    if (!a)
    {
        cout << "Memory allocation failed for array a\n";
    }

    // array to store the n*n random float values from 0.0 to 1.0 for mask
    float *b = new float[n];
    if (!b)
    {
        cout << "Memory allocation failed for array b\n";
    }

    // Write a random value to each slot in N
	for (size_t i = 0; i < n; i++) {
		a[i] = dist1(generator);
	}

    for (size_t i = 0; i < n; i++) {
		b[i] = dist2(generator);
	}

    float *d_a, *d_b;
    int size = n * sizeof(float);

    //allocate memory on the device (GPU)   
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // threads per block
    int M=512;
    
    
    // Get the starting timestamp
    start = high_resolution_clock::now();
    vscale<<<(n + M-1) / M,M>>>(a, b, n);
    end = high_resolution_clock::now();
    // Convert the calculated duration to a double using the standard library
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);

    // Durations are converted to milliseconds already thanks to std::chrono::duration_cast
    cout << duration_sec.count() << endl;
    cout << b[0] << endl;
    cout << b[n - 1] << endl;

    // deallocate memory
    delete[] a;
    delete[] b;
}