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

    // n is given as command line argument
    unsigned int n = atoi(argv[1]);
    

    std::random_device entropy_source;

	// 64-bit variant of mt19937 isn't necessary here, but it's just an example
	std::mt19937_64 generator(entropy_source()); 

    std::uniform_real_distribution<float> dist1(-10.0, 10.0);
	std::uniform_real_distribution<float> dist2(0.0, 1.0);

    // Get the starting timestamp
    start = high_resolution_clock::now();
    vscale<<<(N + M-1) / M,M>>>(d_a, d_b, d_c, N);
    end = high_resolution_clock::now();
    // Convert the calculated duration to a double using the standard library
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    // Durations are converted to milliseconds already thanks to std::chrono::duration_cast
    cout << duration_sec.count() << endl;
    cout << output[0] << endl;
    cout << output[n*n - 1] << endl;
}