#include <iostream>
#include <random>  // To generate random numbers
#include "convolution.h"

// The std::chrono namespace provides timer functions in C++
#include <chrono>

// std::ratio provides easy conversions between metric units
#include <ratio>
using std::chrono::duration;
using std::chrono::high_resolution_clock;
using namespace std;

int main(int argc, char* argv[])
{
    // declarations needed for timer
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    // n & t are given as command line arguments
    size_t n = atoi(argv[1]);

    // t is the number of threads
    size_t t = atoi(argv[2]);

    // size of the mask is fixed at 3
    size_t m = 3;

    // array to store n*n random float values from -10.0 to 10.0 for image
    float *image_arr = new float[n*n];
    if (!image_arr)
    {
        cout << "Memory allocation failed for array of random floats\n";
    }

    // array to store the m*m random float values from -1.0 to 1.0 for mask
    float *mask_arr = new float[m*m];
    if (!mask_arr)
    {
        cout << "Memory allocation failed for scan array of random floats\n";
    }

    // array to store n*n float values for output
    float *output = new float[n*n];

    std::random_device entropy_source;

	// 64-bit variant of mt19937 isn't necessary here, but it's just an example
	std::mt19937_64 generator(entropy_source()); 

    std::uniform_real_distribution<float> dist1(-10.0, 10.0);
	std::uniform_real_distribution<float> dist2(-1.0, 1.0);

	// Write a random value to each slot in N
	for (size_t i = 0; i < n*n; i++) {
		image_arr[i] = dist1(generator);
	}

    for (size_t i = 0; i < m*m; i++) {
		mask_arr[i] = dist2(generator);
	}

    // Get the starting timestamp
    start = high_resolution_clock::now();
    omp_set_num_threads(t);
    #pragma omp parallel
    {
        convolve(image_arr, output, n, mask_arr, m);
    }
    end = high_resolution_clock::now();
    // Convert the calculated duration to a double using the standard library
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    // Durations are converted to milliseconds already thanks to std::chrono::duration_cast
    
    cout << output[0] << endl;
    cout << output[n*n - 1] << endl;
    cout << duration_sec.count() << endl;


    // for(size_t i = 0; i < n*n; i++)
    //     cout<<image_arr[i]<<" ";
    // cout<<endl;

    // for(size_t i = 0; i < m*m; i++)
    //     cout<<mask_arr[i]<<" ";
    // cout<<endl;

    // for(size_t i = 0; i < n*n; i++)
    //     cout<<output[i]<<" ";
    // cout<<endl;

    // deallocate memory
    delete[] image_arr;
    delete[] mask_arr;
    delete[] output;

    return 0;
}