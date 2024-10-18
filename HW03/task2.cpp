#include <iostream>
#include <random>                   // To generate random numbers
#include <cstring>                  // To use memset
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

    // n is dimension of image
    size_t n = atoi(argv[1]);

    // t is the number of threads
    size_t t = atoi(argv[2]);

    // size of the mask is 3
    size_t m = 3;

    // image array of dimension n
    float *image_arr = new float[n*n];
    if (!image_arr)
    {
        cout << "Memory allocation failed for array of random floats\n";
    }

    // array to store extended image
    float *image_mod = new float[(n+2)*(n+2)];
    if (!image_mod)
    {
        cout << "Memory allocation failed for extended image array\n";
    }

    // set all elements in image_mod to 0
    memset(image_mod, 0, (n+2) * (n+2) * sizeof(image_mod[0]));


    // mask array of dimension m
    float *mask_arr = new float[m*m];
    if (!mask_arr)
    {
        cout << "Memory allocation failed for mask array\n";
    }

    // output of the convolution of dimension n
    float *output = new float[n*n];
    memset(output, 0, n * n * sizeof(output[0]));
    if (!output)
    {
        cout << "Memory allocation failed for output array\n";
    }

    std::random_device entropy_source;

	// 64-bit variant of mt19937 isn't necessary here, but it's just an example
	std::mt19937_64 generator(entropy_source()); 

    std::uniform_real_distribution<float> dist1(-10.0, 10.0);
	std::uniform_real_distribution<float> dist2(-1.0, 1.0);

	// Populate the image array
	for (size_t i = 0; i < n*n; i++) {
		image_arr[i] = dist1(generator);
	}

    // for loops to create an extended array 
    for(size_t i=0; i<(n+2); i++)
    {
        for(size_t j=0; j<(n+2); j++)
        {
            if(((i == 0 || i == n+1) && (j>=1 && j <= n)) || ((j==0 || j== n+1) && (i>=1 && i<=n)))
            {
                image_mod[i*(n+2) + j]=1;
            }
            else if((i==0 || i==n+1) && (j==0 || j==n+1))
            {
                continue;
            }
            else
            {
                image_mod[i*(n+2) + j] = image_arr[(i-1)*n + (j-1)];
            }
        }
    }

    // populate the mask array
    for (size_t i = 0; i < m*m; i++) {
		mask_arr[i] = dist2(generator);
	}

    // Get the starting timestamp
    start = high_resolution_clock::now();
    // omp_set_num_threads(t);
    #pragma omp parallel num_threads(t)
    {
        convolve(image_mod, output, n, mask_arr, m);
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
    // cout<<endl<<endl<<endl;

    // for(size_t i = 0; i < m*m; i++)
    //     cout<<mask_arr[i]<<" ";
    // cout<<endl;

    // for(size_t i = 0; i < n*n; i++)
    //     cout<<output[i]<<" ";
    // cout<<endl;

    // deallocate memory
    delete[] image_arr;
    delete[] image_mod;
    delete[] mask_arr;
    delete[] output;

    return 0;
}