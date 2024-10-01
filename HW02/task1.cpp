#include <iostream>
#include <cstdlib> // To generate random numbers
#include <random>  // To generate random numbers
#include "scan.h" // To call the scan function

// The std::chrono namespace provides timer functions in C++
#include <chrono>

// std::ratio provides easy conversions between metric units
#include <ratio>
using std::chrono::duration;
using std::chrono::high_resolution_clock;

using namespace std;

int main(int argc, char *argv[])
{
    // declarations needed for timer
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    // convert command line argument to integer
    size_t n = atoi(argv[1]);

    // array to store n random float values from -1.0 to 1.0
    float *rand_arr = new float[n];
    if (!rand_arr)
    {
        cout << "Memory allocation failed for array of random floats\n";
    }

    // array to store the result after scan function
    float *scan_arr = new float[n];
    if (!scan_arr)
    {
        cout << "Memory allocation failed for scan array of random floats\n";
    }

    // // upper bound & lower bound of the random values defined
    // int lb = -100, ub = 100;
    // // This program will create some sequence of random
    // // numbers on every program run within range lb to ub
    // for (unsigned long int i = 0; i < n; i++)
    // {
    //     rand_arr[i] = float(((rand() % (ub - lb + 1)) + lb) / 100.0);
    // }

    std::random_device entropy_source;

	// 64-bit variant of mt19937 isn't necessary here, but it's just an example
	std::mt19937_64 generator(entropy_source()); 

	std::uniform_real_distribution<float> dist(-1., 1.);

	// Write a random value to each slot in N
	for (size_t i = 0; i < n; i++) {
		rand_arr[i] = dist(generator);
	}


    // // debug loop : Print the rand_arr array
    // for(int i=0; i<n; i++)
    // {
    //     cout<<rand_arr[i]<<" ";
    // }
    // cout<<endl;

    // Get the starting timestamp
    start = high_resolution_clock::now();
    scan(rand_arr, scan_arr, n);
    end = high_resolution_clock::now();
    // Convert the calculated duration to a double using the standard library
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    // Durations are converted to milliseconds already thanks to std::chrono::duration_cast
    cout << duration_sec.count() << endl;
    cout << scan_arr[0] << endl;
    cout << scan_arr[n - 1] << endl;

    // // debug loop : Print the scan_arr array
    // for (int i = 0; i < n; i++)
    // {
    //     cout << scan_arr[i] << " ";
    // }
    // cout << endl;

    // deallocate memory
    delete[] rand_arr;
    delete[] scan_arr;

    return 0;
}
