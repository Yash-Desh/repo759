#include <iostream>
#include <random>  // To generate random numbers
#include "msort.h"
using namespace std;

// The std::chrono namespace provides timer functions in C++
#include <chrono>

// std::ratio provides easy conversions between metric units
#include <ratio>
using std::chrono::duration;
using std::chrono::high_resolution_clock;
/*
Prints the given array
Input Parameters
1. array to print
2. size of array
returns void

*/
void printArr(int *arr, int n)
{
    for (int i = 0; i < n; i++)
    {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char *argv[])
{
    
     // declarations needed for timer
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    // n & t are given as command line arguments
    size_t n = atoi(argv[1]);

    // t is the number of threads
    size_t t = atoi(argv[2]);

    // ts is the threshold below which bubble sort will be called
    size_t ts = atoi(argv[3]);
    
    // array to store n*n random float values from -10.0 to 10.0 for image
    int *arr = new int[n];
    if (!arr)
    {
        cout << "Memory allocation failed for array of random ints\n";
    }
    // cout<<"Array allocated\n";
    std::random_device entropy_source;

	// 64-bit variant of mt19937 isn't necessary here, but it's just an example
	std::mt19937_64 generator(entropy_source()); 

    std::uniform_real_distribution<float> dist(-1000, 1000);
    
    // Write a random value to each slot in N
	for (size_t i = 0; i < n; i++) {
		arr[i] = dist(generator);
	}
    
    // Get the starting timestamp
    start = high_resolution_clock::now();
    omp_set_num_threads(t);
    #pragma omp parallel
    {
        // cout << "Before msort called omp_get_num_threads() =" << omp_get_num_threads() << endl;
        msort(arr, n, ts);
    }
    end = high_resolution_clock::now();
    // Convert the calculated duration to a double using the standard library
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    cout << arr[0] << endl;
    cout << arr[n - 1] << endl;
    cout << duration_sec.count() << endl;
    
    // int *arr = new int[6];
    // arr[0] = 3;
    // arr[1] = 7;
    // arr[2] = 10;
    // arr[3] = 2;
    // arr[4] = 1;
    // arr[5] = 3;

    // printArr(arr, 6);

    // int t = 3;
    // omp_set_num_threads(t);
// set max number of threads to t
// #pragma omp parallel
//     {
        
//         cout << "Before msort called omp_get_num_threads() =" << omp_get_num_threads() << endl;
//         msort(arr, 6, 6);
//     }
    // #pragma omp parallel num_threads(t)
    //     {
    //         cout << "Before msort called omp_get_num_threads() =" << omp_get_num_threads() << endl;
    //         msort(arr, 6, 3);
    //     }

    //printArr(arr, n);
    return 0;
}