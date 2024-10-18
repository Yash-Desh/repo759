#include <iostream>
#include <random>  // To generate random numbers
#include <cstring> // To use memset
#include "matmul.h"

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

    // n is the dimension of A, B, C given as 1st command line argument
    size_t n = atoi(argv[1]);
    // cout<<n<<endl;

    // t is the number of threads given as the 2nd command line argument
    int t = atoi(argv[2]);

    // array to store n*n random float values from -10.0 to 10.0 for image
    float *A = new float[n * n];
    if (!A)
    {
        cout << "Memory allocation failed for image array of random floats\n";
    }

    // array to store the m*m random float values from -1.0 to 1.0 for mask
    float *B = new float[n * n];
    if (!B)
    {
        cout << "Memory allocation failed for mask array of random floats\n";
    }

    // array to store n*n float values for output
    float *C = new float[n * n];
    memset(C, 0, n * n * sizeof(C[0]));

    std::random_device entropy_source;

    // 64-bit variant of mt19937 isn't necessary here, but it's just an example
    std::mt19937_64 generator(entropy_source());

    std::uniform_real_distribution<float> dist1(-10.0, 10.0);
    std::uniform_real_distribution<float> dist2(-1.0, 1.0);

    // Write a random value to each slot in N
    for (size_t i = 0; i < n * n; i++)
    {
        A[i] = dist1(generator);
    }

    for (size_t i = 0; i < n * n; i++)
    {
        B[i] = dist2(generator);
    }

    // Get the starting timestamp
    start = high_resolution_clock::now();

    omp_set_num_threads(t);
#pragma omp parallel
    {
        mmul(A, B, C, n);
    }
    end = high_resolution_clock::now();
    // Convert the calculated duration to a double using the standard library
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    // 1st element of resultant array C
    cout << C[0] << endl;
    // Last element of resultant array C
    cout << C[n * n - 1] << endl;
    // Durations are converted to milliseconds already thanks to std::chrono::duration_cast
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
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}