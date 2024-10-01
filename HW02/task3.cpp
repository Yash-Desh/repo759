#include <iostream>
#include <random>  // To generate random numbers
#include <cstring> // To use memset
#include <vector>
#include "matmul.h"

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
    duration<double, std::milli> duration_sec1;
    duration<double, std::milli> duration_sec2;
    duration<double, std::milli> duration_sec3;
    duration<double, std::milli> duration_sec4;

    // n & m are given as command line arguments
    size_t n = 1024;
    cout<<n<<endl;

    // array to store n*n random float values from -10.0 to 10.0 for image
    vector <double> A1;
    double *A = new double[n*n];
    if (!A)
    {
        cout << "Memory allocation failed for array of random floats\n";
    }

    // array to store the m*m random float values from -1.0 to 1.0 for mask
    vector <double> B1;
    double *B = new double[n*n];
    if (!B)
    {
        cout << "Memory allocation failed for scan array of random floats\n";
    }

    // array to store n*n float values for output
    double *C = new double[n*n];
    memset(C, 0, n*n*sizeof(C[0]));

    std::random_device entropy_source;

	// 64-bit variant of mt19937 isn't necessary here, but it's just an example
	std::mt19937_64 generator(entropy_source()); 

    std::uniform_real_distribution<float> dist1(-10.0, 10.0);
	std::uniform_real_distribution<float> dist2(-1.0, 1.0);

	// Write a random value to each slot in N
	for (size_t i = 0; i < n*n; i++) {
		A[i] = dist1(generator);
        A1.push_back(A[i]);
	}

    for (size_t i = 0; i < n*n; i++) {
		B[i] = dist2(generator);
        B1.push_back(B[i]);
	}

    // Get the starting timestamp
    start = high_resolution_clock::now();
    mmul1(A, B, C, n);
    end = high_resolution_clock::now();
    // Convert the calculated duration to a double using the standard library
    duration_sec1 = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    // Durations are converted to milliseconds already thanks to std::chrono::duration_cast
    cout << duration_sec1.count() << endl;
    cout << C[n*n - 1] << endl;
    memset(C, 0, n*n*sizeof(C[0]));


    // Get the starting timestamp
    start = high_resolution_clock::now();
    mmul2(A, B, C, n);
    end = high_resolution_clock::now();
    // Convert the calculated duration to a double using the standard library
    duration_sec2 = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    // Durations are converted to milliseconds already thanks to std::chrono::duration_cast
    cout << duration_sec2.count() << endl;
    cout << C[n*n - 1] << endl;
    memset(C, 0, n*n*sizeof(C[0]));


    // Get the starting timestamp
    start = high_resolution_clock::now();
    mmul3(A, B, C, n);
    end = high_resolution_clock::now();
    // Convert the calculated duration to a double using the standard library
    duration_sec3 = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    // Durations are converted to milliseconds already thanks to std::chrono::duration_cast
    cout << duration_sec3.count() << endl;
    cout << C[n*n - 1] << endl;
    memset(C, 0, n*n*sizeof(C[0]));


    // Get the starting timestamp
    start = high_resolution_clock::now();
    mmul4(A1, B1, C, n);
    end = high_resolution_clock::now();
    // Convert the calculated duration to a double using the standard library
    duration_sec4 = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    // Durations are converted to milliseconds already thanks to std::chrono::duration_cast
    cout << duration_sec4.count() << endl;
    cout << C[n*n - 1] << endl;
    memset(C, 0, n*n*sizeof(C[0]));

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