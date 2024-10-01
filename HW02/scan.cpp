#include "scan.h"

// Performs an inclusive scan on input array arr and stores
// the result in the output array
// arr and output are arrays of n elements
void scan(const float *arr, float *output, std::size_t n)
{
    float sum =0;
    for (size_t i=0; i<n; i++)
    {
        sum += arr[i];
        output[i] = sum; 
    }
}





