#include <iostream>
#include "matmul.h"
using namespace std;

void mmul(const float *A, const float *B, float *C, const std::size_t n)
{
    #pragma omp parallel for collapse(3)
    for (unsigned int i = 0; i < n; i++)
    {
        // loops over the columns of c
        for (unsigned int k = 0; k < n; k++)
        {
            // loop for rhs
            for (unsigned int j = 0; j < n; j++)
            {
                #pragma omp atomic
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}
