#include <iostream>
#include "matmul.h"

using namespace std;

void mmul1(const double *A, const double *B, double *C, const unsigned int n)
{
    for (unsigned int i = 0; i < n; i++)
    {
        // loops over the columns of c
        for (unsigned int j = 0; j < n; j++)
        {
            // loop for rhs
            for (unsigned int k = 0; k < n; k++)
            {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

void mmul2(const double *A, const double *B, double *C, const unsigned int n)
{
    for (unsigned int i = 0; i < n; i++)
    {
        // loops over the columns of c
        for (unsigned int k = 0; k < n; k++)
        {
            // loop for rhs
            for (unsigned int j = 0; j < n; j++)
            {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

void mmul3(const double *A, const double *B, double *C, const unsigned int n)
{
    for (unsigned int j = 0; j < n; j++)
    {
        // loops over the columns of c
        for (unsigned int k = 0; k < n; k++)
        {
            // loop for rhs
            for (unsigned int i = 0; i < n; i++)
            {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

void mmul4(const std::vector<double> &A, const std::vector<double> &B, double *C, const unsigned int n)
{
    for (unsigned int i = 0; i < n; i++)
    {
        // loops over the columns of c
        for (unsigned int j = 0; j < n; j++)
        {
            // loop for rhs
            for (unsigned int k = 0; k < n; k++)
            {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}