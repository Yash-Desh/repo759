#include "convolution.h"

using namespace std;

void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m)
{
<<<<<<< HEAD
    #pragma omp for collapse(3)
    for (size_t x = 0; x < n; x++)
=======
    unsigned int x = 0;
    unsigned int y = 0;

    #pragma omp for 
    for (unsigned int k = 0; k < n * n; k++)
>>>>>>> 4489d32496f150584e84e3dd86688b7504089cb5
    {
        for (size_t y = 0; y < n; y++)
        {
            for (size_t i = 0; i < m; i++)
            {
                for (size_t j = 0; j < m; j++)
                {
                    #pragma omp atomic
                    output[x * n + y] += mask[i * m + j] * image[(x + i) * (n + 2) + (y + j)];
                }
            }
        }
    }
<<<<<<< HEAD
}
=======

    // loop to print the output array
    // for (unsigned int k = 0; k < n * n; k++)
    // {
    //     cout << output[k] << " ";
    // }
    // cout << endl;
    
}
>>>>>>> 4489d32496f150584e84e3dd86688b7504089cb5
