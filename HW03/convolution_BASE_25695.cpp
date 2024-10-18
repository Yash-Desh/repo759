#include "convolution.h"

using namespace std;

void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m)
{
    unsigned int x = 0;
    unsigned int y = 0;

    #pragma omp parallel for 
    for (unsigned int k = 0; k < n * n; k++)
    {
        if (y == n)
        {
            y = 0;
            x++;
        }

        // loop to calculate element value
        float sum = 0;
        unsigned int i = 0, j = 0;
        for (unsigned int l = 0; l < m * m; l++)
        {
            if (j == m)
            {
                j = 0;
                i++;
            }
            float op = 0;
            unsigned int x_index = x + i - (m - 1) / 2;
            unsigned int y_index = y + j - (m - 1) / 2;

            // cout<<i*m+j<<"  : x_index = "<<x_index<<" , y_index = "<<y_index<<endl;

            // determine the op
            if (!(x_index >= 0 && x_index < n) && !(y_index >= 0 && y_index < n))
            {
                op = 0;
            }
            else if ((((x_index >= 0) && (x_index < n)) && ((y_index < 0) || (y_index >= n))) ||
                     (((x_index < 0) || (x_index >= n)) && ((y_index >= 0) && (y_index < n))))
            {
                // cout<<"x_index evaluator ="<<((x_index >= 0) && (x_index < n))<<endl;
                // cout<<"y_index evaluator ="<< ((y_index >= 0) && (y_index < n))<<endl;
                // cout<<"else if entered"<<endl;
                op = 1;
            }
            else
            {
                op = image[x_index * n + y_index];
            }
            // cout<<"op = "<<op<<" ";
            // cout<<"mask value = "<<mask[i * m + j]<<endl;
            #pragma omp atomic
            sum += (op * mask[i * m + j]);
            j++;
        }
        // cout<<endl;
        output[x * n + y] = sum;
        y++;
    }

    // loop to print the output array
    // for (unsigned int k = 0; k < n * n; k++)
    // {
    //     cout << output[k] << " ";
    // }
    // cout << endl;
    
}