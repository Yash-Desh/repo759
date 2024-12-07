#include <iostream>
#include <cuda.h>
#include "matmul.cuh"
#include <cuda_runtime.h>
#include <random>           // To generate random numbers
using namespace std;


int main(int argc, char *argv[])
{
    // command line arguments
    unsigned int n = std::stoi(argv[1]);                // arrray dimension
    unsigned int block_dim = std::stoi(argv[2]);        // block dimension
    
    // declarations for calculating time
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms_1=0;   
    float ms_2=0;
    float ms_3=0;                                                // time in ms

    // random number generation
    std::random_device entropy_source;
    std::mt19937_64 generator(entropy_source());

    std::uniform_int_distribution<int> dist1(-1, 1);
    std::uniform_real_distribution<float> dist2(-1.0, 1.0);
    std::uniform_real_distribution<double> dist3(-1.0, 1.0);


    // -------------------- host arrays --------------------
    int *h_A_1= (int*)malloc(n*n*(sizeof(int)));
    int *h_B_1= (int*)malloc(n*n*(sizeof(int)));
    int *h_C_1= (int*)malloc(n*n*(sizeof(int)));

    float *h_A_2= (float*)malloc(n*n*(sizeof(float)));
    float *h_B_2= (float*)malloc(n*n*(sizeof(float)));
    float *h_C_2= (float*)malloc(n*n*(sizeof(float)));

    double *h_A_3= (double*)malloc(n*n*(sizeof(double)));
    double *h_B_3= (double*)malloc(n*n*(sizeof(double)));
    double *h_C_3= (double*)malloc(n*n*(sizeof(double)));



    for (size_t i = 0; i < n*n; i++)
    {
        h_A_1[i] = dist1(generator);
        h_B_1[i] = dist1(generator);
        h_C_1[i] = 0;

        h_A_2[i] = dist2(generator);
        h_B_2[i] = dist2(generator);
        h_C_2[i] = 0;

        h_A_3[i] = dist3(generator);
        h_B_3[i] = dist3(generator);
        h_C_3[i] = 0;
    }

    // -------------------- device arrays --------------------
    int *d_A_1, *d_B_1, *d_C_1;
    float *d_A_2, *d_B_2, *d_C_2;
    double *d_A_3, *d_B_3, *d_C_3;

    // -------------------- allocate memory on the device (GPU) --------------------
    cudaMalloc((void **)&d_A_1, n * n *sizeof(int));
    cudaMalloc((void **)&d_B_1, n * n* sizeof(int));
    cudaMalloc((void **)&d_C_1, n * n* sizeof(int));

    cudaMalloc((void **)&d_A_2, n * n *sizeof(float));
    cudaMalloc((void **)&d_B_2, n * n* sizeof(float));
    cudaMalloc((void **)&d_C_2, n * n* sizeof(float));


    cudaMalloc((void **)&d_A_3, n * n *sizeof(double));
    cudaMalloc((void **)&d_B_3, n * n* sizeof(double));
    cudaMalloc((void **)&d_C_3, n * n* sizeof(double));


    // -------------------- copy data into device blocks --------------------

    cudaMemcpy(d_A_1, h_A_1, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_1, h_B_1, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_1, h_C_1, n * n * sizeof(int), cudaMemcpyHostToDevice);


    cudaMemcpy(d_A_2, h_A_2, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_2, h_B_2, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_2, h_C_2, n * n * sizeof(float), cudaMemcpyHostToDevice);


    cudaMemcpy(d_A_3, h_A_3, n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_3, h_B_3, n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_3, h_C_3, n * n * sizeof(double), cudaMemcpyHostToDevice);

    // ----------------------------------- int -----------------------------------

    // #############################################
    cudaEventRecord(start);
    matmul_1(d_A_1, d_B_1, d_C_1, n, block_dim);
    cudaEventRecord(stop);
    // #############################################
    
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_1, start, stop);

    cudaMemcpy(h_C_1, d_C_1, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    cout<<"hello world"<<endl;
    std::cout << h_C_1[0] << std::endl;
    std::cout << h_C_1[n*n - 1] << std::endl;
    std::cout << ms_1 << std::endl;


    // ----------------------------------- float -----------------------------------

    // #############################################
    cudaEventRecord(start);
    matmul_2(d_A_2, d_B_2, d_C_2, n, block_dim);
    cudaEventRecord(stop);
    // #############################################
    
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_2, start, stop);

    cudaMemcpy(h_C_2, d_C_2, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    cout<<"hello world"<<endl;
    std::cout << h_C_2[0] << std::endl;
    std::cout << h_C_2[n*n - 1] << std::endl;
    std::cout << ms_2 << std::endl;


    // ----------------------------------- double -----------------------------------

    // #############################################
    cudaEventRecord(start);
    matmul_3(d_A_3, d_B_3, d_C_3, n, block_dim);
    cudaEventRecord(stop);
    // #############################################
    
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_3, start, stop);

    cudaMemcpy(h_C_3, d_C_3, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // cout<<"hello world"<<endl;
    std::cout << h_C_3[0] << std::endl;
    std::cout << h_C_3[n*n - 1] << std::endl;
    std::cout << ms_3 << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // deallocate memory
    // cudaFree(d_A);
    // cudaFree(d_B);
    // cudaFree(d_C);
    // free(A);
    // free(B);
    // free(C);


    return 0;
    
}
