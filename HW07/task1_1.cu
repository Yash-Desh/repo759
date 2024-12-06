#include <iostream>
#include <cuda.h>
#include <random>
#include "matmul.cuh"



__global__ void warm_up_kernel(){}



int main(int argc, char*argv[])
{
    cudaEvent_t start;
    cudaEvent_t stop;
    float ms1,ms2,ms3;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int n = std::stoi(argv[1]);
    int block_dim = std::stoi(argv[2]);
    std::random_device entropy_source;
    std::mt19937_64 generator(entropy_source());
    std::uniform_int_distribution <int> dist1(-10,10);
    std::uniform_real_distribution <float> dist2(-10.0,10.0);
    std::uniform_real_distribution <double> dist3(-10.0,10.0);


    int*A1 = (int*)malloc(n*n*(sizeof(int)));
    int*B1 = (int*)malloc(n*n*(sizeof(int)));
    int*C1 = (int*)malloc(n*n*(sizeof(int)));
    int*d_A1,*d_B1,*d_C1;

    float*A2 = (float*)malloc(n*n*(sizeof(float)));
    float*B2 = (float*)malloc(n*n*(sizeof(float)));
    float*C2 = (float*)malloc(n*n*(sizeof(float)));
    float*d_A2,*d_B2,*d_C2;

    double*A3 = (double*)malloc(n*n*(sizeof(double)));
    double*B3 = (double*)malloc(n*n*(sizeof(double)));
    double*C3 = (double*)malloc(n*n*(sizeof(double)));
    double*d_A3,*d_B3,*d_C3;

    for(int i = 0; i < n*n; i++)
    {
        A1[i] = dist1(generator);
        B1[i] = dist1(generator);
        A2[i] = dist2(generator);
        B2[i] = dist2(generator);
        A3[i] = dist3(generator);
        B3[i] = dist3(generator);
    }
    /*for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            std::cout<<A[i*n+j]<<" ";
            if(j == n-1)
            {
                std::cout<<std::endl;
            }
        }
    }
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            std::cout<<B[i*n+j]<<" ";
            if(j == n-1)
            {
                std::cout<<std::endl;
            }
        }
    }*/

    cudaMalloc((void**)&d_A1,sizeof(int) * n*n);
    cudaMalloc((void**)&d_B1,sizeof(int) * n*n);
    cudaMalloc((void**)&d_C1,sizeof(int) * n*n);

    cudaMalloc((void**)&d_A2,sizeof(float) * n*n);
    cudaMalloc((void**)&d_B2,sizeof(float) * n*n);
    cudaMalloc((void**)&d_C2,sizeof(float) * n*n);

    cudaMalloc((void**)&d_A3,sizeof(double) * n*n);
    cudaMalloc((void**)&d_B3,sizeof(double) * n*n);
    cudaMalloc((void**)&d_C3,sizeof(double) * n*n);


    cudaMemcpy(d_A1,A1,sizeof(int)*n*n,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B1,B1,sizeof(int)*n*n,cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1,C1,sizeof(int)*n*n,cudaMemcpyHostToDevice);

    cudaMemcpy(d_A2,A2,sizeof(float)*n*n,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B2,B2,sizeof(float)*n*n,cudaMemcpyHostToDevice);
    cudaMemcpy(d_C2,C2,sizeof(float)*n*n,cudaMemcpyHostToDevice);

    cudaMemcpy(d_A3,A3,sizeof(double)*n*n,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B3,B3,sizeof(double)*n*n,cudaMemcpyHostToDevice);
    cudaMemcpy(d_C3,C3,sizeof(double)*n*n,cudaMemcpyHostToDevice);

    warm_up_kernel<<<1,1>>>();

    cudaEventRecord(start);
    matmul_1(d_A1,d_B1,d_C1,n,block_dim);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaMemcpy(C1,d_C1,sizeof(int)*n*n,cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&ms1, start, stop);

    cudaEventRecord(start);
    matmul_2(d_A2,d_B2,d_C2,n,block_dim);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaMemcpy(C2,d_C2,sizeof(float)*n*n,cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&ms2, start, stop);

    cudaEventRecord(start);
    matmul_3(d_A3,d_B3,d_C3,n,block_dim);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaMemcpy(C3,d_C3,sizeof(double)*n*n,cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&ms3, start, stop);



    /*for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            std::cout<<C[i*n+j]<<" ";
            if(j == n-1)
            {
                std::cout<<std::endl;
            }
        }
    }
    */
    std::cout<<C1[0]<<std::endl;
    std::cout<<C1[(n-1)*n + n-1];
    std::cout<<std::endl; 
    std::cout<<"Time ELapsed "<<ms1;
    std::cout<<std::endl;
    std::cout<<std::endl;

    std::cout<<C2[0]<<std::endl;
    std::cout<<C2[(n-1)*n + n-1];
    std::cout<<std::endl; 
    std::cout<<"Time ELapsed "<<ms2;
    std::cout<<std::endl;
    std::cout<<std::endl;

    std::cout<<C3[0]<<std::endl;
    std::cout<<C3[(n-1)*n + n-1];
    std::cout<<std::endl; 
    std::cout<<"Time ELapsed "<<ms3;
    std::cout<<std::endl;
    std::cout<<std::endl;




    cudaFree(d_A1);
    cudaFree(d_B1);
    cudaFree(d_C1);
    free(A1);
    free(B1);
    free(C1);
    cudaFree(d_A2);
    cudaFree(d_B2);
    cudaFree(d_C2);
    free(A2);
    free(B2);
    free(C2);
    cudaFree(d_A3);
    cudaFree(d_B3);
    cudaFree(d_C3);
    free(A3);
    free(B3);
    free(C3);

}
