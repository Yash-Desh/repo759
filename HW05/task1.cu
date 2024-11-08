#include <cuda.h>
#include <iostream>

__global__ void simpleKernel()
{
    int result=1;
    for(int i=1; i<=threadIdx.x+1; i++)
    {
        result = result*i;
    }
    std::printf("%d! = %d\n", threadIdx.x+1, result);
    
}

int main()
{
    const int numThreads = 8;

    // invoke GPU kernel, with one block that has 8 threads
    simpleKernel<<<1, numThreads>>>();
    cudaDeviceSynchronize();
    return 0;
    
}