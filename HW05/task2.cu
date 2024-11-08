#include<cuda.h>
#include<iostream>
#include <random>  // To generate random numbers

__global__ void simpleKernel(int* data, int a)
{
    //this adds a value to a variable stored in global memory
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    data[index] = a*threadIdx.x + blockIdx.x;
}

int main()
{
    std::random_device entropy_source;

	// 64-bit variant of mt19937 isn't necessary here, but it's just an example
	std::mt19937_64 generator(entropy_source()); 

	std::uniform_real_distribution<double> dist(10., 20.);

    int a=dist(generator);

    const int numElems = 16;
    int hA[numElems], *dA;

    //allocate memory on the device (GPU); zero out all entries in this device array
    cudaMalloc((void**)&dA, sizeof(int) * numElems);
    cudaMemset(dA, 0, numElems * sizeof(int));

    //invoke GPU kernel, with one block that has four threads
    simpleKernel<<<2,8>>>(dA, a);

    //bring the result back from the GPU into the hostArray
    cudaMemcpy(&hA, dA, sizeof(int) * numElems, cudaMemcpyDeviceToHost);
    
    //print out the result to confirm that things are looking good
    for (int i = 0; i < numElems; i++)
    std::cout << hA[i] << " ";

    std::cout<<"\n";
    //release the memory allocated on the GPU
    cudaFree(dA);
    return 0;

}