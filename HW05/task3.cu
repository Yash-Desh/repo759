
#include <cstdio>
#include <cuda.h>
#include <cstdlib>
#include <iostream>
#include <random>
#include "vscale.cuh"



int main ( int argc , char* argv[] ) {
   std::random_device rd;  // a seed source for the random number engine
   std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
   std::uniform_real_distribution<> distribution_a(-10., 10.);
   std::uniform_real_distribution<> distribution_b(0., 1.);

   int N = std::atoi(argv[1]);
   
   float* h_a = new float[N] ;
   float* h_b = new float[N] ;

   for (int i = 0 ; i < N ; ++i) {
     h_a[i] = distribution_a(gen) ;
     h_b[i] = distribution_b(gen) ;
   }	   
   
   float* d_a  ;
   float* d_b  ;
   cudaMalloc((void**) &d_a , sizeof(float) * N );
   cudaMalloc((void**) &d_b , sizeof(float) * N );


   cudaMemcpy(d_a,h_a,sizeof(float) * N,cudaMemcpyHostToDevice);
   cudaMemcpy(d_b,h_b,sizeof(float) * N,cudaMemcpyHostToDevice);
   
   const int threadsPerBlock = 512;
   const int blocksPerGrid   = ( N +  threadsPerBlock - 1 ) / threadsPerBlock ;

// Create CUDA events for timing
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

    // Start recording
   cudaEventRecord(start,0);

   vscale<<<blocksPerGrid,threadsPerBlock>>>(d_a,d_b,N);

    // Stop recording
   cudaEventRecord(stop,0);
   cudaEventSynchronize(stop);

    // Calculate elapsed time
   float elapsedTime = 0;
   cudaEventElapsedTime(&elapsedTime, start, stop);


   cudaMemcpy(h_b,d_b,sizeof(float) * N, cudaMemcpyDeviceToHost);

   std::cout << elapsedTime << "ms" << std::endl ;
   std::cout << h_b[0] << std::endl ;
   std::cout << h_b[N-1] << std::endl ;



   delete[] h_a;
   delete[] h_b;

   cudaFree(d_a);
   cudaFree(d_b);

   cudaEventDestroy(start);
   cudaEventDestroy(stop);
   return 0 ;


}	