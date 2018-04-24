#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>

/*
 * This example demonstrates a simple vector sum on the host. sumArraysOnHost
 * sequentially iterates through vector elements on the host.
 */

__global__ 
void checkIndex(void)
{
  //Print thread's coords within a block and the block's
  //coords within the grid
  printf("Thread coords: (%d %d %d), Block coords: (%d %d %d)\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
} 


__global__
void sumArrays(float *A, float *B, float *C)
{
    unsigned int idx = threadIdx.x;
    C[idx] = A[idx] + B[idx];
}

__host__
void initialData(float *ip, int size)
{
    // generate different seed for random number
    time_t t;
    srand((unsigned) time(&t));

    int i;
    for (i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

__host__
int main(int argc, char **argv)
{
    int nElem = 6;
    size_t nBytes = nElem * sizeof(float);
    printf("%d elements, %zu B\n", nElem, nBytes);

    //Allocate vectors in host memory
    float *h_A, *h_B, *h_C, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    h_C = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    //Allocate vectors in device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, nBytes);
    cudaMalloc(&d_B, nBytes);
    cudaMalloc(&d_C, nBytes);
    
    initialData(h_A, nElem);
    initialData(h_B, nElem);

    //Copy the vector data over the device memory
    cudaError_t success = cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    
    if (success != cudaSuccess) //Sample error handling code
    {
      const char *e = cudaGetErrorString(success);
      fprintf(stderr, "%s\n", e);
      exit(EXIT_FAILURE); 
    }

    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    //Define the grid and block dimensions
    dim3 block(3); 
    dim3 grid((nElem + block.x-1)/block.x);
    printf("Block dim3: <%d %d %d>\n", block.x, block.y, block.z);
    printf("Grid dim3: <%d %d %d>\n", grid.x, grid.y, grid.z);

    //Invoke the kernel on the device
    checkIndex<<<grid, block>>>();
   
    //All kernel invocations must return from device
    //before control returns to the host
    cudaDeviceSynchronize(); 

    sumArrays<<<1, nElem>>>(d_A, d_B, d_C);

    //Copy the result vector back to the host
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    int i; 
    for (i = 0; i < nElem; i++)
      fprintf(stdout, "%f ", h_A[i]); 
    fprintf(stdout, "\n");
    for (i = 0; i < nElem; i++)
      fprintf(stdout, "%f ", h_B[i]); 
    fprintf(stdout, "\n");
    for (i = 0; i < nElem; i++)
      fprintf(stdout, "%f ", gpuRef[i]); 
    fprintf(stdout, "\n");

    //Deallocate the device memory vectors
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    //Done using the device. Clean up all state on the device
    //and flush profiling data.
    cudaDeviceReset();

    //Deallocate the host memory vectors
    free(h_A);
    free(h_B);
    free(h_C);
    free(gpuRef);

    return 0;
}
