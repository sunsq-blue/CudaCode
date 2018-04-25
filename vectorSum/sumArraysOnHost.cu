#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>

/*
 * This example demonstrates a simple vector sum on the host. sumArraysOnHost
 * sequentially iterates through vector elements on the host.
 */

void CHECK(const cudaError_t error)
{
  if (error != cudaSuccess)
  {
    fprintf(stderr, "Error: %s:%s:%d, ", __FILE__, __func__, __LINE__);
    fprintf(stderr, "code:%d, reason: %s\n", error, cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}


__global__ 
void checkIndex(void)
{
  //Print thread's coords within a block and the block's
  //coords within the grid
  printf("Thread coords: (%d %d %d), Block coords: (%d %d %d)\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
} 


__global__
void sumArrays(float *A, float *B, float *C, unsigned int N)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < N)    
      C[idx] = A[idx] + B[idx];
}

__host__
void initialData(float *ip, unsigned int size)
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
    int deviceId = 0;
    cudaSetDevice(deviceId);

    unsigned int nElem = 1<<24;
    size_t nBytes = nElem * sizeof(float);
    printf("%u elements, %zu B\n", nElem, nBytes);

    //Allocate vectors in host memory
    float *h_A, *h_B, *h_C, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    h_C = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    //Allocate vectors in device memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc(&d_A, nBytes));
    CHECK(cudaMalloc(&d_B, nBytes));
    CHECK(cudaMalloc(&d_C, nBytes));
    
    initialData(h_A, nElem);
    initialData(h_B, nElem);

    //Copy the vector data over the device memory
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    int deviceCount = 0, maxDevice = 0;
    unsigned long long maxMemory = 0;
    CHECK(cudaGetDeviceCount(&deviceCount));
    printf("Number of devices: %d\n", deviceCount);
    if (deviceCount > 1)
    {
      int device;
      for (device = 0; device < deviceCount; device++)
      {
        cudaDeviceProp props;
        CHECK(cudaGetDeviceProperties(&props, device));
        if (maxMemory < props.totalGlobalMem)
        {
          maxMemory = props.totalGlobalMem;
          maxDevice = device;  
        }
      }
      CHECK(cudaSetDevice(maxDevice));
      printf("Device with largest total global memory: %d (%llu B)\n", maxDevice, maxMemory);
    }
   
    //cudaDeviceProp prop;
    //CHECK(cudaGetDeviceProperties(&prop,

    //Define the grid and block dimensions
    unsigned int blockSize = 1024; 
    dim3 block(blockSize); 
    dim3 grid((nElem + block.x-1)/block.x);
    printf("Block dim3: <%d %d %d>\n", block.x, block.y, block.z);
    printf("Grid dim3: <%d %d %d>\n", grid.x, grid.y, grid.z);
    unsigned int totalThreads = block.x*block.y*block.z*grid.x*grid.y*grid.z;
    printf("Total threads: %u\n", totalThreads);

    //Invoke the kernel on the device
    //checkIndex<<<grid, block>>>();
   
    //All kernel invocations must return from device
    //before control returns to the host
    //cudaDeviceSynchronize(); 

    sumArrays<<<grid, block>>>(d_A, d_B, d_C, nElem);

    //Copy the result vector back to the host
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

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
