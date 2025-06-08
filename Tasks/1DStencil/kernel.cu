
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define VECTOR_SIZE 1024 * 1024 * 512
#define THREADS_PER_BLOCK 256
#define STENCIL_RADIUS 10

#define CEIL(A, B) ((A) + (B) - 1) / (B)

#define BLOCK_NUM CEIL(VECTOR_SIZE, THREADS_PER_BLOCK)

#define CudaCheckError(error) \
	if (error != cudaSuccess) \
		goto exit;

//#define DEBUG_PRINT
#ifdef DEBUG_PRINT
#define DEBUG_ARRAY_PRINT(ARRAY, SIZE) \
	for (int i = 0; i < SIZE; i++) \
		printf("%d ", ARRAY[i]); \
	puts("");
#else
#define DEBUG_ARRAY_PRINT(ARRAY, SIZE)
#endif

__global__ void performStencil(int* in, int* out, size_t size)
{
#define LOCAL_MEMORY_SIZE THREADS_PER_BLOCK + 2 * STENCIL_RADIUS
	__shared__ int localMemory[LOCAL_MEMORY_SIZE];

	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= VECTOR_SIZE)
		return;

	if (threadIdx.x < STENCIL_RADIUS && idx >= STENCIL_RADIUS)
		localMemory[threadIdx.x] = in[idx - STENCIL_RADIUS];

	if (THREADS_PER_BLOCK - threadIdx.x < STENCIL_RADIUS && VECTOR_SIZE - idx >= STENCIL_RADIUS)
		localMemory[threadIdx.x] = in[idx - STENCIL_RADIUS];

	int localIdx = threadIdx.x + STENCIL_RADIUS;
	localMemory[localIdx] = in[idx];

	__syncthreads();

	int sum = 0;
	for (int i = localIdx - STENCIL_RADIUS; i < localIdx + STENCIL_RADIUS; i++)
		sum += localMemory[i];

	out[idx] = sum;
}

void fillArray(int* array, size_t size)
{
	for (int i = 0; i < size; i++)
		array[i] = i;
}

int main(void)
{
	int* hIn = NULL, * hOut = NULL, * dIn = NULL, * dOut = NULL;

	CudaCheckError(cudaHostAlloc(&hIn, VECTOR_SIZE * sizeof(int), 0));
	CudaCheckError(cudaHostAlloc(&hOut, VECTOR_SIZE * sizeof(int), 0));

	CudaCheckError(cudaMalloc(&dIn, VECTOR_SIZE * sizeof(int)));
	CudaCheckError(cudaMalloc(&dOut, VECTOR_SIZE * sizeof(int)));

	fillArray(hIn, VECTOR_SIZE);
	DEBUG_ARRAY_PRINT(hIn, VECTOR_SIZE);

	CudaCheckError(cudaMemcpy(dIn, hIn, VECTOR_SIZE * sizeof(int), cudaMemcpyHostToDevice));

	performStencil<<<BLOCK_NUM, THREADS_PER_BLOCK>>>(dIn, dOut, VECTOR_SIZE);
	CudaCheckError(cudaGetLastError());
	CudaCheckError(cudaDeviceSynchronize());

	CudaCheckError(cudaMemcpy(hOut, dOut, VECTOR_SIZE * sizeof(int), cudaMemcpyDeviceToHost));
	DEBUG_ARRAY_PRINT(hOut, VECTOR_SIZE);

exit:
	if (hIn)
		cudaFreeHost(hIn);
	if (hOut)
		cudaFreeHost(hOut);

	if (dIn)
		cudaFree(dIn);
	if (dOut)
		cudaFree(dOut);
	
	return 0;
}