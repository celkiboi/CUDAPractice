
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define VECTOR_SIZE 1024 * 1024 * 512
#define THREADS_PER_BLOCK 256

#define CEIL(A, B) ((A) + (B) - 1) / (B)

#define BLOCK_NUM CEIL(VECTOR_SIZE, THREADS_PER_BLOCK)

#define CudaCheckError(error) \
	if (error != cudaSuccess) \
		goto exit;

__global__ void add(int* a, int* b, int* result, size_t length)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < length)
		result[idx] = a[idx] + b[idx];
}

void fillArray(int* array, size_t size) 
{
	for (int i = 0; i < size; i++)
		array[i] = i;
}

bool checkResult(int* a, int* b, int* result, size_t size)
{
	for (int i = 0; i < size; i++)
	{
		if (a[i] + b[i] != result[i])
			return false;
	}
	return true;
}

int main(void)
{
	int* hA = NULL, * hB = NULL, * dA = NULL, * dB = NULL, * hResult = NULL, * dResult = NULL;

	CudaCheckError(cudaHostAlloc(&hA, VECTOR_SIZE * sizeof(int), 0));
	CudaCheckError(cudaHostAlloc(&hB, VECTOR_SIZE * sizeof(int), 0));
	CudaCheckError(cudaHostAlloc(&hResult, VECTOR_SIZE * sizeof(int), 0));

	CudaCheckError(cudaMalloc(&dA, VECTOR_SIZE * sizeof(int)));
	CudaCheckError(cudaMalloc(&dB, VECTOR_SIZE * sizeof(int)));
	CudaCheckError(cudaMalloc(&dResult, VECTOR_SIZE * sizeof(int)));

	fillArray(hA, VECTOR_SIZE);
	fillArray(hB, VECTOR_SIZE);

	CudaCheckError(cudaMemcpy(dA, hA, VECTOR_SIZE * sizeof(int), cudaMemcpyHostToDevice));
	CudaCheckError(cudaMemcpy(dB, hB, VECTOR_SIZE * sizeof(int), cudaMemcpyHostToDevice));

	add<<<BLOCK_NUM, THREADS_PER_BLOCK>>>(dA, dB, dResult, VECTOR_SIZE);
	CudaCheckError(cudaGetLastError());
	CudaCheckError(cudaDeviceSynchronize());

	CudaCheckError(cudaMemcpy(hResult, dResult, VECTOR_SIZE * sizeof(int), cudaMemcpyDeviceToHost));
	puts(checkResult(hA, hB, hResult, VECTOR_SIZE) ? "Result ok" : "Result not ok");

exit:
	if (hA) 
		cudaFreeHost(hA);
	if (hB) 
		cudaFreeHost(hB);
	if (hResult) 
		cudaFreeHost(hResult);

	if (dA)
		cudaFree(dA);
	if (dB)
		cudaFree(dB);
	if (dResult)
		cudaFree(dResult);
	return 0;
}