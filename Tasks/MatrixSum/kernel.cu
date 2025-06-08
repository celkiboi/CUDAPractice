
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define M 1025 * 512
#define N 1023
#define MATRIX_VECTOR_SIZE (M) * (N)

#define CEIL(A, B) (((A) + (B) - 1) / (B))

#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16

#define BLOCKS_X CEIL(N, THREADS_PER_BLOCK_X)
#define BLOCKS_Y CEIL(M, THREADS_PER_BLOCK_Y)

#define CudaCheckError(error) \
	if (error != cudaSuccess) \
		goto exit;

//#define DEBUG_PRINT
#ifdef DEBUG_PRINT
#define DEBUG_MATRIX_PRINT(MATRIX, DIM_M, DIM_N) \
	for (int i = 0; i < DIM_M; i++) \
	{ \
		for (int j = 0; j < DIM_N; j++) \
			printf("%d ", MATRIX[i * DIM_N + j]); \
		puts(""); \
	}
#else
#define DEBUG_MATRIX_PRINT(MATRIX, DIM_M, DIM_N)
#endif


__global__ void sum(int* matrixA, int* matrixB, size_t m, size_t n, int* result)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = row * n + col;

	if (row >= m || col >= n)
		return;

	result[idx] = matrixA[idx] + matrixB[idx];
}

void fillMatrix(int* matrix, size_t matrixTotalSize)
{
	for (int i = 0; i < matrixTotalSize; i++)
		matrix[i] = i;
}

bool checkResult(int* matrixA, int* matrixB, int* result, size_t m, size_t n)
{
	for (int i = 0; i < m * n; i++)
	{
		if (matrixA[i] + matrixB[i] != result[i])
			return false;
	}

	return true;
}

int main(void)
{
	bool success = false;
	int* hA = NULL, * hB = NULL, * dA = NULL, * dB = NULL, * dResult = NULL, * hResult = NULL;

	CudaCheckError(cudaHostAlloc(&hA, MATRIX_VECTOR_SIZE * sizeof(int), 0));
	CudaCheckError(cudaHostAlloc(&hB, MATRIX_VECTOR_SIZE * sizeof(int), 0));
	CudaCheckError(cudaHostAlloc(&hResult, MATRIX_VECTOR_SIZE * sizeof(int), 0));
	
	CudaCheckError(cudaMalloc(&dA, MATRIX_VECTOR_SIZE * sizeof(int)));
	CudaCheckError(cudaMalloc(&dB, MATRIX_VECTOR_SIZE * sizeof(int)));
	CudaCheckError(cudaMalloc(&dResult, MATRIX_VECTOR_SIZE * sizeof(int)));

	fillMatrix(hA, MATRIX_VECTOR_SIZE);
	fillMatrix(hB, MATRIX_VECTOR_SIZE);

	CudaCheckError(cudaMemcpy(dA, hA, MATRIX_VECTOR_SIZE * sizeof(int), cudaMemcpyHostToDevice));
	CudaCheckError(cudaMemcpy(dB, hB, MATRIX_VECTOR_SIZE * sizeof(int), cudaMemcpyHostToDevice));

	dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
	dim3 blocksPerGrid(BLOCKS_X, BLOCKS_Y);
	sum<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, M, N, dResult);
	CudaCheckError(cudaGetLastError());
	CudaCheckError(cudaDeviceSynchronize());

	CudaCheckError(cudaMemcpy(hResult, dResult, MATRIX_VECTOR_SIZE * sizeof(int), cudaMemcpyDeviceToHost));
	DEBUG_MATRIX_PRINT(hResult, M, N);
	puts(checkResult(hA, hB, hResult, M, N) ? "Result ok" : "Result not ok");
	
	success = true;
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

	if (!success)
		printf("Cuda error: %s", cudaGetErrorString(cudaGetLastError()));

	return 0;
}