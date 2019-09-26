#include "multiplication.h"

void cpu_multipication(int n, int m, int* a, int* x, int* res) {
	for (int i = 0; i < n; i++) {
		res[i] = 0;
		for (int j = 0; j < m; j++)
			res[i] += a[i * m + j] * x[j];
	}
}

__global__ void gpu_multipication(int n, int m, int* a, int* x, int* res) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		res[i] = 0;
		for (int j = 0; j < m; j++)
			res[i] += a[i * m + j] * x[j];
	}
}

__global__ void gpu_coalescing_multipication(int n, int m, int* a, int* x, int* res) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		res[i] = 0;
		for (int j = 0; j < m; j++)
			res[i] += a[i + j*n] * x[j];
	}
}

__global__ void gpu_shared_multipication(int n, int m, int* a, int* x, int* res) {
	__shared__ int xx[BLOCK_SIZE];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < n) res[idx] = 0;

	for (int i = 0; i < m; i += blockDim.x) {
		if(i + threadIdx.x < m)
			xx[threadIdx.x] = x[i + threadIdx.x];
		__syncthreads();

		if (idx < n) 
			for (int j = 0; j < blockDim.x && i + j < m; j++)
					res[idx] += a[idx * m + i + j] * xx[j];

		__syncthreads();
	}
}

__global__ void gpu_coalescing_shared_multipication(int n, int m, int* a, int* x, int* res) {
	__shared__ int xx[BLOCK_SIZE];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < n) res[idx] = 0;

	for (int i = 0; i < m; i += blockDim.x) {
		if (i + threadIdx.x < m)
			xx[threadIdx.x] = x[i + threadIdx.x];
		__syncthreads();

		if (idx < n)
			for (int j = 0; j < blockDim.x && i + j < m; j++)
					res[idx] += a[idx + (i + j)*n] * xx[j];

		__syncthreads();
	}
}