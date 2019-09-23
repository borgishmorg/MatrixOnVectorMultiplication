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
	__shared__ int xx[8192];
	
	for (int i = threadIdx.x; i < m; i += blockDim.x)
		xx[i] = x[i];

	__syncthreads();

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		res[i] = 0;
		for (int j = 0; j < m; j++)
			res[i] += a[i * m + j] * xx[j];
	}
}

__global__ void gpu_coalescing_shared_multipication(int n, int m, int* a, int* x, int* res) {
	__shared__ int xx[8192];

	for (int i = threadIdx.x; i < m; i += blockDim.x)
		xx[i] = x[i];

	__syncthreads();

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		res[i] = 0;
		for (int j = 0; j < m; j++)
			res[i] += a[i + j * n] * xx[j];
	}
}