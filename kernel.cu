#include "kernel.h"

__global__ void gpu_multiply(int n, int m, int* a, int* x, int* res) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		res[i] = 0;
		for (int j = 0; j < m; j++)
			res[i] += a[i * m + j] * x[j];
	}
}