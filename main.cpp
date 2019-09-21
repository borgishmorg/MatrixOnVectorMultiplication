#include "kernel.h"
#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <chrono>

int* generate(int n) {
	int* res = new int[n];
	for (int i = 0; i < n; i++)
		res[i] = rand() % 100;
	return res;
}

int main()
{
	int n = 1000, m = 1000;
	int* a = generate(n * m);
	int* x = generate(m);
	int* res = new int[m];

	int* dev_a, * dev_x, * dev_res;

	cudaMalloc(&dev_a, n * m * sizeof(int));
	cudaMalloc(&dev_x, m * sizeof(int));
	cudaMalloc(&dev_res, n * sizeof(int));

	cudaMemcpy(dev_a, a, n * m * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_x, x, m * sizeof(int), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();

	auto t0 = std::chrono::steady_clock::now();
	gpu_multiply<<<n, 1 >>>(n, m, dev_a, dev_x, dev_res);
	cudaDeviceSynchronize();
	auto t1 = std::chrono::steady_clock::now();

	std::cout << "Time: " << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
		<< " us\n";

	cudaMemcpy(res, dev_res, n * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(dev_a);
	cudaFree(dev_x);
	cudaFree(dev_res);

	delete a;
	delete x;
	delete res;
	return 0;
}