#include "multiplication.h"
#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <chrono>

#define BLOCK_SIZE 512

int* generate(int n) {
	int* res = new int[n];
	for (int i = 0; i < n; i++)
		res[i] = rand() % 100;
	return res;
}

bool check(int n, int* a, int* b) {
	for (int i = 0; i < n; i++)
		if (a[i] != b[i])
			return false;
	return true;
}

int main()
{
	freopen("log.txt", "w", stdout);
	int n = 8192, m = 8192;
	int* a = generate(n * m);
	int* a_t = new int[n * m];
	int* x = generate(m);
	int* res = new int[n];
	int* res_cpu = new int[n];

	{
		std::cout << "\nCPU test\n";
		auto t0 = std::chrono::steady_clock::now();
		cpu_multipication(n, m, a, x, res_cpu);
		auto t1 = std::chrono::steady_clock::now();
		std::cout << "Time: " << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()<< " us\n";
		std::cout.flush();
	}


	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++)
			a_t[j * n + i] = a[i * m + j];

	int* dev_a, *dev_a_t, * dev_x, * dev_res;

	cudaMalloc(&dev_a, n * m * sizeof(int));
	cudaMalloc(&dev_a_t, n * m * sizeof(int));
	cudaMalloc(&dev_x, m * sizeof(int));
	cudaMalloc(&dev_res, n * sizeof(int));

	cudaMemcpy(dev_a, a, n * m * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_a_t, a_t, n * m * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_x, x, m * sizeof(int), cudaMemcpyHostToDevice);
	
	
	{
		std::cout << "\nGPU test\n";
		cudaDeviceSynchronize();
		auto t0 = std::chrono::steady_clock::now();
		gpu_multipication<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >>>(n, m, dev_a, dev_x, dev_res);
		cudaDeviceSynchronize();
		auto t1 = std::chrono::steady_clock::now();
		cudaMemcpy(res, dev_res, n * sizeof(int), cudaMemcpyDeviceToHost);
		std::cout << "Time: " << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() << " us\n";
		std::cout << "The answer is " << (check(n, res_cpu, res) ? "" : "NOT ") << "correct\n";
		std::cout.flush();
	}

	{
		std::cout << "\nGPU test (with coalescing)\n";
		cudaDeviceSynchronize();
		auto t0 = std::chrono::steady_clock::now();
		gpu_coalescing_multipication<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >>>(n, m, dev_a_t, dev_x, dev_res);
		cudaDeviceSynchronize();
		auto t1 = std::chrono::steady_clock::now();
		cudaMemcpy(res, dev_res, n * sizeof(int), cudaMemcpyDeviceToHost);
		std::cout << "Time: " << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() << " us\n";
		std::cout << "The answer is " << (check(n, res_cpu, res) ? "" : "NOT ") << "correct\n";
		std::cout.flush();
	}

	{
		std::cout << "\nGPU test (with shared memory)\n";
		cudaDeviceSynchronize();
		auto t0 = std::chrono::steady_clock::now();
		gpu_shared_multipication<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >>>(n, m, dev_a, dev_x, dev_res);
		cudaDeviceSynchronize();
		auto t1 = std::chrono::steady_clock::now();
		cudaMemcpy(res, dev_res, n * sizeof(int), cudaMemcpyDeviceToHost);
		std::cout << "Time: " << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() << " us\n";
		std::cout << "The answer is " << (check(n, res_cpu, res) ? "" : "NOT ") << "correct\n";
		std::cout.flush();
	}

	{
		std::cout << "\nGPU test (with coalescing and shared memory)\n";
		cudaDeviceSynchronize();
		auto t0 = std::chrono::steady_clock::now();
		gpu_coalescing_shared_multipication <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >>>(n, m, dev_a_t, dev_x, dev_res);
		cudaDeviceSynchronize();
		auto t1 = std::chrono::steady_clock::now();
		cudaMemcpy(res, dev_res, n * sizeof(int), cudaMemcpyDeviceToHost);
		std::cout << "Time: " << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() << " us\n";
		std::cout << "The answer is " << (check(n, res_cpu, res) ? "" : "NOT ") << "correct\n";
		std::cout.flush();
	}
	
	cudaFree(dev_a);
	cudaFree(dev_a_t);
	cudaFree(dev_x);
	cudaFree(dev_res);

	delete a;
	delete a_t;
	delete x;
	delete res;
	delete res_cpu;
	return 0;
}