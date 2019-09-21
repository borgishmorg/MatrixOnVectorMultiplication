#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

void cpu_multipication(int n, int m, int* a, int* x, int* res);
__global__ void gpu_multipication(int n, int m, int* a, int* x, int* res);
__global__ void gpu_coalescing_multipication(int n, int m, int* a, int* x, int* res);
__global__ void gpu_shared_multipication(int n, int m, int* a, int* x, int* res);
__global__ void gpu_coalescing_shared_multipication(int n, int m, int* a, int* x, int* res);
