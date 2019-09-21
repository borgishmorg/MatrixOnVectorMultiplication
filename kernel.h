#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void gpu_multiply(int n, int m, int* a, int* x, int* res);