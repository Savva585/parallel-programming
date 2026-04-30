#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <string>

void matmulCUDA(const float* A, const float* B, float* C, 
                int N, int M, int K, 
                dim3 blockSize, dim3 gridSize);
void printDeviceInfo();
std::string getDeviceName();

void benchmarkCUDA(const float* A, const float* B, float* C, 
                   int N, int M, int K,
                   int blockSizeX, int blockSizeY,
                   double& timeMs);