#include "matrix_cuda.h"
#include <cuda_runtime.h>
#include <chrono>
#include <stdexcept>

__global__ void matmulKernelSimple(const float* A, const float* B, float* C,
                                    int N, int M, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < M; ++i) {
            sum += A[row * M + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}

__global__ void matmulKernelShared(const float* A, const float* B, float* C,
                                    int N, int M, int K) {
    __shared__ float sharedA[32][32];
    __shared__ float sharedB[32][32];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (M + blockDim.x - 1) / blockDim.x; ++tile) {
        if (row < N && tile * blockDim.x + tx < M) {
            sharedA[ty][tx] = A[row * M + tile * blockDim.x + tx];
        } else {
            sharedA[ty][tx] = 0.0f;
        }
        
        if (col < K && tile * blockDim.y + ty < M) {
            sharedB[ty][tx] = B[(tile * blockDim.y + ty) * K + col];
        } else {
            sharedB[ty][tx] = 0.0f;
        }
        
        __syncthreads();
            for (int i = 0; i < blockDim.x; ++i) {
            sum += sharedA[ty][i] * sharedB[i][tx];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < K) {
        C[row * K + col] = sum;
    }
}

void printDeviceInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "Found " << deviceCount << " CUDA device(s)\n";
    
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "\nDevice " << i << ": " << prop.name << "\n";
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "  CUDA Cores: " << prop.multiProcessorCount * prop.warpSize << "\n";
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << "\n";
        std::cout << "  Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB\n";
    }
}

std::string getDeviceName() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return std::string(prop.name);
}

void matmulCUDA(const float* A, const float* B, float* C,
                int N, int M, int K,
                dim3 blockSize, dim3 gridSize) {
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * M * sizeof(float));
    cudaMalloc(&d_B, M * K * sizeof(float));
    cudaMalloc(&d_C, N * K * sizeof(float));
    
    cudaMemcpy(d_A, A, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, M * K * sizeof(float), cudaMemcpyHostToDevice);
    
    matmulKernelSimple<<<gridSize, blockSize>>>(d_A, d_B, d_C, N, M, K);
    
    cudaMemcpy(C, d_C, N * K * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void benchmarkCUDA(const float* A, const float* B, float* C,
                   int N, int M, int K,
                   int blockSizeX, int blockSizeY,
                   double& timeMs) {
    dim3 blockSize(blockSizeX, blockSizeY);
    dim3 gridSize((K + blockSize.x - 1) / blockSize.x,
                  (N + blockSize.y - 1) / blockSize.y);
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * M * sizeof(float));
    cudaMalloc(&d_B, M * K * sizeof(float));
    cudaMalloc(&d_C, N * K * sizeof(float));
    
    cudaMemcpy(d_A, A, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, M * K * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    matmulKernelSimple<<<gridSize, blockSize>>>(d_A, d_B, d_C, N, M, K);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    matmulKernelSimple<<<gridSize, blockSize>>>(d_A, d_B, d_C, N, M, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&timeMs, start, stop);
    
    cudaMemcpy(C, d_C, N * K * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}