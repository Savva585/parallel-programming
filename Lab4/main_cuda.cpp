#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <cuda_runtime.h>
#include "matrix.h"
#include "matrix_cuda.h"

using namespace std;

Matrix generateRandomMatrix(size_t n) {
    Matrix result(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result(i, j) = static_cast<float>(rand()) / RAND_MAX;
        }
    }
    return result;
}

struct BenchmarkResult {
    int size;
    int blockX;
    int blockY;
    double timeMs;
    double gflops;
};

void runExperiments() {
    vector<int> sizes = {200,400,800,1200,1400,1800,2000};
    
    vector<pair<int, int>> blockConfigs = {
        {8, 8}, {16, 16}, {32, 32}, {16, 8}, {8, 16}, {32, 16}
    };
    
    vector<BenchmarkResult> results;
    
    cout << "\n=== CUDA Matrix Multiplication Experiments ===\n";
    printDeviceInfo();
    cout << "\nStarting benchmarks...\n\n";
    
    for (int size : sizes) {
        cout << "Testing size: " << size << "x" << size << "\n";
        
        Matrix A = generateRandomMatrix(size);
        Matrix B = generateRandomMatrix(size);
        Matrix C_cpu(size, size);
        
        auto start_cpu = chrono::high_resolution_clock::now();
        Matrix C_cpu_result = A * B;
        auto end_cpu = chrono::high_resolution_clock::now();
        double cpuTimeMs = chrono::duration_cast<chrono::microseconds>(end_cpu - start_cpu).count() / 1000.0;
        
        vector<float> flatA(size * size);
        vector<float> flatB(size * size);
        vector<float> flatC(size * size);
        
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < size; ++j) {
                flatA[i * size + j] = A(i, j);
                flatB[i * size + j] = B(i, j);
            }
        }
        
        for (auto& config : blockConfigs) {
            double timeMs;
            try {
                benchmarkCUDA(flatA.data(), flatB.data(), flatC.data(),
                             size, size, size,
                             config.first, config.second, timeMs);
                
                double gflops = (2.0 * size * size * size) / (timeMs * 1e6);
                
                results.push_back({size, config.first, config.second, timeMs, gflops});
                
                cout << "  Block " << config.first << "x" << config.second 
                     << ": " << fixed << setprecision(2) << timeMs << " ms"
                     << " (" << setprecision(2) << gflops << " GFLOPS)\n";
            }
            catch (const exception& e) {
                cerr << "  Block " << config.first << "x" << config.second 
                     << ": FAILED - " << e.what() << "\n";
            }
        }
        
        cout << "  CPU Reference: " << setprecision(2) << cpuTimeMs << " ms\n\n";
    }
    
    ofstream csv("cuda_results.csv");
    csv << "Size,BlockX,BlockY,TimeMs,GFLOPS\n";
    for (const auto& r : results) {
        csv << r.size << "," << r.blockX << "," << r.blockY << ","
            << r.timeMs << "," << r.gflops << "\n";
    }
    csv.close();
    
    ofstream best("cuda_best_results.csv");
    best << "Size,BestBlockX,BestBlockY,BestTimeMs,BestGFLOPS,CPUTimeMs,Speedup\n";
    
    for (int size : sizes) {
        double bestTime = 1e9;
        pair<int, int> bestConfig;
        double bestGflops = 0;
        
        for (const auto& r : results) {
            if (r.size == size && r.timeMs < bestTime) {
                bestTime = r.timeMs;
                bestConfig = {r.blockX, r.blockY};
                bestGflops = r.gflops;
            }
        }
        
        double cpuEstimate = 2.0 * size * size * size / 1e9; 
        
        best << size << "," << bestConfig.first << "x" << bestConfig.second << ","
             << bestTime << "," << bestGflops << "," << cpuEstimate << ","
             << (cpuEstimate / bestTime) << "\n";
    }
    best.close();
    
    cout << "\n=== Results saved to cuda_results.csv and cuda_best_results.csv ===\n";
}

int main(int argc, char* argv[]) {
    setlocale(LC_ALL, ".UTF-8");
    
    try {
        runExperiments();
        
        if (argc == 5) {
            string fileA = argv[1];
            string fileB = argv[2];
            string fileOut = argv[3];
            
            cout << "\n=== Single multiplication mode ===\n";
            Matrix A = readMatrixFromFile(fileA);
            Matrix B = readMatrixFromFile(fileB);
            
            if (A.rows() != A.cols() || B.rows() != B.cols()) {
                cerr << "Matrices are not square\n";
                return 1;
            }
            if (A.rows() != B.rows()) {
                cerr << "Size mismatch\n";
                return 1;
            }
            
            size_t n = A.rows();
            
            vector<float> flatA(n * n);
            vector<float> flatB(n * n);
            vector<float> flatC(n * n);
            
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    flatA[i * n + j] = A(i, j);
                    flatB[i * n + j] = B(i, j);
                }
            }
            
            double timeMs;
            benchmarkCUDA(flatA.data(), flatB.data(), flatC.data(),
                         n, n, n, 16, 16, timeMs);
            
            cout << "CUDA multiplication time: " << fixed << setprecision(2) << timeMs << " ms\n";
            
            Matrix C(n, n);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    C(i, j) = flatC[i * n + j];
                }
            }
            writeMatrixToFile(fileOut, C);
            cout << "Result written to " << fileOut << "\n";
        }
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}