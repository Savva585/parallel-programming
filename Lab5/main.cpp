#include <mpi.h>
#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include "matrix.h"

using namespace std;

int main(int argc, char* argv[]) {
    
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc != 5) {
        if (rank == 0) {
            cerr << "Usage: " << argv[0] << " <fileA> <fileB> <fileOut> <num_procs>\n";
            cerr << "Note: num_procs is ignored, use mpirun -np X\n";
        }
        MPI_Finalize();
        return 1;
    }
    
    string fileA = argv[1];
    string fileB = argv[2];
    string fileOut = argv[3];
    
    Matrix A(0,0), B(0,0);
    size_t N = 0;
    
    if (rank == 0) {
        try {
            cout << "Reading matrix A from " << fileA << "...\n";
            A = readMatrixFromFile(fileA);
            cout << "Reading matrix B from " << fileB << "...\n";
            B = readMatrixFromFile(fileB);
            
            if (A.rows() != A.cols() || B.rows() != B.cols()) {
                cerr << "Matrices are not square\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            if (A.rows() != B.rows()) {
                cerr << "Size mismatch\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            N = A.rows();
            cout << "Matrix size: " << N << " x " << N << "\n";
            cout << "Number of MPI processes: " << size << "\n";
        } catch (const exception& e) {
            cerr << "Error: " << e.what() << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    
    MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    
    int rows_per_proc = N / size;
    int remainder = N % size;
    int my_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    
    vector<float> A_local(my_rows * N);

    vector<float> B_full(N * N);
    
    if (rank == 0) {
        vector<int> sendcounts(size);
        vector<int> displs(size);
        int offset = 0;
        for (int i = 0; i < size; i++) {
            int rows = rows_per_proc + (i < remainder ? 1 : 0);
            sendcounts[i] = rows * N;
            displs[i] = offset;
            offset += sendcounts[i];
        }

        vector<float> A_full(N * N);
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < N; j++) {
                A_full[i * N + j] = A(i, j);
            }
        }
        
        MPI_Scatterv(A_full.data(), sendcounts.data(), displs.data(), MPI_FLOAT,
                     A_local.data(), my_rows * N, MPI_FLOAT,
                     0, MPI_COMM_WORLD);
        
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < N; j++) {
                B_full[i * N + j] = B(i, j);
            }
        }
    } else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_FLOAT,
                     A_local.data(), my_rows * N, MPI_FLOAT,
                     0, MPI_COMM_WORLD);
    }
    
    MPI_Bcast(B_full.data(), N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    vector<float> C_local(my_rows * N, 0.0f);
    
    auto start = chrono::high_resolution_clock::now();
    
    const int BLOCK_SIZE = 64;
    for (int i = 0; i < my_rows; i++) {
        for (int j = 0; j < (int)N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < (int)N; k++) {
                sum += A_local[i * N + k] * B_full[k * N + j];
            }
            C_local[i * N + j] = sum;
        }
    }
    
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    
    if (rank == 0) {
        cout << "Multiplication time (parallel): " << duration.count() << " microseconds\n";
    }
    
    Matrix C(N, N);
    
    if (rank == 0) {
        vector<int> recvcounts(size);
        vector<int> displs(size);
        int offset = 0;
        for (int i = 0; i < size; i++) {
            int rows = rows_per_proc + (i < remainder ? 1 : 0);
            recvcounts[i] = rows * N;
            displs[i] = offset;
            offset += recvcounts[i];
        }
        
        vector<float> C_full(N * N);
        MPI_Gatherv(C_local.data(), my_rows * N, MPI_FLOAT,
                    C_full.data(), recvcounts.data(), displs.data(), MPI_FLOAT,
                    0, MPI_COMM_WORLD);
        
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < N; j++) {
                C(i, j) = C_full[i * N + j];
            }
        }
        
        try {
            writeMatrixToFile(fileOut, C);
            cout << "Result written to " << fileOut << "\n";
        } catch (const exception& e) {
            cerr << "Error writing file: " << e.what() << endl;
        }
    } else {
        MPI_Gatherv(C_local.data(), my_rows * N, MPI_FLOAT,
                    nullptr, nullptr, nullptr, MPI_FLOAT,
                    0, MPI_COMM_WORLD);
    }
    std::cout << "Process " << rank << " finished" << std::endl;
std::cout.flush();
    MPI_Finalize();
    return 0;
}