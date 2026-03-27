#include <omp.h>
#include <iostream>
#include <chrono>
#include <string>
#include "matrix.h"

using namespace std;

int main(int argc, char* argv[]) {
    setlocale(LC_ALL, ".UTF-8");

    if (argc != 5) {
        cerr << "Usage: " << argv[0] << " <fileA> <fileB> <fileOut> <num_threads>\n";
        return 1;
    }

    string fileA = argv[1];
    string fileB = argv[2];
    string fileOut = argv[3];
    int num_threads = atoi(argv[4]);

    omp_set_num_threads(num_threads);

    try {
        cout << "Reading matrix A from " << fileA << "...\n";
        Matrix A = readMatrixFromFile(fileA);
        cout << "Reading matrix B from " << fileB << "...\n";
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
        cout << "Matrix size: " << n << " x " << n << "\n";
        cout << "Number of threads: " << num_threads << "\n";

        auto start = chrono::high_resolution_clock::now();
        Matrix C = A * B;
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        cout << "Multiplication time: " << duration.count() << " microseconds\n";

        writeMatrixToFile(fileOut, C);
        cout << "Result written to " << fileOut << "\n";
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    return 0;
}