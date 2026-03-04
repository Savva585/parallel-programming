#include <iostream>
#include <chrono>
#include <string>
#include "matrix.h"
#include <windows.h>

using namespace std;

int main() {
    setlocale(LC_ALL, ".UTF-8");
    try {
        string fileA = "matrix_500.txt";
        string fileB = "matrix_500.txt";
        string fileOut = "result.txt";

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
        size_t operations = n * n * n;
        cout << "Max operations: " << operations << "\n";

        auto start = chrono::high_resolution_clock::now();
        Matrix C = A * B;
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        cout << "Multiplication time: " << duration.count() << " microseconds\n";

        cout << "Result written to " << fileOut << "...\n";
        writeMatrixToFile(fileOut, C);

        cout << "Program finished.\n";
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    return 0;
}