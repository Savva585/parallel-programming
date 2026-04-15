#include <iostream>
#include <fstream>
#include <chrono>
#include <string>

#include "matrix_math.h"
using namespace std;
int main() {
    setlocale(LC_ALL, "Russian"); 
    string fileA, fileB, fileOut;
    const string defA = "dataA.txt";
    const string defB = "dataB.txt";
    const string defOut = "output.txt";


    cout << "Enter matrix A file name [" << defA << "]: ";
    getline(cin, fileA);
    if (fileA.empty()) fileA = defA;

    cout << "Enter matrix B file name [" << defB << "]: ";
    getline(cin, fileB);
    if (fileB.empty()) fileB = defB;

    cout << "Enter output file name [" << defOut << "]: ";
    getline(cin, fileOut);
    if (fileOut.empty()) fileOut = defOut;

    ifstream testA(fileA);
    if (!testA.is_open()) {
        cerr << "Error: cannot open file '" << fileA << "'\n";
        return 1;
    }
    testA.close();

    ifstream testB(fileB);
    if (!testB.is_open()) {
        cerr << "Error: cannot open file '" << fileB << "'\n";
        return 1;
    }
    testB.close();

    try {
        cout << "\nReading first matrix from " << fileA << "\n";
        Matr A = loadMatrix(fileA);

        cout << "Reading second matrix from " << fileB << "\n";
        Matr B = loadMatrix(fileB);

        if (A.getRows() != A.getCols() || B.getRows() != B.getCols()) {
            cerr << "Error: both matrices must be square.\n";
            return 1;
        }
        if (A.getRows() != B.getRows()) {
            cerr << "Error: dimension mismatch. A is " << A.getRows()
                      << "x" << A.getCols() << ", B is " << B.getRows()
                      << "x" << B.getCols() << "\n";
            return 1;
        }

        size_t n = A.getRows();
        cout << "Matrix size: " << n << " x " << n << "\n";
        size_t totalOps = n * n * n;
        cout << "Approximate operations count: " << totalOps << "+-1\n";

        auto start = chrono::high_resolution_clock::now();
        Matr C = mul(A, B);
        auto end = chrono::high_resolution_clock::now();

        auto elapsedMs = chrono::duration_cast<chrono::milliseconds>(end - start);
        auto elapsedUs = chrono::duration_cast<chrono::microseconds>(end - start);
        cout << "Multiplication took: " << elapsedMs.count() << " ms ("
                  << elapsedUs.count() << " µs)\n";

        cout << "Saving result to " << fileOut << " ...\n";
        saveMatrix(fileOut, C);

        cout << "Done.\n";
    } catch (const std::exception& ex) {
        cerr << "Exception: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}