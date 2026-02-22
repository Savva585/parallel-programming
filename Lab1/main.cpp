#include <iostream>
#include <chrono>
#include <string>
#include "matrix.h"
#include<windows.h>


using namespace std;

int main() {
    setlocale(LC_ALL, "Russian");
    try {
        string fileA = "matrixA.txt";
        string fileB =  "matrixB.txt";
        string fileOut = "result.txt";

        cout << "Чтение из матрицы А " << fileA << "...\n";
        Matrix A = readMatrixFromFile(fileA);
        cout << "Чтение из матрицы В " << fileB << "...\n";
        Matrix B = readMatrixFromFile(fileB);

        if (A.rows() != A.cols() || B.rows() != B.cols()) {
            cerr << "Матрицы не квадратные\n";
            return 1;
        }
        if (A.rows() != B.rows()) {
            cerr << "Несовпадение по размеру\n";
            return 1;
        }

        size_t n = A.rows();
        cout << "Размер Матриц " << n << " x " << n << "\n";
        size_t operations = n * n * n; 
        cout << "Макс число операций: " << operations << "\n";

        auto start = chrono::high_resolution_clock::now();
        Matrix C = A * B;
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        cout << "Время умножения: " << duration.count() << " мкс\n";

        cout << "Результат умножения находится в " << fileOut << "...\n";
        writeMatrixToFile(fileOut, C);

        cout << "Программа завершена.\n";
    }
    catch (const exception& e) {
        cerr << "Ошибка: " << e.what() << endl;
        return 1;
    }
    return 0;
}