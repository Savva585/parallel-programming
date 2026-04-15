#include "matrix_math.h"
#include <fstream>
#include <stdexcept>

using namespace std;

Matr::Matr(size_t r, size_t c) : data_(r * c, 0.0f), rows_(r), cols_(c) {}

size_t Matr::getRows() const { return rows_; }
size_t Matr::getCols() const { return cols_; }

float Matr::get(size_t i, size_t j) const {
    return data_[i * cols_ + j];
}

void Matr::set(size_t i, size_t j, float val) {
    data_[i * cols_ + j] = val;
}

Matr mul(const Matr& A, const Matr& B) {
    if (A.getCols() != B.getRows()) {
        throw invalid_argument("Dimension mismatch in multiplication");
    }
    Matr result(A.getRows(), B.getCols());
    for (size_t i = 0; i < A.getRows(); ++i) {
        for (size_t k = 0; k < A.getCols(); ++k) {
            float aik = A.get(i, k);
            for (size_t j = 0; j < B.getCols(); ++j) {
                result.set(i, j, result.get(i, j) + aik * B.get(k, j));
            }
        }
    }
    return result;
}

Matr loadMatrix(const string& fname) {
    ifstream file(fname);
    if (!file.is_open()) {
        throw runtime_error("Cannot open file: " + fname);
    }
    size_t n;
    file >> n;
    if (!file || n == 0) {
        throw runtime_error("Invalid matrix size");
    }
    Matr res(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float val;
            if (!(file >> val)) {
                throw runtime_error("Not enough data");
            }
            res.set(i, j, val);
        }
    }
    return res;
}

void saveMatrix(const string& fname, const Matr& m) {
    ofstream file(fname);
    if (!file.is_open()) {
        throw runtime_error("Cannot create file: " + fname);
    }
    file << m.getRows() << '\n';
    for (size_t i = 0; i < m.getRows(); ++i) {
        for (size_t j = 0; j < m.getCols(); ++j) {
            file << m.get(i, j) << ' ';
        }
        file << '\n';
    }
}

