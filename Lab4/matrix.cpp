#include "matrix.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <omp.h>

using namespace std;


Matrix::Matrix(size_t rows, size_t cols) :
  _data(rows * cols),
  _rows(rows),
  _cols(cols)
{
}

size_t Matrix::rows() const {
  return _rows;
}

size_t Matrix::cols() const {
  return _cols;
}

float Matrix::operator()(size_t row, size_t col) const {
  return _data[row * _cols + col];
}

float& Matrix::operator()(size_t row, size_t col) {
  return _data[row * _cols + col];
}

Matrix& operator+=(Matrix& lhs, const Matrix& rhs) {
  const size_t rows = lhs.rows();
  const size_t cols = lhs.cols();

  if (rows != rhs.rows() || cols != rhs.cols()) {
    cerr << "Incompatible matrix sizes in operator+=.\n";
    exit(EXIT_FAILURE);
  }

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      lhs(i, j) += rhs(i, j);
    }
  }

  return lhs;
}

Matrix operator+(Matrix lhs, const Matrix& rhs) {
  return lhs += rhs;
}

Matrix& operator*=(Matrix& lhs, float rhs) {
  const size_t rows = lhs.rows();
  const size_t cols = lhs.cols();

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      lhs(i, j) *= rhs;
    }
  }

  return lhs;
}

Matrix operator*(Matrix lhs, float rhs) {
  return lhs *= rhs;
}

Matrix operator*(float lhs, Matrix rhs) {
  return rhs *= lhs;
}

ostream& operator<<(ostream& os, const Matrix& m) {
  const size_t rows = m.rows();
  const size_t cols = m.cols();

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      os << m(i, j) << ' ';
    }
    os << '\n';
  }
  os << '\n';

  return os;
}


Matrix operator*(const Matrix& lhs, const Matrix& rhs) {
    if (lhs.cols() != rhs.rows()) {
        throw std::invalid_argument("Matrix dimensions mismatch for multiplication");
    }
    Matrix result(lhs.rows(), rhs.cols());
    
    #pragma omp parallel for
    for (size_t i = 0; i < lhs.rows(); ++i) {
        for (size_t j = 0; j < rhs.cols(); ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < lhs.cols(); ++k) {
                sum += lhs(i, k) * rhs(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}

Matrix readMatrixFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    size_t n;
    file >> n;                  
    if (!file || n == 0) {
        throw std::runtime_error("Invalid matrix size in file");
    }

    Matrix result(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float value;
            if (!(file >> value)) {
                throw std::runtime_error("Not enough data in file");
            }
            result(i, j) = value;
        }
    }
    return result;
}

void writeMatrixToFile(const string& filename, const Matrix& m) {
    ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    const size_t rows = m.rows();
    const size_t cols = m.cols();
    file << rows << '\n';
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            file << m(i, j) << ' ';
        }
        file << '\n';
    }
}