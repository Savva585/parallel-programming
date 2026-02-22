#include <iostream>
#include <vector>

class Matrix {
public:
  Matrix(size_t rows, size_t cols);

  size_t rows() const;
  size_t cols() const;

  float operator()(size_t row, size_t col) const;
  float& operator()(size_t row, size_t col);

private:
  std::vector<float> _data;
  size_t _rows;
  size_t _cols;
};
Matrix& operator+=(Matrix& lhs, const Matrix& rhs);
Matrix operator+(Matrix lhs, const Matrix& rhs);

Matrix& operator*=(Matrix& lhs, float rhs);
Matrix operator*(Matrix lhs, float rhs);
Matrix operator*(float lhs, Matrix rhs);
Matrix operator*(const Matrix& lhs, const Matrix& rhs);
Matrix readMatrixFromFile(const std::string& filename);
void writeMatrixToFile(const std::string& filename, const Matrix& m);
std::ostream& operator<<(std::ostream& os, const Matrix& m);
