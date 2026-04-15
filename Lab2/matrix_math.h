#ifndef MATRIX_MATH_H
#define MATRIX_MATH_H

#include <iostream>
#include <vector>

class Matr {
public:
    Matr(size_t r, size_t c);
    size_t getRows() const;
    size_t getCols() const;
    float get(size_t i, size_t j) const;
    void set(size_t i, size_t j, float val);
private:
    std::vector<float> data_;
    size_t rows_;
    size_t cols_;
};

Matr mul(const Matr& A, const Matr& B);
Matr loadMatrix(const std::string& fname);
void saveMatrix(const std::string& fname, const Matr& m);

#endif