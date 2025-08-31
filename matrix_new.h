#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <stdexcept>
#include <iostream>

class Matrix {
private:
    int rows, cols;
    std::vector<double> data;

public:
    Matrix();
    Matrix(int rows, int cols, double init_value = 0);
    
    double& operator()(int row, int col);
    double operator()(int row, int col) const;
    
    Matrix& array_set(std::vector<double> array);
    void print() const;
    void shape() const;
    
    Matrix T() const;
    Matrix reshape(int new_rows, int new_cols) const;
    Matrix matmul(const Matrix& other) const;
    Matrix hadamard_product(const Matrix& other) const;
    
    int get_cols();
    int get_rows();
    double get_single_value() const;
    
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(double c) const;
    Matrix operator/(double c) const;
    Matrix operator*(const Matrix& other) const;
};

#endif