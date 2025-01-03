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
    Matrix() : rows(0), cols(0), data(0) {}

    Matrix(int rows, int cols, double init_value = 0) 
    : rows(rows), cols(cols), data(rows * cols, init_value) {}

    double& operator()(int row, int col) {
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            throw std::out_of_range("Index out of range");
        }
        return data[row*cols + col];
    }

    double operator()(int row, int col) const {
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            throw std::out_of_range("Index out of range");
        }
        return data[row*cols + col];
    }

    void print() const {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                std::cout<<(*this)(i,j)<< " ";
            }
            std::cout<<"\n";
        }
        std::cout<<"\n";
    }

    void shape() const {
        std::cout<<"("<<rows<<","<<cols<<")"<<"\n";
    }

    Matrix T() const {

        Matrix result(cols, rows);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result(j,i) = (*this)(i,j);
            }
        }
        return result;
    }

    Matrix reshape(int new_rows, int new_cols) const {
        if (new_rows*new_cols != rows*cols) {
            throw std::invalid_argument("Invalid shape");
        }

        Matrix result(new_rows, new_cols);
        int count = 0;

        for (int i = 0; i < new_rows; i++) {
            for (int j = 0; j < new_cols; j++) {
                result(i,j) = data[count];
                count++;
            }
        }
        return result;
    }

    Matrix matmul(const Matrix& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Inner dimensions do not match");
        }
        Matrix result(rows, other.cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                for (int k = 0; k < cols; k++) {
                    result(i,j) += (*this)(i,k) * other(k,j);
                }
            }
        }
        return result;
    }
    //there is a way to optimize this from using data or not creating a new matrix
    Matrix hadamard_product(const Matrix& other) const {
        if (cols != other.cols || rows != other.rows) {
            throw std::invalid_argument("Matrix dimensions must match");
        }
        Matrix result(rows, cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result(i,j) = (*this)(i,j) + other(i,j);
            }
        }
        return result;
    }

    int get_cols() {
        return cols;
    }

    int get_rows() {
        return rows;
    }

    double get_single_value() {
        if (data.size() != 1) {
            throw std::invalid_argument("Not a 1x1 matrix");
        }
        return data[0];
    }

    Matrix operator+(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions do not match for addition");
        }

        Matrix result(rows, cols);

        for (int i = 0; i<rows; i++) {
            for (int j = 0; j < cols; j++) {
                result(i,j) = (*this)(i,j) + other(i,j);
            }
        }
        return result;
    }
    Matrix operator-(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions do not match for subtraction");
        }

        Matrix result(rows, cols);

        for (int i = 0; i<rows; i++) {
            for (int j = 0; j < cols; j++) {
                result(i,j) = (*this)(i,j) - other(i,j);
            }
        }
        return result;
    }

    Matrix operator*(double c) const {
        Matrix result(rows, cols);
        for (int i = 0; i<rows; i++) {
            for (int j = 0; j < cols; j++) {
                result(i,j) = (*this)(i,j) * c;
            }
        }
        return result;
    }

    Matrix operator/(double c) const {
        return (*this)*(1/c);
    }

    Matrix operator*(const Matrix& other) const {
        return (*this).matmul(other);
    }
};

#endif
