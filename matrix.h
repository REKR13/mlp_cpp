#include <vector>
#include <stdexcept>
#include <iostream>

class Matrix {
    private:
        int rows, cols;
        std::vector<double> data;
    
    public:
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
                    std::cout<<(*this)(i,j);
                }
                std::cout<<"\n";
            }
        }

};