#include <iostream>
#include "matrix.h"

int main() {
    std::cout << "Hello World" << std::endl;

    Matrix mat1(3, 3, 1);
    Matrix mat2(3,8,5);

    std::cout << "Matrices created" << std::endl;
    
    Matrix matf = mat1*mat2;
    matf(0,1) = 8.0;
    matf = matf.T().reshape(12,2);
    matf.print();
    matf.shape();

    return 0;
}
