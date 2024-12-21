#include <iostream>
#include "matrix.h"

int main() {
    std::cout << "Hello World" << std::endl;

    Matrix mat1(3, 3, 1);
    Matrix mat2(5,7,5);

    std::cout << "Matrices created" << std::endl;
    
    Matrix matf = mat1*mat2;
    matf.print();

    return 0;
}
