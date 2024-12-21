#include <iostream>
#include "matrix.h"

int main() {
    std::cout << "Hello World" << std::endl;

    Matrix mat(3, 3, 1.0);

    std::cout << "Matrix created" << std::endl;
    mat.print();

    return 0;
}
