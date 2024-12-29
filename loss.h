#ifndef LOSS_H
#define LOSS_H

#include "matrix.h"

double mean_squared_error(Matrix& predicted, Matrix& actual) {
    // can use the matrix form (1/n)e^Te
    Matrix e = actual - predicted;
    return ((e.T() * e) / e.get_rows()).get_single_value();
}

#endif
