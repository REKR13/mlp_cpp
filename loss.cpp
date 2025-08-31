#include "loss.h"

double MeanSquaredError::compute(const Matrix& predicted, const Matrix& target) const {
    Matrix e = predicted - target;
    return ((e.T() * e) / e.get_rows()).get_single_value() / 2.0;
}

Matrix MeanSquaredError::gradient(const Matrix& predicted, const Matrix& target) const {
    Matrix e = predicted - target;
    return e / e.get_rows();
}

double BinaryCrossEntropyLoss::compute(const Matrix& predicted, const Matrix& target) const {
    double y_hat = predicted.get_single_value();
    double y = target.get_single_value();
    
    y_hat = std::max(1e-15, std::min(1.0 - 1e-15, y_hat));
    
    return -1.0 * (y * std::log(y_hat) + (1 - y) * std::log(1 - y_hat));
}

Matrix BinaryCrossEntropyLoss::gradient(const Matrix& predicted, const Matrix& target) const {
    double y_hat = predicted.get_single_value();
    double y = target.get_single_value();
    
    y_hat = std::max(1e-15, std::min(1.0 - 1e-15, y_hat));
    
    Matrix e = Matrix(1, 1, (y_hat - y) / (y_hat * (1 - y_hat)));
    return e;
}