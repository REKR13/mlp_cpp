#ifndef LOSS_H
#define LOSS_H

#include "matrix.h"
#include <cmath>


class Loss {
    public:
    virtual double compute(const Matrix& predicted, const Matrix& target) const = 0;
    virtual Matrix gradient(const Matrix& predicted, const Matrix& target) const = 0;
    virtual ~Loss() = default;
};

class MeanSquaredError : public Loss {
    public:
    double compute(const Matrix& predicted, const Matrix& target) const override {
        // can use the matrix form (1/n)e^Te
        Matrix e = predicted - target;
        return ((e.T() * e) / e.get_rows()).get_single_value()/2.0;
    }
    Matrix gradient(const Matrix& predicted, const Matrix& target) const override {
        Matrix e = predicted - target;
        return e/e.get_rows();
    }
};

class BinaryCrossEntropyLoss : public Loss { // this loss function is valid when using sigmoid loss s.t. y_hat \in {0,1}
    public:
    double compute(const Matrix& predicted, const Matrix& target) const override {
        double y_hat = predicted.get_single_value();
        double y = target.get_single_value();

        return -1.0*(y*std::log(y_hat) + (1-y)*std::log(1-y_hat));
    }
    Matrix gradient(const Matrix& predicted, const Matrix& target) const override {
        double y_hat = predicted.get_single_value();
        double y = target.get_single_value();

        Matrix e = Matrix {1,1,0}.array_set({(y_hat-y)/(y_hat*(1-y_hat))});
        return e;
    }
};

#endif
