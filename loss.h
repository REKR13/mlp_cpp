#ifndef LOSS_H
#define LOSS_H

#include "matrix.h"


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
        return ((e.T() * e) / e.get_rows()).get_single_value();
    }
    Matrix gradient(const Matrix& predicted, const Matrix& target) const override {
        Matrix e = predicted - target;
        return e*(2.0/e.get_rows());
    }
};

class BinaryCrossEntropyLoss : public Loss {
    public:
    double compute(Matrix& predicted, Matrix& actual) {
        
    }
    Matrix gradient(Matrix& predicted, Matrix& actual) {
        Matrix e = predicted - actual;
        return (e*2.0/e.get_rows());
    }
};


#endif
