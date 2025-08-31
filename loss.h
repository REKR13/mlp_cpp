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
    double compute(const Matrix& predicted, const Matrix& target) const override;
    Matrix gradient(const Matrix& predicted, const Matrix& target) const override;
};

class BinaryCrossEntropyLoss : public Loss {
public:
    double compute(const Matrix& predicted, const Matrix& target) const override;
    Matrix gradient(const Matrix& predicted, const Matrix& target) const override;
};

#endif
