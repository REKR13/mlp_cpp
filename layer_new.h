#ifndef LAYER_H
#define LAYER_H

#include "matrix_new.h"
#include "activation.h"
#include <cmath>
#include <stdexcept>
#include <random>

class Layer {
private:
    Matrix weights;
    Matrix biases;
    Activation activation;
    int input_size, output_size;
    
    Matrix applyActivation(const Matrix& z, bool grad = false) const;
    double activator(double x, bool grad) const;
    double relu(double x) const;
    double relu_grad(double x) const;
    double sigmoid(double x) const;
    double sigmoid_grad(double x) const;

public:
    Layer(int input_size, int output_size);
    
    int get_input_size() const;
    int get_output_size() const;
    Matrix get_weights() const;
    Matrix get_biases() const;
    std::string get_activation() const;
    
    Matrix forward(const Matrix& input);
    void set_activation(const Activation& input);
    void set_weights(const Matrix& input);
    void set_biases(const Matrix& input);
    Matrix activation_grad(const Matrix& layer_output) const;
};

#endif