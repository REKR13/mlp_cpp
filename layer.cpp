#include "layer.h"
#include "activation.h"

Layer::Layer(int input_size, int output_size)
    : activation(Activation::NONE), input_size(input_size), output_size(output_size) {
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dis(0.0, sqrt(2.0 / input_size));
    
    weights = Matrix(output_size, input_size);
    biases = Matrix(output_size, 1, 0.0);
    
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size; j++) {
            weights(i, j) = dis(gen);
        }
    }
}

int Layer::get_input_size() const {
    return input_size;
}

int Layer::get_output_size() const {
    return output_size;
}

Matrix Layer::get_weights() const {
    return weights;
}

Matrix Layer::get_biases() const {
    return biases;
}

std::string Layer::get_activation() const {
    switch (activation) {
        case Activation::RELU:
            return "Activation: relu";
        case Activation::SIGMOID:
            return "Activation: sigmoid"; 
        case Activation::NONE:
            return "Activation: none";
        default:
            throw std::invalid_argument("Invalid Activation");
    }
}

Matrix Layer::forward(const Matrix& input) {
    Matrix z = weights * input + biases;
    return applyActivation(z);
}

void Layer::set_activation(const Activation& input) {
    activation = input;
}

void Layer::set_weights(const Matrix& input) {
    weights = input;
}

void Layer::set_biases(const Matrix& input) {
    biases = input;
}

Matrix Layer::activation_grad(const Matrix& layer_output) const {
    return applyActivation(layer_output, true);
}

Matrix Layer::applyActivation(const Matrix& z, bool grad) const {
    Matrix result = z;
    for (int i = 0; i < result.get_rows(); i++) {
        for (int j = 0; j < result.get_cols(); j++) {
            result(i,j) = activator(result(i,j), grad);
        }
    }
    return result;
}

double Layer::activator(double x, bool grad) const {
    if (activation == Activation::RELU) {
        return grad ? relu_grad(x) : relu(x);
    }
    else if (activation == Activation::SIGMOID) {
        return grad ? sigmoid_grad(x) : sigmoid(x);
    }
    return x;
}

double Layer::relu(double x) const {
    return std::max(0.0, x);
}

double Layer::relu_grad(double x) const {
    return x > 0 ? 1.0 : 0.0;
}

double Layer::sigmoid(double x) const {
    return 1.0 / (1.0 + std::exp(-x));
}

double Layer::sigmoid_grad(double x) const {
    return x * (1.0 - x);
}
