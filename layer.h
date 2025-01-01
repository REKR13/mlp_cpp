#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"
#include "activation.h"
#include <cmath>
#include <stdexcept>

class Layer {
    private:
        Matrix weights;
        Matrix biases;
        Activation activation = Activation::NONE;
        int input_size, output_size;
        
    public:
        Layer(int input_size, int output_size)
        : weights(output_size, input_size, 0.1), biases(output_size, 1, 0.0),
        input_size(input_size), output_size(output_size) {}

        int get_input_size() const {return input_size;}
        int get_output_size() const {return output_size;}
        Matrix get_weights() const {return weights;}

        std::string get_activation() const {
            switch (activation)
            {
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

        Matrix forward(const Matrix& input) {
            // z = Wx + b
            Matrix z = weights*input + biases;
            return applyActivation(z);
        }

        void set_activation(const Activation& input) {
            activation = input;
        }

        void set_weights(const Matrix& input) {
            weights = input;
        }

        Matrix activation_grad(Matrix& layer_output) {
            return applyActivation(layer_output, true);
        }

    private:
        Matrix applyActivation(Matrix& z, bool grad = false) {
            for (int i = 0; i < z.get_rows(); i++) {
                for (int j = 0; j < z.get_cols(); j++) {
                    z(i,j) = activator(z(i,j), grad);
                }
            }
            return z;
        }

        double activator(double& x, bool grad) {
            if (activation == Activation::RELU) {
                if (grad == false) {x=relu(x);}
                else {x=relu_grad(x);}
            }
            else if (activation == Activation::SIGMOID) {
                if (grad == false) {x=sigmoid(x);}
                else {x=sigmoid_grad(x);}
            }
            return x;
        }

        //need to make this modular with classes

        double relu(double& x) {
            return std::max(0.0,x);
        }

        double relu_grad(double& x) {
            return x > 0 ? 1 : 0;
        }

        double sigmoid(double& x) {
            return 1.0/(1.0+std::exp(-x));
        }

        double sigmoid_grad(double& x) {
            return sigmoid(x)*(1.0-sigmoid(x));
        }
};

#endif
