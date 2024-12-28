#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"
#include "activation.h"
#include <cmath>

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
                return "ACTIVATION: invalid";
            }
        }

        Matrix forward(const Matrix& input) {
            // z = Wx + b
            Matrix z = weights*input + biases;
            return applyActivation(z);
        }

        

        void set_activation(Activation input) {
            activation = input;
        }

    private:

        Matrix applyActivation(Matrix& z) {
            for (int i = 0; i < z.get_rows(); i++) {
                for (int j = 0; j < z.get_cols(); j++) {
                    z(i,j) = activator(z(i,j));
                }
            }
            return z;
        }

        double activator(double& x) {
            if (activation == Activation::RELU) {
                x = relu(x);
            }
            else if (activation == Activation::SIGMOID) {
                x = sigmoid(x);
            }
            return x;
        }

        double relu(double& x) {
            return std::max(0.0,x);
        }

        double sigmoid(double& x) {
            return 1.0/(1.0+std::exp(-x));
        }
};

#endif
