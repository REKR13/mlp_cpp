#include "matrix.h"
#include <cmath>

class Layer {
    private:
        Matrix weights;
        Matrix biases;
        std::string activation;
    public:
        Layer(int input_size, int output_size)
        : weights(output_size, input_size, 0.1), biases(input_size, 1, 0.0) {}

        Matrix forward(const Matrix& input) {
            // z = Wx + b
            Matrix z = weights*input + biases;
            return applyActivation(z);
        }

        Matrix applyActivation(Matrix& z) {
            for (int i = 0; i < z.get_rows(); i++) {
                for (int j = 0; j < z.get_cols(); j++) {
                    z(i,j) = activator(z(i,j));
                }
            }
            return z;
        }

        double activator(double& x) {
            if (activation == "relu") {
                x = relu(x);
            }
            else if (activation == "sigmoid") {
                x = relu(x);
            }
            return x;
        }

        double relu(double x) {
            return std::max(0.0,x);
        }

        double sigmoid(double x) {
            return 1.0/(1.0+std::exp(-x));
        }



        


};