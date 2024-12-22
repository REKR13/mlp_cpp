#include "matrix.h"

class Layer {
    private:
        Matrix weights;
        Matrix biases;
    public:
        Layer(int input_size, int output_size)
        : weights(output_size, input_size, 0.1), biases(input_size, 1, 0.0) {}

        Matrix forward(const Matrix& input) {
            // z = Wx + b
            Matrix z = weights*input + biases;
            return z; // apply activation separately
        }
};