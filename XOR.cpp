#include <iostream>
#include <vector>
#include "loss.h"
#include "matrix.h"
#include "mlp.h"

int main() {
    std::vector<Matrix> inputs = {
        Matrix(2, 1).array_set({0, 0}),  // (0, 0)
        Matrix(2, 1).array_set({0, 1}),  // (0, 1)
        Matrix(2, 1).array_set({1, 0}),  // (1, 0)
        Matrix(2, 1).array_set({1, 1})  // (1, 1)
    };

    std::vector<Matrix> targets = {
        Matrix(1, 1, 0),  // XOR(0, 0) = 0
        Matrix(1, 1, 1),  // XOR(0, 1) = 1
        Matrix(1, 1, 1),  // XOR(1, 0) = 1
        Matrix(1, 1, 0)  // XOR(1, 1) = 0
    };

    std::vector<int> layer_sizes = {2, 6, 1};

    std::vector<std::string> activations = {"relu","sigmoid"};
    
    std::unique_ptr<Loss> mse(new MeanSquaredError());
    double learning_rate = 0.1;
    int epochs = 15000;
    
    MLP mlp(layer_sizes, std::move(mse), learning_rate, activations);

    std::cout << "Training XOR neural network..." << std::endl;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        
        for (size_t i = 0; i < inputs.size(); i++) {
            mlp.set_target(targets[i]);
            Matrix output = mlp.forward(inputs[i]);
            double loss = mlp.compute_loss(output, targets[i]);
            total_loss += loss;
            mlp.backward();
        }
        
        if (epoch % 1000 == 0) {
            std::cout << "Epoch: " << epoch << ", Loss: " << total_loss / inputs.size() << std::endl;
        }
    }

    std::cout << "\nTesting XOR network:" << std::endl;
    for (size_t i = 0; i < inputs.size(); i++) {
        Matrix output = mlp.predict(inputs[i]);
        std::cout << "Input: (" << inputs[i](0,0) << ", " << inputs[i](1,0) 
                  << ") -> Output: " << output(0,0) << " (Target: " << targets[i](0,0) << ")" << std::endl;
    }

    return 0;
}
