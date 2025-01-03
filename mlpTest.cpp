#include <iostream>
#include <vector>
#include "matrix.h"
#include "layer.h"
#include "mlp.h"

int main() {
    // Define the XOR data
    std::vector<Matrix> inputs = {
        Matrix{2, 1, 0},  // (0, 0)
        Matrix{2, 1, 1}.array_set({0,1}),  // (0, 1)
        Matrix{2, 1, 1}.array_set({1,0}),  // (1, 0)
        Matrix{2, 1, 1}   // (1, 1)
    };

    std::vector<Matrix> targets = {
        Matrix{2, 1, 0},  // output for (0, 0)
        Matrix{2, 1, 1}.array_set({0,1}),  // output for (0, 1)
        Matrix{2, 1, 1}.array_set({1,0}),  // output for (1, 0)
        Matrix{2, 1, 1}   // output for (1, 1)
    };

    // Define your network architecture
    std::vector<int> layer_sizes = {2, 4, 1};  // 2 inputs, 1 output, hidden layer with 4 neurons
    std::vector<std::string> activations = {"relu", "sigmoid"};  // Use ReLU for hidden layer, Sigmoid for output layer

    std::unique_ptr<Loss> mse = std::make_unique<MeanSquaredError>();  // Mean Squared Error loss

    int epochs = 10000;
    double learning_rate = 0.01;

    MLP mlp(layer_sizes, std::move(mse), learning_rate);

    std::cout << "Made it to training loop" << "\n";

    // Training loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        // Training over the XOR data
        for (size_t i = 0; i < inputs.size(); i++) {
            const Matrix& input = inputs[i];
            const Matrix& target = targets[i];

            mlp.set_target(target);

            Matrix output = mlp.forward(input);
            std::cout << "Made it to loss" << "\n";
            output.shape();
            target.shape();
            double loss = mlp.compute_loss(output, target);
            total_loss += loss;

            std::cout << "Made it to backward with loss of " << loss << "\n";
            mlp.backward();
        }

        // Print loss every 1000 epochs
        if (epoch % 1000 == 0) {
            std::cout << "Epoch: " << epoch << ", Loss: " << total_loss / inputs.size() << "\n";
        }
    }

    // Test the model after training
    std::cout<< "Testing model: " << "\n";
    for (size_t i = 0; i < inputs.size(); i++) {
        const Matrix& input = inputs[i];
        Matrix output = mlp.predict(input);
        std::cout << "Input: " << "\n";
        input.print();
        std::cout << "Predicted: " << "\n";
        output.print();
        std::cout << "Target: " << "\n";
        targets[i].print();
    }

    return 0;
}