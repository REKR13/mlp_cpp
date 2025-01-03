#include <iostream>
#include <vector>
#include "matrix.h"
#include "layer.h"
#include "mlp.h"

int main() {
    
    ///*
    std::vector<Matrix> inputs;
    std::vector<Matrix> targets;

    for (int i = 1; i <= 100; i++) {
        inputs.push_back(Matrix(1, 1, i));          // Inputs from 1 to 100
        targets.push_back(Matrix(1, 1, 2 * i));    // Targets are double the input value
    }

    double learning_rate = 0.0000001;
    int epochs = 100000;

    std::vector<int> layer_sizes = {1,3,3,1};
    std::vector<std::string> activations = {"relu","relu","relu"};
    std::unique_ptr<Loss> mse = std::make_unique<MeanSquaredError>();
    MLP model(layer_sizes, std::move(mse), learning_rate);

    std::cout << "Made it to training loop" << "\n";

    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        for (size_t i = 0; i < inputs.size(); i++) {
            const Matrix& input = inputs[i];
            const Matrix& target = targets[i];

            model.set_target(target);

            Matrix output = model.forward(input);
            double loss = model.compute_loss(output, target);
            total_loss += loss;

            //std::cout << "Made it to backward with loss of " << loss << "\n";
            model.backward();
        }
        if (epoch % 1000 == 0) {
            std::cout<<"Epoch: " << epoch << ", Loss: " << total_loss / inputs.size() << "\n";
        }
    }

    std::cout<< "Testing model: " << "\n";
    for (size_t i = 0; i < inputs.size(); i++) {
        const Matrix& input = inputs[i];
        Matrix output = model.predict(input);
        std::cout << "Input: " << "\n";
        input.print();
        std::cout << "Predicted: " << "\n";
        output.print();
        std::cout << "Target: " << "\n";
        targets[i].print();
    }
    //*/

    /*
    Matrix input{1,1,3};
    Matrix output{1,1,10};

    std::vector<int> layer_sizes = {1,3,3,1};
    std::vector<std::string> activations = {"relu","relu","relu"};
    std::unique_ptr<Loss> mse = std::make_unique<MeanSquaredError>();
    MLP mlp(layer_sizes, std::move(mse));

    std::cout << mlp.get_layers().size() << "\n";

    for (int i = 0; i < 1000; i++) {
        mlp.forward(input);
        std::cout<<"forward complete" << "\n";
        std::cout << mlp.get_layer_outputs().size() << "\n";
        mlp.set_target(output);
        mlp.backward();
    }

    mlp.predict(input).print();
    */
    return 0;
}

