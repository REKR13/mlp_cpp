#ifndef MLP_H
#define MLP_H

#include "layer.h"
#include "activation.h"
#include <vector>

class MLP {
    private:
        std::vector<Layer> layers;
    public:
        MLP(const std::vector<int>& layer_sizes) {
            for (int i = 1; i < layer_sizes.size(); i++) {
                layers.emplace_back(layer_sizes[i-1], layer_sizes[i]); // creating N-1 neurons
            }
        }

        Matrix forward(const Matrix& input) {
            Matrix output = input;
            for (auto& layer : layers) {
                output = layer.forward(output);
            }
            return output;
        }

        Activation string_to_activation(const std::string& input) {
            if (input == "relu") return Activation::RELU;
            if (input == "sigmoid") return Activation::SIGMOID;
            ///if (input == "tanh") return Activation::TANH;
            if (input == "" || input == "none") return Activation::NONE;
            throw std::invalid_argument("Invalid activation function: " + input);
        }


        void set_activations(const std::vector<std::string>& activations) {
            if (activations.size() != layers.size()) {
                throw std::invalid_argument("Mismatch between activations and layers: " +
                                            std::to_string(activations.size()) + " vs " +
                                            std::to_string(layers.size()));
            }

            for (int i = 0; i < activations.size(); i++) {
                layers[i].set_activation(string_to_activation(activations[i]));
            }
        }

        std::vector<Layer> get_layers() {
            return layers;
        }
};

#endif
