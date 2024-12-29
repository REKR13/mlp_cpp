#ifndef MLP_H
#define MLP_H

#include "layer.h"
#include "activation.h"
#include <memory>
#include <stdexcept>
#include <vector>
#include "loss.h"

class MLP {
    private:
        std::vector<Layer> layers;
        std::vector<Matrix> layer_outputs;
        std::unique_ptr<Loss> loss_function;
    public:
        MLP(const std::vector<int>& layer_sizes) {
            for (int i = 1; i < layer_sizes.size(); i++) {
                layers.emplace_back(layer_sizes[i-1], layer_sizes[i]); // creating N-1 neurons
            }
        }

        void forward(const Matrix& input) {
            Matrix output = input;
            for (auto& layer : layers) {
                output = layer.forward(output);
                layer_outputs.push_back(output);
            }
            //return output;
        }

        void backward() {
            if (layer_outputs.size() == 0) {
                throw std::invalid_argument("forward likely not complete, no layer outputs stored");
            }
            std::vector<Matrix> layer_gradients(layers.size());
            //TODO:Replace TARGET with actual target which needs to be implemented
            Matrix TARGET = Matrix(5,1,0);
            layer_gradients[layers.size()-1] = layers[layers.size()-1].activation_grad(layer_outputs(layers.size()-1)) * 
                                                loss_function.gradient(layer_outputs[layers.size()-1], TARGET);
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

        void set_loss(std::unique_ptr<Loss> loss) {
            loss_function = std::move(loss); //have to use set_loss(std::make_unique<MSE>())
        }

};

#endif
