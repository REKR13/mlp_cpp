#ifndef MLP_H
#define MLP_H

#include "layer.h"
#include "activation.h"
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#include "loss.h"

class MLP {
    private:
    std::vector<Layer> layers;
    std::vector<Matrix> layer_outputs; // 1 longer than layers because it includes original input
    std::unique_ptr<Loss> loss_function;
    double learning_rate;
    Matrix target;

    public:
    explicit MLP(const std::vector<int>& layer_sizes, std::unique_ptr<Loss> loss_function, double lr = 0.01) : loss_function(std::move(loss_function)), learning_rate(lr) {
        for (int i = 1; i < layer_sizes.size(); i++) {
            layers.emplace_back(layer_sizes[i-1], layer_sizes[i]); // creating N-1 neurons
        }
    }

    Matrix predict(const Matrix& input) {
        Matrix output = input;
        for (auto& layer : layers) {
            output = layer.forward(output);
        }
        return output;
    }

    Matrix forward(const Matrix& input) {
        Matrix output = input;
        layer_outputs.push_back(output);
        for (auto& layer : layers) {
            output = layer.forward(output);
            layer_outputs.push_back(output);
        }
        return output;
    }

    void backward() {
        if (layer_outputs.size() == 0) {
            throw std::invalid_argument("forward likely not complete, no layer outputs stored");
        }
        std::vector<Matrix> layer_gradients(layers.size());
        
        // get gradient for the last layer
        /*
        std::cout << "Layer gradient shape: " << "\n"; 
        layers[layers.size()-1].activation_grad(layer_outputs[layers.size()]).shape();
        std::cout << "Loss grad shape: " << "\n"; 
        loss_function->gradient(layer_outputs[layers.size()], target).shape();
        */

        layer_gradients[layers.size()-1] = (layers[layers.size()-1].activation_grad(layer_outputs[layers.size()]))
                                            .hadamard_product(loss_function->gradient(layer_outputs[layers.size()], target));
        //std::cout << "Layer " << layers.size() << " grad complete" << "\n";
        // use gradient from last layer to move backwards getting the gradient for each layer
        for (int i = layers.size()-2; i >= 0; i--) {
            /*
            std::cout << "Weights shape: " << "\n"; 
            layers[i+1].get_weights().T().shape();
            std::cout << "Layer gradient shape: " << "\n"; 
            layer_gradients[i+1].shape();
            std::cout << "Activation grad shape: " << "\n"; 
            layers[i].activation_grad(layer_outputs[i+1]).shape();
            */

            layer_gradients[i] = (layers[i+1].get_weights().T()*layer_gradients[i+1]).hadamard_product(layers[i].activation_grad(layer_outputs[i+1]));
            //std::cout << "Layer " << i+1 << " grad complete" << "\n";

        }
        //update weights with SGD
        //TODO: make the optimizer a class like for loss so Adam and other optimizers can be implemented 
        for (int i = 0; i < layers.size(); i++) {
            // W_new = W_old - lr*weight_grad
            
            Matrix activation_prev = layer_outputs[i]; // not necessary with current output implementation

            Matrix new_weights = layers[i].get_weights() - (layer_gradients[i]*activation_prev.T()) * learning_rate;
            layers[i].set_weights(new_weights);
            Matrix new_biases = layers[i].get_biases() - layer_gradients[i]*learning_rate;
            layers[i].set_biases(new_biases);
            //std::cout<<"Layer " << i+1 << " weights and biases updated" << "\n";
        }

        // reset layer_outputs for next forward pass
        layer_outputs.clear();

    }

    double compute_loss(const Matrix& output, const Matrix& target) {
        return loss_function->compute(output, target);
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
    
    std::vector<Matrix> get_layer_outputs() {
        return layer_outputs;
    }

    void set_target(const Matrix& input) {
        target = input;
    }
};

#endif
