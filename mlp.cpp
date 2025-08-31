#include "mlp.h"

MLP::MLP(const std::vector<int>& layer_sizes, 
         std::unique_ptr<Loss> loss_function, 
         double lr,
         const std::vector<std::string>& activations) 
    : loss_function(std::move(loss_function)), learning_rate(lr) {
    
    for (size_t i = 1; i < layer_sizes.size(); i++) {
        layers.emplace_back(layer_sizes[i-1], layer_sizes[i]);
    }
    
    if (!activations.empty()) {
        set_activations(activations);
    }
}

Matrix MLP::predict(const Matrix& input) {
    Matrix output = input;
    for (auto& layer : layers) {
        output = layer.forward(output);
    }
    return output;
}

Matrix MLP::forward(const Matrix& input) {
    layer_outputs.clear();
    Matrix output = input;
    layer_outputs.push_back(output);
    
    for (auto& layer : layers) {
        output = layer.forward(output);
        layer_outputs.push_back(output);
    }
    return output;
}

void MLP::backward() {
    if (layer_outputs.size() == 0) {
        throw std::invalid_argument("forward likely not complete, no layer outputs stored");
    }
    
    std::vector<Matrix> layer_gradients(layers.size());
    
    layer_gradients[layers.size()-1] = 
        layers[layers.size()-1].activation_grad(layer_outputs[layers.size()])
        .hadamard_product(loss_function->gradient(layer_outputs[layers.size()], target));
    
    for (int i = layers.size()-2; i >= 0; i--) {
        layer_gradients[i] = 
            (layers[i+1].get_weights().T() * layer_gradients[i+1])
            .hadamard_product(layers[i].activation_grad(layer_outputs[i+1]));
    }
    
    for (size_t i = 0; i < layers.size(); i++) {
        Matrix activation_prev = layer_outputs[i];
        
        Matrix new_weights = layers[i].get_weights() - 
            (layer_gradients[i] * activation_prev.T()) * learning_rate;
        layers[i].set_weights(new_weights);
        
        Matrix new_biases = layers[i].get_biases() - layer_gradients[i] * learning_rate;
        layers[i].set_biases(new_biases);
    }
    
    layer_outputs.clear();
}

double MLP::compute_loss(const Matrix& output, const Matrix& target) {
    return loss_function->compute(output, target);
}

Activation MLP::string_to_activation(const std::string& input) {
    if (input == "relu") return Activation::RELU;
    if (input == "sigmoid") return Activation::SIGMOID;
    if (input == "" || input == "none") return Activation::NONE;
    throw std::invalid_argument("Invalid activation function: " + input);
}

void MLP::set_activations(const std::vector<std::string>& activations) {
    if (activations.size() != layers.size()) {
        throw std::invalid_argument("Mismatch between activations and layers: " +
                                    std::to_string(activations.size()) + " vs " +
                                    std::to_string(layers.size()));
    }

    for (size_t i = 0; i < activations.size(); i++) {
        layers[i].set_activation(string_to_activation(activations[i]));
    }
}

std::vector<Layer> MLP::get_layers() {
    return layers;
}

std::vector<Matrix> MLP::get_layer_outputs() {
    return layer_outputs;
}

void MLP::set_target(const Matrix& input) {
    target = input;
}