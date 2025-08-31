#ifndef MLP_H
#define MLP_H

#include "layer_new.h"
#include "activation.h"
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#include "loss_new.h"

class MLP {
private:
    std::vector<Layer> layers;
    std::vector<Matrix> layer_outputs;
    std::unique_ptr<Loss> loss_function;
    double learning_rate;
    Matrix target;
    
    Activation string_to_activation(const std::string& input);

public:
    explicit MLP(const std::vector<int>& layer_sizes, 
                 std::unique_ptr<Loss> loss_function, 
                 double lr = 0.01,
                 const std::vector<std::string>& activations = {});
    
    Matrix predict(const Matrix& input);
    Matrix forward(const Matrix& input);
    void backward();
    double compute_loss(const Matrix& output, const Matrix& target);
    
    void set_activations(const std::vector<std::string>& activations);
    std::vector<Layer> get_layers();
    std::vector<Matrix> get_layer_outputs();
    void set_target(const Matrix& input);
};

#endif