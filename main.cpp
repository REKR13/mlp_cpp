#include <iostream>
#include <vector>
#include "matrix.h"
#include "layer.h"
#include "mlp.h"

int main() {
    std::cout << "Hello World" << std::endl;


    /*
    std::vector<int> layer_sizes = {1,3,3,1};
    std::vector<std::string> activations = {"relu","relu","sigmoid"};
    Matrix input_matrix(1,1,2);
    MLP mlp(layer_sizes);

    std::cout << "Layers created" << std::endl;

    mlp.set_activations(activations);

    std::vector<Layer> layers = mlp.get_layers(); // creating one fewer layer than what is put into layer_sizes
    std::cout << layers.size() << "\n";

    for (auto& layer : layers) {
        std::cout << "Input size: " << layer.get_input_size() << " " << "Output size: " << layer.get_output_size() << "\n";
        std::cout << layer.get_activation() << "\n";
    }

    Matrix output = mlp.forward(input_matrix);
    output.print();
    */
    return 0;
}

