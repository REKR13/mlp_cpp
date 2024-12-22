#include <iostream>
#include <vector>
#include "matrix.h"
#include "layer.h"
#include "mlp.h"

int main() {
    std::cout << "Hello World" << std::endl;

    std::vector<int> layer_sizes = {1,3,1};
    MLP mlp(layer_sizes);

    std::cout << "Layers created" << std::endl;

    std::vector<Layer> layers = mlp.get_layers();

    for (auto& layer : layers) {
        
    }

    return 0;
}
