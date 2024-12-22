#include "layer.h"
#include <vector>

class MLP {
    private:
        std::vector<Layer> layers;
    public:
        MLP(const std::vector<Layer>& layer_sizes) {
            for (int i = 1; i < layer_sizes.size(); i++) {
                layers.emplace_back(layer_sizes[i-1], layer_sizes[i]);
            }
        }

        Matrix forward(const Matrix& input) {
            Matrix output = input;
            for (auto& layer : layers) {
                output = layer.forward(output);
            }
            return output;
        }
};