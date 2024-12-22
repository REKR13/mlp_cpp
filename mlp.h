#include "layer.h"
#include "activation.h"
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

        void set_activations(const std::vector<Activation>& activations) {
            if (activations.size() != layers.size()) {
                throw std::invalid_argument("Mismatch between activations and layers: " +
                                std::to_string(activations.size()) + " vs " +
                                std::to_string(layers.size()));
            }
            for (int i = 0; i < activations.size(); i++) {
                layers[i].set_activation(activations[i]);
            }
        }
};