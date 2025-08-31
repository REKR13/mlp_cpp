#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include "matrix.h"
#include <vector>
#include <string>

struct MNISTData {
    std::vector<Matrix> images;
    std::vector<Matrix> labels;
    size_t size() const { return images.size(); }
};

class MNISTLoader {
public:
    static MNISTData load_training_data(const std::string& images_path, 
                                       const std::string& labels_path);
    static MNISTData load_test_data(const std::string& images_path, 
                                   const std::string& labels_path);
    static void download_mnist_data();
    
private:
    static std::vector<Matrix> read_images(const std::string& path);
    static std::vector<Matrix> read_labels(const std::string& path);
    static uint32_t read_uint32(std::ifstream& file);
    static Matrix normalize_image(const std::vector<uint8_t>& pixels);
    static Matrix label_to_one_hot(uint8_t label);
};

#endif