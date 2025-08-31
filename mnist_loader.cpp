#include "mnist_loader.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>

MNISTData MNISTLoader::load_training_data(const std::string& images_path, 
                                          const std::string& labels_path) {
    MNISTData data;
    data.images = read_images(images_path);
    data.labels = read_labels(labels_path);
    
    if (data.images.size() != data.labels.size()) {
        throw std::runtime_error("Mismatch between number of images and labels");
    }
    
    std::cout << "Loaded " << data.size() << " training samples" << std::endl;
    return data;
}

MNISTData MNISTLoader::load_test_data(const std::string& images_path, 
                                      const std::string& labels_path) {
    MNISTData data;
    data.images = read_images(images_path);
    data.labels = read_labels(labels_path);
    
    if (data.images.size() != data.labels.size()) {
        throw std::runtime_error("Mismatch between number of images and labels");
    }
    
    std::cout << "Loaded " << data.size() << " test samples" << std::endl;
    return data;
}

std::vector<Matrix> MNISTLoader::read_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open images file: " + path);
    }
    
    uint32_t magic = read_uint32(file);
    if (magic != 0x00000803) {
        throw std::runtime_error("Invalid magic number in images file");
    }
    
    uint32_t num_images = read_uint32(file);
    uint32_t num_rows = read_uint32(file);
    uint32_t num_cols = read_uint32(file);
    
    std::cout << "Reading " << num_images << " images of size " << num_rows << "x" << num_cols << std::endl;
    
    std::vector<Matrix> images;
    images.reserve(num_images);
    
    for (uint32_t i = 0; i < num_images; i++) {
        std::vector<uint8_t> pixels(num_rows * num_cols);
        file.read(reinterpret_cast<char*>(pixels.data()), pixels.size());
        
        if (file.gcount() != static_cast<std::streamsize>(pixels.size())) {
            throw std::runtime_error("Error reading image data");
        }
        
        images.push_back(normalize_image(pixels));
    }
    
    return images;
}

std::vector<Matrix> MNISTLoader::read_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open labels file: " + path);
    }
    
    uint32_t magic = read_uint32(file);
    if (magic != 0x00000801) {
        throw std::runtime_error("Invalid magic number in labels file");
    }
    
    uint32_t num_labels = read_uint32(file);
    std::cout << "Reading " << num_labels << " labels" << std::endl;
    
    std::vector<Matrix> labels;
    labels.reserve(num_labels);
    
    for (uint32_t i = 0; i < num_labels; i++) {
        uint8_t label;
        file.read(reinterpret_cast<char*>(&label), 1);
        
        if (file.gcount() != 1) {
            throw std::runtime_error("Error reading label data");
        }
        
        labels.push_back(label_to_one_hot(label));
    }
    
    return labels;
}

uint32_t MNISTLoader::read_uint32(std::ifstream& file) {
    uint32_t value;
    file.read(reinterpret_cast<char*>(&value), 4);
    
    // Convert from big-endian to little-endian
    return ((value & 0xFF000000) >> 24) |
           ((value & 0x00FF0000) >> 8)  |
           ((value & 0x0000FF00) << 8)  |
           ((value & 0x000000FF) << 24);
}

Matrix MNISTLoader::normalize_image(const std::vector<uint8_t>& pixels) {
    Matrix image(784, 1);  // 28x28 = 784 pixels as column vector
    
    for (size_t i = 0; i < pixels.size(); i++) {
        image(i, 0) = static_cast<double>(pixels[i]) / 255.0;  // Normalize to [0,1]
    }
    
    return image;
}

Matrix MNISTLoader::label_to_one_hot(uint8_t label) {
    Matrix one_hot(10, 1, 0.0);  // 10 classes (0-9)
    one_hot(label, 0) = 1.0;
    return one_hot;
}

void MNISTLoader::download_mnist_data() {
    std::cout << "MNIST data files not found. Please download them from:" << std::endl;
    std::cout << "http://yann.lecun.com/exdb/mnist/" << std::endl;
    std::cout << std::endl;
    std::cout << "Required files:" << std::endl;
    std::cout << "- train-images-idx3-ubyte (Training images)" << std::endl;
    std::cout << "- train-labels-idx1-ubyte (Training labels)" << std::endl;
    std::cout << "- t10k-images-idx3-ubyte (Test images)" << std::endl;
    std::cout << "- t10k-labels-idx1-ubyte (Test labels)" << std::endl;
    std::cout << std::endl;
    std::cout << "Extract the gz files and place them in the project directory." << std::endl;
}