#include "mlp.h"
#include "mnist_loader.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <random>
#include <iomanip>
#include <numeric>

int main() {
    try {
        std::cout << "=== MNIST Neural Network Demo ===" << std::endl;
        
        // Try to load MNIST data
        MNISTData train_data, test_data;
        try {
            std::cout << "Loading MNIST data..." << std::endl;
            train_data = MNISTLoader::load_training_data("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
            test_data = MNISTLoader::load_test_data("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
        } catch (const std::exception& e) {
            std::cout << "Error loading MNIST data: " << e.what() << std::endl;
            MNISTLoader::download_mnist_data();
            return 1;
        }
        
        // Use smaller subset for faster training (optional)
        size_t train_samples = std::min(train_data.size(), size_t(10000));
        size_t test_samples = std::min(test_data.size(), size_t(1000));
        
        std::cout << "Using " << train_samples << " training samples and " << test_samples << " test samples" << std::endl;
        
        // Network architecture: 784 -> 128 -> 64 -> 10
        std::vector<int> layer_sizes = {784, 128, 64, 10};
        std::vector<std::string> activations = {"relu", "relu", "sigmoid"};
        
        std::unique_ptr<Loss> mse = std::make_unique<MeanSquaredError>();
        double learning_rate = 0.01;
        
        std::cout << "Creating neural network: ";
        for (size_t i = 0; i < layer_sizes.size(); i++) {
            std::cout << layer_sizes[i];
            if (i < layer_sizes.size() - 1) std::cout << " -> ";
        }
        std::cout << std::endl;
        
        MLP network(layer_sizes, std::move(mse), learning_rate, activations);
        
        // Training parameters
        int epochs = 10;
        int batch_size = 32;
        
        std::cout << "\nStarting training..." << std::endl;
        std::cout << "Epochs: " << epochs << ", Batch size: " << batch_size << ", Learning rate: " << learning_rate << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Training loop
        for (int epoch = 0; epoch < epochs; epoch++) {
            double total_loss = 0.0;
            int batches = 0;
            
            // Shuffle training data
            std::vector<size_t> indices(train_samples);
            std::iota(indices.begin(), indices.end(), 0);
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices.begin(), indices.end(), g);
            
            // Mini-batch training
            for (size_t i = 0; i < train_samples; i += batch_size) {
                double batch_loss = 0.0;
                size_t batch_end = std::min(i + batch_size, train_samples);
                
                for (size_t j = i; j < batch_end; j++) {
                    size_t idx = indices[j];
                    network.set_target(train_data.labels[idx]);
                    Matrix output = network.forward(train_data.images[idx]);
                    double loss = network.compute_loss(output, train_data.labels[idx]);
                    batch_loss += loss;
                    network.backward();
                }
                
                total_loss += batch_loss;
                batches++;
                
                // Progress indicator
                if (batches % 100 == 0) {
                    double progress = double(i) / train_samples * 100;
                    std::cout << "\rEpoch " << (epoch + 1) << "/" << epochs 
                             << " - Progress: " << std::fixed << std::setprecision(1) << progress << "%" << std::flush;
                }
            }
            
            double avg_loss = total_loss / train_samples;
            std::cout << "\rEpoch " << (epoch + 1) << "/" << epochs 
                     << " - Loss: " << std::fixed << std::setprecision(6) << avg_loss << std::endl;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        std::cout << "Training completed in " << duration.count() << " seconds" << std::endl;
        
        // Test the network
        std::cout << "\nEvaluating on test set..." << std::endl;
        int correct = 0;
        
        for (size_t i = 0; i < test_samples; i++) {
            Matrix prediction = network.predict(test_data.images[i]);
            
            // Find predicted class (max output)
            int predicted_class = 0;
            double max_val = prediction(0, 0);
            for (int j = 1; j < 10; j++) {
                if (prediction(j, 0) > max_val) {
                    max_val = prediction(j, 0);
                    predicted_class = j;
                }
            }
            
            // Find actual class (one-hot encoded)
            int actual_class = 0;
            for (int j = 0; j < 10; j++) {
                if (test_data.labels[i](j, 0) > 0.5) {
                    actual_class = j;
                    break;
                }
            }
            
            if (predicted_class == actual_class) correct++;
            
            // Show progress
            if ((i + 1) % 200 == 0) {
                std::cout << "\rTested " << (i + 1) << "/" << test_samples << " samples" << std::flush;
            }
        }
        
        double accuracy = double(correct) / test_samples * 100.0;
        std::cout << "\rTest Accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%" 
                 << " (" << correct << "/" << test_samples << ")" << std::endl;
        
        // Show some example predictions
        std::cout << "\nSample predictions:" << std::endl;
        for (int i = 0; i < std::min(10, static_cast<int>(test_samples)); i++) {
            Matrix prediction = network.predict(test_data.images[i]);
            
            int predicted_class = 0;
            double max_val = prediction(0, 0);
            for (int j = 1; j < 10; j++) {
                if (prediction(j, 0) > max_val) {
                    max_val = prediction(j, 0);
                    predicted_class = j;
                }
            }
            
            int actual_class = 0;
            for (int j = 0; j < 10; j++) {
                if (test_data.labels[i](j, 0) > 0.5) {
                    actual_class = j;
                    break;
                }
            }
            
            std::cout << "Sample " << i << ": Predicted " << predicted_class 
                     << ", Actual " << actual_class 
                     << " (confidence: " << std::fixed << std::setprecision(3) << max_val << ")"
                     << (predicted_class == actual_class ? " ✓" : " ✗") << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
