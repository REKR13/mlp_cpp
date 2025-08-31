# Multi-Layer Perceptron (MLP) in C++

A from-scratch implementation of a Multi-Layer Perceptron neural network in C++. This project demonstrates core neural network concepts including forward propagation, backpropagation, and gradient descent optimization.

## Features

- **Custom Matrix Class**: Complete linear algebra operations with operator overloading
- **Flexible Architecture**: Support for arbitrary network topologies
- **Multiple Activations**: ReLU, Sigmoid, and linear activation functions
- **Loss Functions**: Mean Squared Error and Binary Cross Entropy
- **He Weight Initialization**: Proper weight initialization for stable training
- **Clean C++ Design**: Separated headers and implementation files

## Functionality

### XOR Learning
The network successfully learns the XOR function, demonstrating its ability to solve non-linear classification problems:

```
XOR Truth Table Results:
Input (0,0) → Output: ~0.005 (Target: 0)
Input (0,1) → Output: ~0.992 (Target: 1)  
Input (1,0) → Output: ~0.992 (Target: 1)
Input (1,1) → Output: ~0.012 (Target: 0)
```

### MNIST Digit Recognition
The network can also learn to classify handwritten digits from the MNIST dataset, achieving competitive accuracy on this standard machine learning benchmark.

## Project Structure

```
mlp_cpp/
├── matrix.h/.cpp        # Matrix operations and linear algebra
├── layer.h/.cpp         # Neural network layer implementation
├── loss.h/.cpp          # Loss functions (MSE, Binary Cross Entropy)
├── mlp.h/.cpp           # Main MLP class with training logic
├── mnist_loader.h/.cpp  # MNIST dataset loader
├── activation.h         # Activation function enumerations
├── XOR.cpp             # XOR learning demonstration
├── mnist_demo.cpp       # MNIST digit recognition demo
├── download_mnist.sh    # Script to download MNIST data
├── Makefile            # Build configuration
└── README.md           # Documentation
```

## Quick Start

### Build

```bash
make                    # Build both demos
make xor_demo          # Build XOR demo only
make mnist_demo        # Build MNIST demo only
```

### Run XOR Demo

```bash
./xor_demo
```

### Run MNIST Demo

First download the MNIST dataset:
```bash
make download-mnist     # Download MNIST data automatically
```

Or manually download from [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/) and extract the files to the project directory.

Then run the demo:
```bash
./mnist_demo
```

## Usage Example

```cpp
#include "mlp.h"

// Define network: 2 inputs → 6 hidden → 1 output
std::vector<int> layer_sizes = {2, 6, 1};
std::vector<std::string> activations = {"sigmoid", "sigmoid"};

// Create network with MSE loss
std::unique_ptr<Loss> mse = std::make_unique<MeanSquaredError>();
MLP network(layer_sizes, std::move(mse), 1.0, activations);

// Training data for XOR
std::vector<Matrix> inputs = {
    Matrix(2, 1).array_set({0, 0}),
    Matrix(2, 1).array_set({0, 1}),
    Matrix(2, 1).array_set({1, 0}),
    Matrix(2, 1).array_set({1, 1})
};

std::vector<Matrix> targets = {
    Matrix(1, 1, 0), Matrix(1, 1, 1), 
    Matrix(1, 1, 1), Matrix(1, 1, 0)
};

// Train the network
for (int epoch = 0; epoch < 15000; epoch++) {
    for (size_t i = 0; i < inputs.size(); i++) {
        network.set_target(targets[i]);
        network.forward(inputs[i]);
        network.backward();
    }
}
```

## Architecture

### Matrix Class
Provides essential linear algebra operations with bounds checking and intuitive operator overloading for mathematical expressions.

### Layer Class
Implements individual neural network layers with configurable activation functions and proper gradient computation.

### MLP Class
Orchestrates the complete neural network with automatic backpropagation and support for different loss functions.

## Mathematical Foundation

**Forward Propagation:**
```
z = Wx + b
a = activation(z)
```

**Backpropagation:**
```
∂L/∂W = ∂L/∂a × ∂a/∂z × ∂z/∂W
W := W - α∇W
```

## Requirements

- C++17 compatible compiler
- Standard library only (no external dependencies)

## Performance

- **XOR Problem**: Converges with loss < 0.0001 in ~15,000 epochs
- **MNIST Classification**: Achieves 85-90% accuracy on test set with basic architecture
- **Training Speed**: Processes thousands of samples per second on modern hardware

---

*A complete neural network implementation showcasing fundamental machine learning concepts in modern C++.*