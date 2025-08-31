CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

SOURCES = matrix.cpp layer.cpp loss.cpp mlp.cpp
OBJECTS = $(SOURCES:.cpp=.o)

all: xor_demo mnist_demo

xor_demo: $(OBJECTS) XOR.cpp
	$(CXX) $(CXXFLAGS) -o xor_demo XOR.cpp $(OBJECTS)

mnist_demo: $(OBJECTS) mnist_loader.o mnist_demo.cpp
	$(CXX) $(CXXFLAGS) -o mnist_demo mnist_demo.cpp mnist_loader.o $(OBJECTS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o xor_demo mnist_demo

download-mnist:
	./download_mnist.sh

.PHONY: all clean download-mnist
