CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

SOURCES = matrix.cpp layer.cpp loss.cpp mlp.cpp
OBJECTS = $(SOURCES:.cpp=.o)

all: xor_demo

xor_demo: $(OBJECTS) xor.cpp
	$(CXX) $(CXXFLAGS) -o xor_demo xor.cpp $(OBJECTS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o xor_demo

.PHONY: all clean
